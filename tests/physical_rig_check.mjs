import assert from 'node:assert/strict';
import test from 'node:test';
import {readFile} from 'node:fs/promises';
import * as THREE from 'three';
import loadMujoco from '@mujoco/mujoco';
import {importVisualFiles,loadVisualScene,VisualBinding,rebuildPhysicalRig,validatePhysicalRig} from '../morpholab/static/scene-runtime.mjs';
import {helperHumanoidGLB,tinyRigGLB} from './scene-fixtures.mjs';
globalThis.ProgressEvent ||= class{constructor(type,options){Object.assign(this,options)}};
const mj=await loadMujoco();
const files=[{path:'helper-character.glb',data:helperHumanoidGLB()}];
const imported=await importVisualFiles(files);
function physics(body){const model=mj.MjModel.from_xml_string(body.xml),data=new mj.MjData(model);mj.mj_forward(model,data);return {model,data,close(){data.delete();model.delete();}};}
function frame({model,data}){return {bodies:Array.from({length:model.nbody-1},(_,i)=>{i++;return {name:mj.mj_id2name(model,mj.mjtObj.mjOBJ_BODY.value,i),pos:Array.from(data.xpos.slice(3*i,3*i+3)),quat:Array.from(data.xquat.slice(4*i,4*i+4))};})};}
function bones(scene){const map=new Map();scene.traverse(node=>{if(node.isBone)map.set(node.name,node);});return map;}
function vertices(scene){const values=[];scene.traverse(mesh=>{if(mesh.isSkinnedMesh)for(let i=0;i<mesh.geometry.attributes.position.count;i++)values.push(mesh.getVertexPosition(i,new THREE.Vector3()).applyMatrix4(mesh.matrixWorld));});return values;}

test('helper hierarchy becomes articulated legs without changing the source asset',()=>{
 const rig=imported.physical_rig,byName=new Map(rig.bodies.map(b=>[b.label,b]));
 assert.equal(rig.profile,'humanoid');assert.equal(byName.get('Pelvis').id,rig.root);
 assert.ok(!byName.has('Bone')&&!byName.has('PoleTargetL'));
 for(const side of ['L','R']){
  const shin=byName.get('LowerLeg'+side),foot=byName.get('Foot'+side);
  assert.equal(shin.joint.type,'hinge');assert.equal(foot.parent,shin.id);
  assert.equal(shin.colliders[0].type,'capsule');assert.ok(shin.colliders[0].size[1]>.1);
  assert.equal(foot.colliders[0].type,'box');
 }
 assert.deepEqual(Buffer.from(imported.visual.files[0].data,'base64'),Buffer.from(files[0].data));
 assert.ok(Math.abs(rig.bodies.reduce((sum,b)=>sum+b.mass,0)-10)<1e-8);
 assert.ok(new Set(rig.bodies.map(b=>b.mass.toFixed(3))).size>3,'Mass should follow fitted volume, not bone count');
 assert.ok(imported.metadata.rig_report.repairs.some(s=>s.includes('+X')));
});

test('bind pose, outer transforms and physical-to-visual offsets preserve neutral skin',async()=>{
 const loaded=await loadVisualScene(files),before=bones(loaded.scene);
 assert.ok(before.get('FootL').position.z>.3,'Fixture must start in an asymmetric posed frame');
 const binding=new VisualBinding(loaded.scene,imported.visual),env=physics(imported);
 try{
  const sourceBones=bones(loaded.scene);
  assert.ok(Math.abs(sourceBones.get('FootL').position.y-.08)<1e-6,'Inverse bind pose must restore ankle position');
  assert.equal(sourceBones.get('FootL').parent.name,'Bone','Original visual topology is preserved');
  const neutral=vertices(loaded.scene);binding.setFrame(frame(env));const matched=vertices(loaded.scene);
  neutral.forEach((p,i)=>assert.ok(p.distanceTo(matched[i])<2e-6,'Rest physics must reproduce the unchanged neutral mesh'));
  const restFoot=sourceBones.get('FootL').getWorldPosition(new THREE.Vector3());
  const shin=imported.physical_rig.bodies.find(b=>b.label==='LowerLegL');
  const id=mj.mj_name2id(env.model,mj.mjtObj.mjOBJ_JOINT.value,`joint_${shin.id}`);
  env.data.qpos[env.model.jnt_qposadr[id]]=.5;mj.mj_forward(env.model,env.data);binding.setFrame(frame(env));
  assert.ok(sourceBones.get('FootL').getWorldPosition(new THREE.Vector3()).distanceTo(restFoot)>.1,'Knee articulation must move the originally root-parented foot');
  assert.ok(vertices(loaded.scene).some((p,i)=>p.distanceTo(matched[i])>.1),'The original skin follows the new physical chain');
  for(let i=0;i<100;i++){env.data.ctrl.fill(.01);mj.mj_step(env.model,env.data);}
  assert.ok(env.data.qpos.every(Number.isFinite));
 }finally{binding.dispose();env.close();}
});

test('edited pivots/mass/limits rebuild physics while keeping neutral visual positions',async()=>{
 const edited=structuredClone(imported.physical_rig),shin=edited.bodies.find(b=>b.label==='LowerLegL');
 shin.position[2]+=.02;shin.mass=.7;shin.joint.torque=22;shin.joint.range=[-2,120];
 const body=rebuildPhysicalRig(imported,edited),env=physics(body),loaded=await loadVisualScene(files),binding=new VisualBinding(loaded.scene,body.visual);
 try{
  const neutral=vertices(loaded.scene);binding.setFrame(frame(env));vertices(loaded.scene).forEach((p,i)=>assert.ok(p.distanceTo(neutral[i])<2e-6));
  const id=mj.mj_name2id(env.model,mj.mjtObj.mjOBJ_BODY.value,shin.id);assert.ok(Math.abs(env.model.body_mass[id]-.7)<1e-7);
  assert.ok(body.xml.includes('gear="22 0 0 0 0 0"'));
  assert.equal(imported.physical_rig.bodies.find(b=>b.id===shin.id).joint.torque,60,'Rebuild must not mutate original project');
 }finally{binding.dispose();env.close();}
});

test('invalid topology/numbers fail before reaching the simulator',()=>{
 for(const alter of [r=>r.bodies[0].mass=NaN,r=>r.bodies[0].parent=r.bodies[0].id,r=>r.bodies[0].colliders[0].size[0]=-1,r=>r.bodies[0].quaternion=[0,0,0,0],r=>r.bodies[0].source_bones=r.bodies[1].source_bones]){
  const rig=structuredClone(imported.physical_rig);alter(rig);assert.throws(()=>validatePhysicalRig(rig));
 }
});

test('generic rigs keep structural zero-weight ancestors and do not invent humanoid repairs',async()=>{
 const body=await importVisualFiles([{path:'chain.glb',data:tinyRigGLB(8)}]);
 assert.equal(body.physical_rig.profile,'generic');assert.equal(body.physical_rig.bodies.length,8);
 assert.ok(body.physical_rig.warnings.some(w=>w.includes('Generic')));
 assert.ok(!body.physical_rig.repairs.some(w=>w.includes('knee')));
 const forced=await importVisualFiles(files,{profile:'generic'});assert.equal(forced.physical_rig.profile,'generic');
 await assert.rejects(importVisualFiles([{path:'chain.glb',data:tinyRigGLB()}],{profile:'humanoid'}),/confidently/);
});

test('Casual Female asset produces valid humanoid physics when available locally',{skip:!process.env.RLSIM_FEMALE_ASSET},async()=>{
 const local=[{path:'Casual_Female.glb',data:await readFile(process.env.RLSIM_FEMALE_ASSET)}];
 const body=await importVisualFiles(local),env=physics(body),loaded=await loadVisualScene(local),binding=new VisualBinding(loaded.scene,body.visual);
 try{
  assert.equal(body.physical_rig.profile,'humanoid');assert.equal(body.physical_rig.bodies.length,20);
  const head=body.physical_rig.bodies.find(b=>b.label==='Head');assert.ok(head.mass<1.3,'Hairstyle must not dominate mass');
  const root=body.physical_rig.bodies.find(b=>b.id===body.root_body);assert.ok(root.position[2]>.5);
  const neutral=vertices(loaded.scene);binding.setFrame(frame(env));const actual=vertices(loaded.scene);
  assert.ok(Math.max(...actual.map((p,i)=>p.distanceTo(neutral[i])))<1e-5);
  for(let i=0;i<100;i++){env.data.ctrl.fill(.01);mj.mj_step(env.model,env.data);}
  assert.ok(env.data.qpos.every(Number.isFinite));
 }finally{binding.dispose();env.close();}
});

test('legacy saved visual metadata still maps body transforms without rest offsets',async()=>{
 const local=[{path:'legacy.glb',data:tinyRigGLB()}],body=await importVisualFiles(local),loaded=await loadVisualScene(local);
 const legacy=structuredClone(body.visual);
 delete legacy.rest_locals;delete legacy.source_bindings;delete legacy.bone_offsets;
 legacy.root_body='legacy_root';delete legacy.bone_paths;
 const binding=new VisualBinding(loaded.scene,legacy);
 try{
  const initial=vertices(loaded.scene),states=binding.bones.map(({bone,bodyName})=>{
   const p=bone.getWorldPosition(new THREE.Vector3()),q=bone.getWorldQuaternion(new THREE.Quaternion());
   return {name:bodyName,pos:p.toArray(),quat:[q.w,q.x,q.y,q.z]};
  });
  const root=new THREE.Matrix4().fromArray(legacy.bind_root),p=new THREE.Vector3(),q=new THREE.Quaternion();root.decompose(p,q,new THREE.Vector3());
  states.push({name:'legacy_root',pos:p.toArray(),quat:[q.w,q.x,q.y,q.z]});
  binding.setFrame({bodies:states});vertices(loaded.scene).forEach((value,i)=>assert.ok(value.distanceTo(initial[i])<1e-6));
  states.forEach(state=>state.pos[0]+=.3);binding.setFrame({bodies:states});
  vertices(loaded.scene).forEach((value,i)=>assert.ok(value.distanceTo(initial[i].clone().add(new THREE.Vector3(.3,0,0)))<1e-6));
 }finally{binding.dispose();}
});
