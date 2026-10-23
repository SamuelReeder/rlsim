import * as THREE from 'three';
import {validatePhysicalRig, physicalRigXML, colliderVolume} from './physical-rig.mjs';
export {validatePhysicalRig, physicalRigXML};
const v = a=>new THREE.Vector3().fromArray(a);
const clamp=(x,a,b)=>Math.max(a,Math.min(b,x));
const qArray=q=>[q.w,q.x,q.y,q.z];
const matrix=b=>new THREE.Matrix4().compose(v(b.position),new THREE.Quaternion(b.quaternion[1],b.quaternion[2],b.quaternion[3],b.quaternion[0]),new THREE.Vector3(1,1,1));
export function bonePath(object,root){const path=[];for(let n=object;n!==root;n=n.parent){if(!n?.parent)throw new Error('Skeleton is outside its scene.');path.push(n.parent.children.indexOf(n));}return path.reverse().join('/');}
const depth=object=>{let n=0;for(;object.parent;object=object.parent)n++;return n;};

/** Three's inverse-bind convention supplies the anatomical rest pose. Keep the
 * non-bone scene ancestors (unit conversion/placement) outside that skeleton. */
export function recoverRestPose(scene){
  scene.updateMatrixWorld(true);
  const inverses=new Map(),warnings=[];
  scene.traverse(mesh=>{if(!mesh.isSkinnedMesh)return;
    mesh.skeleton.bones.forEach((bone,i)=>{
      const inverse=mesh.skeleton.boneInverses[i];
      if(!inverse||!inverse.elements.every(Number.isFinite)||Math.abs(inverse.determinant())<1e-12)throw new Error('The source has invalid inverse-bind matrices. Repair its skin before importing.');
      const previous=inverses.get(bone);
      if(!bone.userData.rlsim_rest_pose&&previous&&previous.elements.some((x,j)=>Math.abs(x-inverse.elements[j])>1e-4))throw new Error('Shared bones have inconsistent bind poses across meshes. Export a single consistent deformation rig.');
      inverses.set(bone,inverse.clone());
    });
  });
  // This is Skeleton.pose(), applied once across shared skeletons. Saving local
  // transforms also lets the reloaded original asset reproduce this rest pose.
  // Preservation exports keep original mesh-local coordinates, so each mesh
  // can have a different bind-shape matrix. Their added bones explicitly store
  // the rest pose; inverting their IBMs alone would discard that mesh transform.
  const bindWorld=new Map([...inverses].map(([bone,inverse])=>[bone,bone.userData.rlsim_rest_pose?bone.matrixWorld.clone():inverse.clone().invert()]));
  for(const [bone,world] of [...bindWorld].sort((a,b)=>depth(a[0])-depth(b[0]))){
    const parent=bindWorld.get(bone.parent);
    const local=parent?parent.clone().invert().multiply(world):world;
    local.decompose(bone.position,bone.quaternion,bone.scale);bone.updateMatrix();
  }
  scene.updateMatrixWorld(true);
  const locals={};scene.traverse(bone=>{if(bone.isBone)locals[bonePath(bone,scene)]={position:bone.position.toArray(),quaternion:bone.quaternion.toArray(),scale:bone.scale.toArray()};});
  scene.traverse(mesh=>{if(mesh.isSkinnedMesh)mesh.skeleton.update();});
  return {locals,warnings};
}
function roleOf(name){
  let s=name.toLowerCase().replace(/^mixamorig[:_]?/,'').replace(/[^a-z0-9]/g,'');
  let side='';if(/left/.test(s)){side='L';s=s.replace('left','');}else if(/right/.test(s)){side='R';s=s.replace('right','');}
  else if(/[lr]$/.test(s)){side=s.at(-1).toUpperCase();s=s.slice(0,-1);}else if(/^[lr](upper|lower|thigh|calf|foot|hand|arm|leg)/.test(s)){side=s[0].toUpperCase();s=s.slice(1);}
  if(!side)return null;
  const segment=/upperleg|upleg|thigh/.test(s)?'thigh':/lowerleg|shin|calf/.test(s)?'shin':/^(foot|ankle)$/.test(s)?'foot':/upperarm|^arm$/.test(s)?'upperarm':/lowerarm|forearm/.test(s)?'forearm':/hand|fist/.test(s)?'hand':null;
  return segment?segment+side:null;
}
function pointClouds(scene,bones){
  const clouds=new Map(bones.map(bone=>[bone,new Map()])),weights=new Map(bones.map(bone=>[bone,0]));
  const point=new THREE.Vector3();
  scene.traverse(mesh=>{
    if(!mesh.isSkinnedMesh)return;
    const positions=mesh.geometry.attributes.position,indices=mesh.geometry.attributes.skinIndex,skin=mesh.geometry.attributes.skinWeight;
    if(!positions||!indices||!skin)return;
    const stride=Math.max(1,Math.ceil(positions.count/100000));
    for(let i=0;i<positions.count;i+=stride){
      mesh.getVertexPosition(i,point).applyMatrix4(mesh.matrixWorld);
      if(!point.toArray().every(Number.isFinite))throw new Error('Skin deformation produced invalid vertices.');
      const key=point.toArray().map(x=>Math.round(x*1e5)).join(',');
      for(let k=0;k<4;k++){
        const weight=skin.getComponent(i,k),bone=mesh.skeleton.bones[indices.getComponent(i,k)];
        if(weight<.05||!clouds.has(bone))continue;
        weights.set(bone,weights.get(bone)+weight);
        // Attribute seams and separate material primitives must not count as
        // extra mass or bias geometric bounds toward dense tessellation.
        clouds.get(bone).set(key,point.clone());
      }
    }
  });
  return {clouds:new Map([...clouds].map(([bone,map])=>[bone,[...map.values()]])),weights};
}
function commonAncestor(a,b,parents){const ancestors=new Set();for(let n=a;n;n=parents.get(n))ancestors.add(n);for(let n=b;n;n=parents.get(n))if(ancestors.has(n))return n;return null;}
function collider(type,position,size,quaternion=[1,0,0,0]){return {type,position:position.toArray(),quaternion,size};}
function fitColliders(bone,physicalChildren,clouds,positions,height,role){
  const origin=positions.get(bone),points=clouds.get(bone)||[],minimum=height*.006;
  if(role==='head'&&points.length){
    // Long hair follows the head skin but is not a solid block of skull mass.
    // Fit a conservative head core; keep the complete hairstyle visually.
    const box=new THREE.Box3().setFromPoints(points),half=box.getSize(new THREE.Vector3()).multiplyScalar(.5);
    const radius=clamp(Math.min(half.x,half.y,half.z)*.65,height*.045,height*.085);
    return [collider('sphere',new THREE.Vector3(0,0,radius*.75),[radius])];
  }
  if(role==='core'&&points.length){
    const sorted=[0,1,2].map(axis=>points.map(point=>point.getComponent(axis)).sort((a,b)=>a-b));
    const low=sorted.map(a=>a[Math.floor((a.length-1)*.05)]),high=sorted.map(a=>a[Math.ceil((a.length-1)*.95)]);
    const center=v(low).add(v(high)).multiplyScalar(.5).sub(origin),half=v(high).sub(v(low)).multiplyScalar(.5);
    half.set(Math.max(half.x,minimum),Math.max(half.y,minimum),Math.max(half.z,minimum));
    return [collider('box',center,half.toArray())];
  }
  if(role?.startsWith('foot')||(!physicalChildren.length&&points.length)){
    const box=new THREE.Box3().setFromPoints(points.length?points:[origin]),center=box.getCenter(new THREE.Vector3()).sub(origin),half=box.getSize(new THREE.Vector3()).multiplyScalar(.5);
    half.set(Math.max(half.x,minimum),Math.max(half.y,minimum),Math.max(half.z,minimum));
    return [collider('box',center,half.toArray())];
  }
  const segments=physicalChildren.map(child=>positions.get(child).clone().sub(origin)).filter(end=>end.length()>height*.002);
  if(!segments.length)return [collider('sphere',new THREE.Vector3(),[minimum*1.5])];
  return segments.map(end=>{
    const length=end.length(),direction=end.clone().normalize();
    const radial=points.map(point=>{
      const offset=point.clone().sub(origin),along=clamp(offset.dot(direction),0,length);
      return offset.addScaledVector(direction,-along).length();
    }).sort((a,b)=>a-b);
    const radius=clamp(radial.length?radial[Math.floor((radial.length-1)*.8)]:length*.12,minimum,Math.max(minimum,Math.min(length*.3,height*.1)));
    const rotation=new THREE.Quaternion().setFromUnitVectors(new THREE.Vector3(0,0,1),direction);
    return collider('capsule',end.clone().multiplyScalar(.5),[radius,Math.max(length*.5-radius*.35,minimum)],qArray(rotation));
  });
}
function minColliderZ(body,c){
  const q=new THREE.Quaternion(c.quaternion[1],c.quaternion[2],c.quaternion[3],c.quaternion[0]);
  const base=body.position[2]+c.position[2];
  if(c.type==='sphere')return base-c.size[0];
  if(c.type==='capsule')return base-Math.abs(new THREE.Vector3(0,0,1).applyQuaternion(q).z)*c.size[1]-c.size[0];
  const axes=[new THREE.Vector3(1,0,0),new THREE.Vector3(0,1,0),new THREE.Vector3(0,0,1)];
  return base-axes.reduce((sum,a,i)=>sum+Math.abs(a.applyQuaternion(q).z)*c.size[i],0);
}
export function inferPhysicalRig(scene,{mass,torque,height,profile='auto',rig='auto',bounds}){
  if(!['auto','humanoid','generic'].includes(profile))throw new Error('Choose automatic, humanoid or generic physics fitting.');
  const bones=[];scene.traverse(bone=>{if(bone.isBone)bones.push(bone);});
  const boneSet=new Set(bones),parents=new Map();
  for(const bone of bones){let p=bone.parent;while(p&&!boneSet.has(p))p=p.parent;parents.set(bone,p||null);}
  const {clouds,weights}=pointClouds(scene,bones),positions=new Map(bones.map(b=>[b,b.getWorldPosition(new THREE.Vector3())]));
  const warnings=[],repairs=[],roles=new Map(),byRole=new Map();
  for(const bone of bones){const role=roleOf(bone.name);if(role){roles.set(bone,role);if(byRole.has(role))byRole.set(role,null);else byRole.set(role,bone);}}
  const core=['thighL','shinL','footL','thighR','shinR','footR'];
  let pelvis=core.every(role=>byRole.get(role))?commonAncestor(byRole.get('thighL'),byRole.get('thighR'),parents):null;
  const humanoid=profile!=='generic'&&pelvis&&core.every(role=>weights.get(byRole.get(role))>0)
    &&['L','R'].every(side=>positions.get(byRole.get('thigh'+side)).z>positions.get(byRole.get('foot'+side)).z+height*.1)
    &&Math.abs(positions.get(pelvis).z-(positions.get(byRole.get('thighL')).z+positions.get(byRole.get('thighR')).z)/2)<height*.15;
  if(profile==='humanoid'&&!humanoid)throw new Error('Cannot confidently identify both hips, knees and feet. Use generic fitting or provide a clearly named humanoid deformation rig.');
  const helper=bone=>weights.get(bone)===0&&/(?:pole|target|ik(?:$|[_.:-])|controller|control|effector)/i.test(bone.name);
  const kept=new Set();
  for(const bone of bones)if(weights.get(bone)>0){for(let n=bone;n;n=parents.get(n)){if(humanoid&&n===parents.get(pelvis))break;if(!helper(n))kept.add(n);}}
  // Some authored skeletons contain no weighted vertices (mechanical nodes).
  // Keep their structure and flag the lack of evidence instead of inventing anatomy.
  if(!kept.size)for(const bone of bones)if(!helper(bone))kept.add(bone);
  if(rig==='rigid'||kept.size<2){
    if(rig==='skeleton')throw new Error('This model has no articulated skeleton. Use rigid-body physics or import a rigged character.');
    const center=bounds.getCenter(new THREE.Vector3()),half=bounds.getSize(new THREE.Vector3()).multiplyScalar(.5);
    half.set(Math.max(half.x,.005),Math.max(half.y,.005),Math.max(half.z,.005));
    const graph={version:1,profile:'rigid',root:'character_root',bodies:[{id:'character_root',label:'Rigid mesh',parent:null,position:center.toArray(),quaternion:[1,0,0,0],mass,
      joint:{type:'free',axis:[1,0,0],range:[0,85],torque,damping:1},colliders:[collider('box',new THREE.Vector3(),half.toArray())],source_bones:[]}],
      warnings:['Rigid mesh with a bounding-box collision body and six force/torque controls. It has no articulated limbs.'],repairs:[]};
    return {rig:validatePhysicalRig(graph),bindings:{},lift:0};
  }
  let root=humanoid?pelvis:[...kept].find(b=>!parents.get(b)||!kept.has(parents.get(b)));
  if(!root)throw new Error('Cannot identify a physical root.');
  kept.add(root);
  const physicalParents=new Map();
  for(const bone of kept){if(bone===root){physicalParents.set(bone,null);continue;}let p=parents.get(bone);while(p&&!kept.has(p))p=parents.get(p);physicalParents.set(bone,p&&p!==bone?p:root);}
  if(humanoid){
    for(const side of ['L','R']){
      const thigh=byRole.get('thigh'+side),shin=byRole.get('shin'+side),foot=byRole.get('foot'+side);
      for(const [bone,parent] of [[thigh,pelvis],[shin,thigh],[foot,shin]]){
        if(physicalParents.get(bone)!==parent)repairs.push(`Connected ${bone.name} to ${parent.name} in the physical rig; visual parenting is preserved.`);
        physicalParents.set(bone,parent);
      }
    }
    repairs.push('Tracked the pelvis and fitted knee hinges, shin capsules and foot contact boxes.');
  }else warnings.push('Generic articulated fitting: joint roles, axes and limits are estimates. Review the collision overlay and physical rig before training.');
  const removed=bones.filter(b=>!kept.has(b));
  if(removed.length)repairs.push(`Excluded ${removed.slice(0,12).map(b=>(b.name||'unnamed helper').slice(0,50)).join(', ')+(removed.length>12?` and ${removed.length-12} others`:'')} from physical bodies; visual nodes remain intact.`);
  warnings.push('Mass is estimated from fitted collider volume. Joint limits and motor strength are editable estimates; self-contact is disabled.');
  const bodyNames=new Map([...kept].map(b=>[b,`bone_${String(bones.indexOf(b)).padStart(3,'0')}`]));
  const lateral=humanoid?positions.get(byRole.get('thighL')).clone().sub(positions.get(byRole.get('thighR'))).normalize():new THREE.Vector3(1,0,0);
  const bodies=[...kept].map(bone=>{
    const children=[...kept].filter(child=>physicalParents.get(child)===bone),role=roles.get(bone);
    let type=bone===root?'free':'ball',range=[0,85],axis=[1,0,0];
    if(humanoid&&role?.startsWith('shin')){type='hinge';range=[-5,145];axis=lateral.toArray();}
    else if(humanoid&&role?.startsWith('foot'))range=[0,35];
    else if(humanoid&&/neck|abdomen|hips|spine|torso/i.test(bone.name))range=[0,35];
    else if(humanoid&&/head/i.test(bone.name))type='fixed';
    const fittingRole=humanoid&&/head/i.test(bone.name)?'head':humanoid&&/^(body|hips|abdomen|torso|spine)/i.test(bone.name)?'core':role;
    const colliders=fitColliders(bone,children,clouds,positions,height,fittingRole);
    return {id:bodyNames.get(bone),label:bone.name||bodyNames.get(bone),parent:physicalParents.get(bone)?bodyNames.get(physicalParents.get(bone)):null,
      position:positions.get(bone).toArray(),quaternion:[1,0,0,0],mass:1,
      joint:{type,axis,range,torque:type==='free'||type==='fixed'?0:torque,damping:type==='hinge'?2:1},colliders,source_bones:[bonePath(bone,scene)]};
  });
  const volumes=bodies.map(b=>b.colliders.reduce((sum,c)=>sum+colliderVolume(c),0)),totalVolume=volumes.reduce((a,b)=>a+b,0);
  const shares=volumes.map(volume=>volume/totalVolume);
  if(humanoid){
    // Volume remains the estimator; caps stop clothing/hair geometry from
    // assigning nearly all mass to one appendage.
    const caps=bodies.map(body=>/head/i.test(body.label)?.12:/foot/i.test(body.label)?.04:.4),fixed=new Set();
    for(let pass=0;pass<bodies.length;pass++){
      let changed=false;
      for(let i=0;i<shares.length;i++)if(!fixed.has(i)&&shares[i]>caps[i]){shares[i]=caps[i];fixed.add(i);changed=true;}
      if(!changed)break;
      const left=1-[...fixed].reduce((sum,i)=>sum+shares[i],0),denom=volumes.reduce((sum,x,i)=>sum+(fixed.has(i)?0:x),0);
      shares.forEach((_,i)=>{if(!fixed.has(i))shares[i]=left*volumes[i]/denom;});
    }
    warnings.push('Humanoid fitting uses a head core and mass-share caps so hair and clothing do not dominate inertia. Review stylized proportions.');
  }
  const floorShare=.02/bodies.length;
  bodies.forEach((body,i)=>body.mass=mass*(floorShare+.98*shares[i]));
  const minimum=Math.min(...bodies.flatMap(body=>body.colliders.map(c=>minColliderZ(body,c))));
  const lift=Math.max(0,Math.max(height*.01,.005)-minimum);
  if(lift>0)bodies.forEach(body=>body.position[2]+=lift);
  const graph=validatePhysicalRig({version:1,profile:humanoid?'humanoid':'generic',root:bodyNames.get(root),bodies,warnings,repairs});
  const bindings={};
  for(const bone of kept){
    const position=positions.get(bone).clone();position.z+=lift;
    const quaternion=bone.getWorldQuaternion(new THREE.Quaternion());
    bindings[bonePath(bone,scene)]={body:bodyNames.get(bone),rest_world:new THREE.Matrix4().compose(position,quaternion,new THREE.Vector3(1,1,1)).toArray(),name:bone.name};
  }
  const forward=humanoid?lateral.clone().cross(new THREE.Vector3(0,0,1)).normalize():null;
  return {rig:graph,bindings,lift,heading:forward?-Math.atan2(forward.y,forward.x):0};
}
export function rebuildPhysicalRig(body,input){
  const rig=validatePhysicalRig(input),byID=new Map(rig.bodies.map(b=>[b.id,b]));
  const visual=structuredClone(body.visual||{}),source=visual.source_bindings||{};
  const bone_paths={},bone_offsets={},bone_bodies={};
  for(const physical of rig.bodies)for(const path of physical.source_bones){
    const binding=source[path];if(!binding)throw new Error(`No original rest transform for source bone ${path}. Keep source_bones mapped to the original asset.`);
    bone_paths[path]=physical.id;
    bone_offsets[path]=matrix(physical).invert().multiply(new THREE.Matrix4().fromArray(binding.rest_world)).toArray();
    if(binding.name)bone_bodies[binding.name]=physical.id;
  }
  visual.bone_paths=bone_paths;visual.bone_offsets=bone_offsets;visual.bone_bodies=bone_bodies;
  visual.root_body=rig.root;visual.bind_root=matrix(byID.get(rig.root)).toArray();
  const controls=rig.bodies.reduce((sum,b)=>sum+(b.joint.torque>0?({ball:3,hinge:1,free:rig.profile==='rigid'?6:0,fixed:0})[b.joint.type]:0),0);
  return {...structuredClone(body),xml:physicalRigXML(body.name,rig),root_body:rig.root,physical_rig:rig,visual,
    metadata:{...body.metadata,controls,mass:rig.bodies.reduce((sum,b)=>sum+b.mass,0),notes:[...rig.repairs,...rig.warnings],
      rig_report:{profile:rig.profile,warnings:[...rig.warnings],repairs:[...rig.repairs],bodies:rig.bodies.length,controls}}};
}
