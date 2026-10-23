import assert from "node:assert/strict";
import loadMujoco from "@mujoco/mujoco";
import {physicalRigXML,validatePhysicalRig} from "../morpholab/static/physical-rig.mjs";
import {ImportedWalker} from "../morpholab/static/imported-physics.mjs";
import {LearningCurriculum} from "../morpholab/static/learning-curriculum.mjs";

const curriculum=new LearningCurriculum(1);
for(let i=0;i<20;i++)curriculum.record({fell:true,speed_error:0});
assert.equal(curriculum.advance(),false,"Falling bodies must not advance just because time passes.");
for(let stage=0;stage<3;stage++){
  for(let i=0;i<12;i++)curriculum.record({fell:false,speed_error:0});
  assert.equal(curriculum.advance(),true);assert.equal(curriculum.index,stage+1);
}
for(let i=0;i<12;i++)curriculum.record({fell:false,speed_error:0});
assert.equal(curriculum.advance(),false);
const speedGate=new LearningCurriculum(1);
for(let i=0;i<12;i++)speedGate.record({fell:false,speed_error:.4});
assert.equal(speedGate.advance(),false,"Standing with excessive drift must not pass balance.");

const part=(id,parent,position,type)=>({id,label:id,parent,position,quaternion:[1,0,0,0],mass:1,
  joint:{type,axis:[0,1,0],range:type==="hinge"?[-5,90]:[0,85],torque:type==="free"?0:10,damping:.1},
  colliders:[{type:"capsule",position:[0,0,-.1],quaternion:[1,0,0,0],size:[.04,.1]}],source_bones:[id]});
const rig={version:1,profile:"generic",root:"base",warnings:[],repairs:[],
  bodies:[part("base",null,[0,0,2],"free"),part("arm","base",[0,0,1.8],"ball"),part("tip","arm",[0,0,1.6],"hinge")],
  control:{mode:"joint_target",excursion:30,stiffness:1,damping_ratio:.7}};
assert.deepEqual(validatePhysicalRig(rig).control,rig.control);
const xml=physicalRigXML("servo test",rig),mj=await loadMujoco();
assert.match(xml,/<orientation /);assert.match(xml,/<position /);
const body={kind:"mjcf",name:"servo test",xml,assets:[],physical_rig:rig,root_body:"base"};
const env=new ImportedWalker(mj,body,{gravity:.5,friction:1},{randomize:false});
try{
  assert.equal(env.actionSchema,"rlsim-joint-target-v1");
  assert.equal(env.actionSize,4);assert.equal(env.model.nactuator,2);
  assert.deepEqual(env.actionNames,["target_arm[0]","target_arm[1]","target_arm[2]","target_tip"]);
  const joint=mj.mj_name2id(env.model,mj.mjtObj.mjOBJ_JOINT.value,"joint_arm");
  const at=env.model.jnt_qposadr[joint];
  env.data.qpos.set([Math.cos(.2),Math.sin(.2),0,0],at);mj.mj_forward(env.model,env.data);
  for(let i=0;i<25;i++)env.step([0,0,0,0]);
  assert.ok(2*Math.acos(Math.min(1,Math.abs(env.data.qpos[at])))<.08,"Servo restores a perturbed ball joint using orientation error.");
  env.step([0,0,0,0]);
  assert.equal(env.data.ctrl[3],0,"Zero action holds zero even with asymmetric hinge bounds.");
  env.step([1,1,1,1]);
  assert.ok(Math.hypot(...env.data.ctrl.slice(0,3))<=Math.PI/6+1e-8);
  assert.ok(Math.hypot(...env.data.actuator_force.slice(0,3))<=10+1e-7,"Ball output is bounded by total torque.");
  env.step([0,0,0,-1]);assert.ok(env.data.ctrl[3]<0&&env.data.ctrl[3]>=-5*Math.PI/180-1e-8);
  env.setLearningStage({preset:"balance",speed_fraction:0,randomization:0});
  env.randomize=true;env.reset(12);
  assert.equal(env.task.preset,"balance");assert.equal(env.target_speed,0);
  assert.deepEqual(env._scales,[1,1,1]);
  env.setLearningStage({preset:"velocity",speed_fraction:.25,randomization:.25});env.reset(12);
  assert.equal(env.target_speed,.25);
  assert.ok(Math.abs(env._scales[0]-1)<=.05);
  assert.deepEqual(Array.from(env.model.actuator_gear),Array.from(env._base.actuator_gear),"Servo strength randomization must not distort its target angles.");
  assert.ok(Math.abs(env.model.actuator_gainprm[0]/env._base.actuator_gainprm[0]-env._scales[1])<1e-9);
  assert.ok(env.data.qpos.every(Number.isFinite));
}finally{env.close();}
const bare=new ImportedWalker(mj,{kind:"mjcf",name:"xml",xml,assets:[]},{gravity:.5,friction:1},{});
assert.equal(bare.actionSchema,"rlsim-joint-target-v1","XML export carries action semantics.");
bare.close();
console.log("Curriculum gates and real MuJoCo hinge/orientation servo checks passed.");
