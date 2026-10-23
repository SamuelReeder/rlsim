import assert from "node:assert/strict";
import {readFile} from "node:fs/promises";
import vm from "node:vm";
import test from "node:test";
import loadMujoco from "@mujoco/mujoco";
import * as physics from "../morpholab/static/browser-physics.mjs";
import * as imported from "../morpholab/static/imported-physics.mjs";
import * as tasks from "../morpholab/static/task-regimes.mjs";

const source=await readFile(new URL("../morpholab/static/browser-worker.js",import.meta.url),"utf8");
const scope={onmessage:null};
vm.runInNewContext(`${source}\nglobalThis.api={modelInfo,preview};`,scope);
const mj=await loadMujoco(),runtime={...physics,...imported,...tasks,mj};
const fixture=async name=>new Uint8Array(await readFile(new URL(`../examples/imports/${name}`,import.meta.url)));
const keepRunning={preset:"balance",terminate_on_fall:false};

test("all built-in previews expose nominal mass and useful body bounds without floor geometry",()=>{
  for(const body of Object.values(physics.PRESETS)){
    const result=scope.api.preview({body,environment:{},task:keepRunning},runtime);
    const info=result.model_info,summary=info.physics_summary;
    assert.ok(Math.abs(summary.total_mass-body.body_mass)<1e-9);
    assert.equal(summary.articulated_joints,body.legs*2);
    assert.equal(summary.actuated_joints,info.action_size);
    assert.equal(summary.moving_bodies,1+body.legs*2);
    assert.equal(summary.floating_base,false);
    assert.ok(summary.tracked_height>.5&&summary.tracked_height<2);
    assert.ok(summary.body_height>.5&&summary.body_height<2);
    assert.ok(summary.extent.every(value=>Number.isFinite(value)&&value>0&&value<3));
    assert.equal(summary.bounds_source,"compiled_collision_aabb");
    assert.equal(info.observation_names.length,info.observation_size);
  }
});

test("authored free-body summary counts compiled actuation and excludes unrelated massive scenery",async()=>{
  const xml=new TextDecoder().decode(await fixture("free-body.xml"))
    .replace("</worldbody>",'<body name="scenery" pos="100 0 50"><geom type="box" size="20 20 20" mass="1000"/></body></worldbody>');
  const body=await imported.preparePhysicsFiles(mj,[{path:"free-body.xml",data:new TextEncoder().encode(xml)}],{root_body:"root"});
  const env=new imported.ImportedWalker(mj,body,{}, {task:keepRunning,randomize:true,seed:19});
  try{
    const summary=scope.api.modelInfo(env,runtime).physics_summary;
    assert.equal(summary.total_mass,2.5,"Mass is nominal even when environment randomization is active.");
    assert.equal(summary.floating_base,true);
    assert.equal(summary.articulated_joints,1);
    assert.equal(summary.actuated_joints,1,"Three ball-axis motors control one joint.");
    assert.equal(summary.moving_bodies,2);
    assert.ok(summary.extent.every(value=>value>0&&value<1));
    assert.ok(Math.abs(summary.tracked_height-1)<.03);
    const saved=JSON.stringify(summary);
    for(let i=0;i<80;i++)env.step(new Array(env.actionSize).fill(0));
    assert.equal(JSON.stringify(scope.api.modelInfo(env,runtime).physics_summary),saved,
      "Checkpoints retain initial body dimensions after falls and motion.");
    assert.ok(Math.abs(env.data.xpos[env._root*3+2]-summary.tracked_height)>.1);
  }finally{env.close();}
});

test("a fixed-base authored arm keeps fixed ancestors in its mass and reports movable parts",async()=>{
  const body=await imported.preparePhysicsFiles(mj,[
    {path:"robot/serial-arm.urdf",data:await fixture("serial-arm.urdf")},
    {path:"robot/meshes/box.obj",data:await fixture("meshes/box.obj")},
  ],{root_body:"forearm"});
  const result=scope.api.preview({body,environment:{},task:keepRunning},runtime);
  const summary=result.model_info.physics_summary;
  assert.equal(result.model_info.tracked_body,"forearm");
  assert.equal(summary.total_mass,4.6);
  assert.equal(summary.floating_base,false);
  assert.equal(summary.articulated_joints,2);
  assert.equal(summary.actuated_joints,2);
  assert.equal(summary.moving_bodies,2);
  assert.ok(summary.body_height>0&&summary.body_height<2);
});

test("unactuated rigid bodies remain valid and expose zero articulation",async()=>{
  const xml='<mujoco><worldbody><body name="root" pos="0 0 1"><freejoint/><geom type="box" size=".1 .2 .3" mass="2"/></body></worldbody></mujoco>';
  const body=await imported.preparePhysicsFiles(mj,[{path:"rigid.xml",data:new TextEncoder().encode(xml)}],{auto_actuators:false});
  const result=scope.api.preview({body,environment:{},task:keepRunning},runtime);
  assert.equal(result.model_info.action_size,0);
  assert.equal(result.model_info.physics_summary.actuated_joints,0);
  assert.equal(result.model_info.physics_summary.articulated_joints,0);
  assert.equal(result.model_info.physics_summary.floating_base,true);
  assert.ok(Math.abs(result.model_info.physics_summary.body_height-.6)<.01);
});
