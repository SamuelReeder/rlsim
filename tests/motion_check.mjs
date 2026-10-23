import assert from 'node:assert/strict';
import test from 'node:test';
import loadMujoco from '@mujoco/mujoco';
import {BrowserWalker} from '../morpholab/static/browser-physics.mjs';
import {TaskWalker,validateTask} from '../morpholab/static/task-regimes.mjs';
import {walkingPass} from '../morpholab/static/motion.mjs';
const mj=await loadMujoco();
test('Walking changes the action contract, preserves limits and exposes measured foot use',()=>{
 const base=new BrowserWalker(mj,{}, {},{task:{preset:'walk'}}),env=new TaskWalker(base,{preset:'walk'});
 try{
  const initial=env.reset(13);assert.equal(initial.length,57);assert.equal(env.actionSchema,'rlsim-joint-targets-v1');
  for(let i=0;i<100;i++){
   const r=env.step(Array(8).fill(0));assert.ok(r.obs.every(Number.isFinite));
   assert.ok(Math.abs(r.reward-Object.values(r.info.reward_components).reduce((a,b)=>a+b,0))<1e-10);
   assert.ok(env.data.ctrl.every(x=>Math.abs(x)<=1));assert.equal(r.terminated,false);
  }
  assert.deepEqual(base.motion.summary().steps_per_leg,[0,0,0,0]);
  assert.equal(walkingPass({gait:base.motion.summary(),seconds:20,speed_error:0,fell:false}),false);
  assert.deepEqual(env.reset(13),initial);
 }finally{env.close();}
});
test('Walking cannot silently use an unrelated tracked body or custom weights',()=>{
 assert.throws(()=>validateTask({preset:'walk',tracked_body:'arm'}),/template torso/);
 assert.throws(()=>validateTask({preset:'walk',weights:{alive:1}}),/built-in/);
});
