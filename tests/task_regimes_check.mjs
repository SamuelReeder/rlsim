import assert from 'node:assert/strict';
import test from 'node:test';
import loadMujoco from '@mujoco/mujoco';
import {compileReward,validateTask,evaluateTask,TaskWalker} from '../morpholab/static/task-regimes.mjs';
import {BrowserWalker,PRESETS} from '../morpholab/static/browser-physics.mjs';
const mj=await loadMujoco();
test('custom formulas use bounded math syntax and reject executable code',()=>{
  assert.equal(compileReward('vx > 1 ? clamp(vx,0,2)**2 + sin(pi/2) : 0')({vx:3,pi:Math.PI}),5);
  assert.equal(compileReward('0 && 1/0')({}),0);
  for(const expression of ['globalThis.fetch("x")','constructor(1)','x=2','height;throw 1','foo + 1','1e999','()'])assert.throws(()=>compileReward(expression));
  assert.throws(()=>compileReward('1/0')({}),/non-finite/);
  assert.throws(()=>validateTask({target:[Infinity,0,0]}));
  assert.throws(()=>validateTask({weights:{execute:1}}));
});
test('task rewards and fall/target termination are explicit and finite',()=>{
  const task=validateTask({preset:'custom',expression:'2*vx + energy',terminate_on_fall:false});
  const state={velocity:[2,0,0],position:[1,0,.8],upright:1,height:.8,initial_height:.8,target_speed:1,energy:.5,smoothness:0,elapsed:1,dt:.02};
  assert.equal(evaluateTask(task,state).reward,3.5);
  const fallen=evaluateTask(task,{...state,height:0,upright:0});
  assert.equal(fallen.fallen,true);assert.equal(fallen.terminated,false);
  assert.equal(evaluateTask(validateTask({preset:'target'}),state).distance,1);
});
test('planar task adds actual tracked position/velocity and obeys custom duration',()=>{
  const env=new TaskWalker(new BrowserWalker(mj,PRESETS.quadruped,{}),{preset:'custom',expression:'1',episode_seconds:1,terminate_on_fall:false});
  try{
    let obs=env.reset(10);assert.equal(obs.length,env.env.observationSize+9);
    assert.equal(env.setTrainingProgress(.5).length,obs.length);
    let total=0,last;for(let i=0;i<50;i++){last=env.step(new Array(env.actionSize).fill(0));total+=last.reward;}
    assert.equal(total,50);assert.equal(last.truncated,true);assert.equal(last.terminated,false);
    assert.notDeepEqual(Array.from(last.obs.slice(-9,-6)),Array.from(obs.slice(-9,-6)));
  }finally{env.close();}
});
