import assert from 'node:assert/strict';
import test from 'node:test';
import loadMujoco from '@mujoco/mujoco';
import {BrowserWalker,PRESETS} from '../morpholab/static/browser-physics.mjs';
import {TaskWalker,validateTask} from '../morpholab/static/task-regimes.mjs';
import {EffortMeter} from '../morpholab/static/effort.mjs';
import {validateMorphology} from '../morpholab/static/morphology.mjs';
const mj=await loadMujoco();

test('measured work, support contacts and fall state survive the browser task wrapper',()=>{
  const task=validateTask({preset:'walk',foot_only_support:true,mechanical_work_weight:.2,min_height_ratio:0,max_tilt_degrees:180});
  const physics=new BrowserWalker(mj,PRESETS.quadruped,{}, {task});
  const env=new TaskWalker(physics,task);
  try{
    env.reset(13);let work=0;
    for(let i=0;i<30;i++){
      const r=env.step(new Float32Array(8));work+=r.info.motor_work_j;
      assert.equal(r.terminated,false);assert.equal(r.info.nonfoot_contact,false);
      assert.ok(Math.abs(r.reward-Object.values(r.info.reward_components).reduce((a,b)=>a+b,0))<1e-10);
      const expected=-.2*r.info.motor_work_j/(physics.model.body_mass.reduce((a,b)=>a+b,0)*9.81*physics.target_speed*physics.dt);
      assert.ok(Math.abs(r.info.reward_components.mechanical_work-expected)<1e-10);
    }
    assert.ok(work>0);assert.ok(Math.abs(physics.effort.summary(2,4,9.81).absolute_motor_work_j-work)<1e-9);
    physics.data.qpos[1]=.05-physics.model.body_pos[physics._torsoId*3+2];
    mj.mj_forward(physics.model,physics.data);
    const r=env.step(new Float32Array(8));
    assert.equal(r.info.nonfoot_contact,true);assert.equal(r.info.fallen,true);assert.equal(r.terminated,true);
    assert.equal(physics._fallen,true);assert.equal(r.info.reward_components.nonfoot_contact,-2);
    assert.equal(r.info.reward_components.gait_contacts,undefined);
    assert.ok(Math.abs(r.reward-Object.values(r.info.reward_components).reduce((a,b)=>a+b,0))<1e-10);
    env.reset(13);assert.equal(physics.effort.summary(2,4,9.81).absolute_motor_work_j,0);
  }finally{env.close();}
});

test('measurement does not count braking as positive motor work or backward travel as transport',()=>{
  const meter=new EffortMeter();meter.observe([2,-3],[4,5],[12,-18],[.99,-.5],.004);
  const s=meter.summary(-2,4,10);
  assert.equal(s.mechanical_cost_of_transport,null);assert.equal(s.positive_motor_work_j,.032);
  assert.equal(s.absolute_motor_work_j,.092);assert.equal(s.actuator_saturation_fraction,.5);
});

test('old records retain their fields; new settings validate before running',()=>{
  const old=validateTask({preset:'walk'});
  assert.equal(Object.hasOwn(old,'mechanical_work_weight'),false);assert.equal(Object.hasOwn(old,'foot_only_support'),false);
  for(const input of [{preset:'velocity',foot_only_support:true},{preset:'walk',mechanical_work_weight:-1},{preset:'walk',mechanical_work_weight:3},{preset:'walk',foot_only_support:1}])assert.throws(()=>validateTask(input));
  assert.deepEqual(validateMorphology({}),{enabled:true,width:.2,ramp_steps:131072});
  const input={width:.4,withhold_width:.2,speed_range:[.5,1.5],boundary_fraction:.2,robust_validation:true};
  assert.deepEqual(validateMorphology(input),{enabled:true,ramp_steps:131072,...input});
  for(const input of [{speed_range:[2,1]},{speed_range:[.1,Infinity]},{boundary_fraction:.8},{width:.2,withhold_width:.3},{robust_validation:1}])assert.throws(()=>validateMorphology(input));
});
