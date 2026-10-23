import test from 'node:test';
import assert from 'node:assert/strict';
import * as tf from '@tensorflow/tfjs';
import {PPO} from '../morpholab/static/browser-policy.mjs';
import {LearningCurriculum} from '../morpholab/static/learning-curriculum.mjs';
import {aggregateTrials,comparisonWarnings} from '../morpholab/static/experiment-results.mjs';
import vm from 'node:vm';
import {readFile} from 'node:fs/promises';
globalThis.tf=tf;await tf.setBackend('cpu');await tf.ready();

test('Adam moments, iterations and policy RNG continue the same next PPO update after JSON roundtrip',async()=>{
  const before=tf.memory().numTensors;
  const p=new PPO(3,2,{hidden_size:16,epochs:2,batch_size:16,seed:71,target_kl:0});let q;
  const observations=Array.from({length:32},(_,i)=>[Math.sin(i),i/32,Math.cos(i)]);
  const makeBatch=async policy=>{const act=await policy.act(observations);return {observations,actions:act.actions,oldLogProbs:act.logProbs,advantages:observations.map(o=>o[0]),returns:observations.map(o=>o[1])};};
  try{
    await p.update(await makeBatch(p));
    const data=JSON.parse(JSON.stringify({policy:await p.serialize(),optimizer:await p.trainingState()}));
    assert.ok(data.optimizer.iterations>0);assert.ok(data.optimizer.slots.length>0);
    q=await PPO.continue(data.policy,data.optimizer);
    const a=await makeBatch(p),b=await makeBatch(q);
    assert.deepEqual(a,b,'Action sampling continues from the same random state.');
    await p.update(a);await q.update(b);
    const w1=(await p.serialize()).weights,w2=(await q.serialize()).weights;
    let max=0;for(const name of Object.keys(w1))w1[name].values.forEach((v,i)=>max=Math.max(max,Math.abs(v-w2[name].values[i])));
    assert.ok(max<1e-6,`Maximum resumed weight difference: ${max}`);
    assert.equal((await p.trainingState()).iterations,(await q.trainingState()).iterations);
    const malformed=structuredClone(data);malformed.optimizer.slots[0].shape=[999];
    await assert.rejects(PPO.continue(malformed.policy,malformed.optimizer),/shape/);
  }finally{p.dispose();q?.dispose();}
  assert.equal(tf.memory().numTensors,before,'Training serialization and invalid restore must not leak tensors.');
});

test('curriculum continuation keeps completed-episode gates',()=>{
  const c=new LearningCurriculum(1);
  for(let i=0;i<11;i++)c.record({fell:false,speed_error:0});
  const r=new LearningCurriculum(1).restore(JSON.parse(JSON.stringify(c.serialize())));
  assert.equal(r.advance(),false);r.record({fell:false,speed_error:0});assert.equal(r.advance(),true);
  assert.equal(r.index,1);assert.throws(()=>r.restore({index:99,episodes:[]}),/Invalid/);
});

test('experiment aggregation counts training runs and retains failed trials',()=>{
  const rows=aggregateTrials([
    {body:'biped',condition:'A',status:'completed',metrics:{survival_rate:1,speed_error:1}},
    {body:'biped',condition:'A',status:'completed',metrics:{survival_rate:0,speed_error:3}},
    {body:'biped',condition:'A',status:'failed'}]);
  assert.equal(rows[0].completed,2);assert.equal(rows[0].failed,1);
  assert.equal(rows[0].speed_error.mean,2);assert.equal(rows[0].speed_error.sd,Math.sqrt(2));
  assert.equal(rows[0].survival.mean,.5);
  assert.match(comparisonWarnings([{steps:10,config:{seed:1,body:{legs:2}}},{steps:20,config:{seed:1,body:{legs:4}}}]).join(' '),/descriptive.*not independent/);
});

test('continuation compatibility includes nested physical settings but excludes the new step budget',async()=>{
  const source=await readFile(new URL('../morpholab/static/browser-worker.js',import.meta.url),'utf8');
  const scope={onmessage:null,performance,setTimeout,clearTimeout,postMessage:()=>{}};
  vm.runInNewContext(`${source}\nglobalThis.signature=trainingSignature`,scope);
  const config={body:{mass:10,joints:[{limit:30}]},total_steps:4096,device:'auto'};
  assert.equal(scope.signature(config),scope.signature({...config,total_steps:65536}));
  assert.notEqual(scope.signature(config),scope.signature({...config,body:{...config.body,mass:11}}));
  assert.notEqual(scope.signature(config),scope.signature({...config,body:{...config.body,joints:[{limit:40}]}}));
});
