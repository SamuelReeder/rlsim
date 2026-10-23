import assert from "node:assert/strict";
import {readFile} from "node:fs/promises";
import vm from "node:vm";
import * as tf from "@tensorflow/tfjs";
import {PPO,InferenceActor,RunningNormalizer} from "../morpholab/static/browser-policy.mjs";
globalThis.tf=tf;
await tf.setBackend("cpu");
const tensors=tf.memory().numTensors;

const policy=new PPO(2,1,{hidden_size:16,epochs:3,batch_size:16,target_kl:.03});
const observations=Array.from({length:32},(_,i)=>[Math.sin(i),Math.cos(i)]);
const sample=await policy.act(observations),before=await policy.serialize();
const batch={observations,actions:sample.actions,oldLogProbs:sample.logProbs.map(p=>p+5),
  advantages:observations.map(o=>o[0]),returns:observations.map(o=>o[1])};
const stopped=await policy.update(batch);
assert.equal(stopped.kl_early_stopped,1);
assert.equal(stopped.updates_completed,0);
assert.deepEqual((await policy.serialize()).weights,before.weights,"Diverged rollout must not apply another update.");
policy.config.target_kl=0;
const continued=await policy.update({...batch,oldLogProbs:sample.logProbs});
assert.equal(continued.updates_completed,6);
assert.equal(continued.kl_early_stopped,0);
const saved=await policy.serialize(),actor=await InferenceActor.restore(saved);
assert.deepEqual(await actor.predict(observations),await policy.predict(observations));
assert.ok(!actor.optimizer&&!actor.weights.critic_w1);
assert.ok(Object.keys(actor.weights).every(k=>k.startsWith("actor_")));
const corrupt=structuredClone(saved);corrupt.weights.actor_w1.values[0]=NaN;
await assert.rejects(InferenceActor.restore(corrupt),/finite/);
actor.dispose();policy.dispose();

const source=await readFile(new URL("../morpholab/static/browser-worker.js",import.meta.url),"utf8");
const records=[];
const worker={performance,setTimeout,clearTimeout,onmessage:null,postMessage(message){
  if(message.event==="checkpoint"){records.push(structuredClone(message.record));worker.onmessage({data:{op:"saved",token:message.token}});}
}};
vm.runInNewContext(`${source}\nglobalThis.test={train,replay,clearInferenceCache,selectedCheckpoint};`,worker);
let constructed=0,closed=0;
class Walker{
  observationSize=2;actionSize=1;dt=.02;max_episode_steps=3;
  constructor(_mj,_body,_world,options){constructed++;this.options=options;this.reset();}
  reset(seed=7){this.steps=0;this.position=0;this.seed=seed;return [seed%3,1];}
  step(action){this.steps++;this.position+=action[0]*.02;return {obs:[this.seed%3,1],reward:1-(action[0]-1)**2,
    terminated:false,truncated:this.steps===3,info:{velocity:action[0],energy_cost:action[0]**2}};}
  getFrame(){return {position:this.position,time:this.steps*this.dt};}
  close(){closed++;}
}
class LearningPolicy{
  value=0;
  act(obs){return {actions:obs.map(()=>[this.value]),values:obs.map(()=>0),logProbs:obs.map(()=>0)};}
  predict(obs){return obs.map(()=>[this.value]);}
  update(){this.value++;return {loss:0};}
  serialize(){return {value:this.value};}
  dispose(){}
}
const runtime={validateBody:v=>v,validateEnvironment:v=>v,BrowserWalker:Walker,
  PPO:LearningPolicy,RunningNormalizer,exportMJCF:()=>"<mujoco/>",hardware:{},mj:{}};
await worker.test.train({body:{},environment:{},target_speed:1,seed:7,n_envs:1,
  total_steps:8,rollout_steps:2,gamma:.99,curriculum:false,randomize:true,
  best_checkpoint:true,evaluation_interval:2,evaluation_episodes:2},runtime);
const record=records.at(-1);
assert.equal(record.best.steps,2);
assert.equal(record.best.policy.value,1,"Keep the early good controller when later updates regress.");
assert.equal(record.policy.value,4,"Keep latest weights separately.");
assert.ok(record.best.normalizer.count<record.normalizer.count,"Best retains its matching earlier normalizer.");
assert.equal(record.best.validation.mean_return,3);
assert.equal(worker.test.selectedCheckpoint(record,"best").policy.value,1);
assert.equal(worker.test.selectedCheckpoint(record,"latest").policy.value,4);
assert.equal(worker.test.selectedCheckpoint({...record,best:undefined},"best").checkpoint,"latest");
assert.equal(constructed,closed,"Training and validation environments are released.");

const normalizer=new RunningNormalizer(2);normalizer.update(observations);
const replayRecord={format:"morpholab-browser-run-v1",id:"same-id",steps:4096,
  config:{body:{name:"test"},environment:{gravity:9.81,friction:1},target_speed:1},
  policy:saved,normalizer:normalizer.serialize()};
const replayRuntime={...runtime,PPO,InferenceActor};
const first=await worker.test.replay({record:replayRecord,seed:17},replayRuntime);
const warmConstructed=constructed;
const second=await worker.test.replay({record:replayRecord,seed:17},replayRuntime);
assert.equal(first.inference_cache_hit,false);assert.equal(second.inference_cache_hit,true);
assert.equal(constructed,warmConstructed,"Warm replay reuses compiled physics.");
assert.deepEqual(first.frames,second.frames,"Cache must reset all episode state.");
const changedSeed=await worker.test.replay({record:replayRecord,seed:18},replayRuntime);
assert.equal(changedSeed.seed,18);assert.equal(changedSeed.inference_cache_hit,true);
const replacement=structuredClone(replayRecord);replacement.policy.weights.actor_out_b.values[0]+=.5;
const changedWeights=await worker.test.replay({record:replacement,seed:17},replayRuntime);
assert.equal(changedWeights.inference_cache_hit,false,"Same id with new weights invalidates cache.");
assert.notDeepEqual(changedWeights.frames,first.frames);
const changedWorld=await worker.test.replay({record:replacement,environment:{gravity:5,friction:1}},replayRuntime);
assert.equal(changedWorld.inference_cache_hit,false);
await worker.test.clearInferenceCache();
assert.equal(constructed,closed);
assert.equal(tf.memory().numTensors,tensors,"Actors, failures and evictions release tensors.");
console.log("PPO safeguard, best/last pairing, actor parity and inference cache checks passed.");
