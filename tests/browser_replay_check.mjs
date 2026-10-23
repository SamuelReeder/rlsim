import assert from "node:assert/strict";
import {readFile} from "node:fs/promises";
import vm from "node:vm";
import test from "node:test";

const workerSource=await readFile(new URL("../morpholab/static/browser-worker.js",import.meta.url),"utf8");
const clientSource=await readFile(new URL("../morpholab/static/browser-client.js",import.meta.url),"utf8");

function workerHarness(){
  const messages=[],counts={created:0,closed:0,policiesClosed:0,predictions:0};
  const scope={onmessage:null,performance,setTimeout,clearTimeout,postMessage:message=>messages.push(message)};
  vm.runInNewContext(`${workerSource}\nglobalThis.api={replay,clearInferenceCache,publish,statusSnapshot,setStatus:value=>status=value,setRuntime:value=>runtime=Promise.resolve(value)};`,scope);
  class Walker{
    observationSize=2;actionSize=1;max_episode_steps=130;
    constructor(_mj,_body,environment,options){counts.created++;this.environment=environment;this.options=options;this.reset();}
    reset(){this.time=0;this.position=0;return [0,0];}
    step(action){
      this.time++;
      this.position+=action[0]*.01;
      return {obs:[this.time,this.position],reward:this.position,info:{energy_cost:action[0]**2},
        terminated:false,truncated:this.time===this.max_episode_steps};
    }
    getFrame(){return {time:this.time*.02,position:this.position};}
    close(){counts.closed++;}
  }
  const runtime={validateBody:body=>body,validateEnvironment:env=>env||{gravity:9.81,friction:1},
    BrowserWalker:Walker,mj:{},hardware:{training_device:"cpu"},
    RunningNormalizer:{restore:()=>({normalize:obs=>obs})},
    PPO:{restore:async()=>({
      predict:async()=>{counts.predictions++;return [[.3]];},
      act:async()=>{throw new Error("Replay should use the actor-only path.");},
      dispose:()=>{counts.policiesClosed++;}
    })}};
  const record={format:"morpholab-browser-run-v1",config:{body:{},environment:{gravity:9.81,friction:1}},
    policy:{obsSize:2,actionSize:1},normalizer:{size:2}};
  return {...scope.api,dispatch:scope.onmessage,runtime,record,messages,counts};
}

test("stop cancels the requested replay, flushes partial frames, disposes resources and permits another replay",async()=>{
  const h=workerHarness();h.setRuntime(h.runtime);
  const running=h.dispatch({data:{id:1,op:'replay',payload:{record:h.record,replay_id:'first',stream:true}}});
  while(h.counts.predictions<10)await new Promise(resolve=>setTimeout(resolve,0));
  await h.dispatch({data:{id:2,op:'stop_replay',payload:{replay_id:'other'}}});
  assert.equal(h.messages.find(m=>m.id===2).result.stopping,false);
  await h.dispatch({data:{id:3,op:'stop_replay',payload:{replay_id:'first'}}});
  const predictions=h.counts.predictions;
  await running;
  const result=h.messages.find(m=>m.id===1&&!m.event).result;
  assert.equal(result.cancelled,true);
  assert.ok(result.summary.steps>0&&result.summary.steps<130);
  assert.equal(h.counts.predictions,predictions);
  assert.equal(h.counts.closed,1);assert.equal(h.counts.policiesClosed,1);
  const frames=h.messages.filter(m=>m.id===1&&m.event==='replay_progress').flatMap(m=>m.progress.frames||[]);
  assert.equal(frames.at(-1).time,result.summary.seconds);
  await h.dispatch({data:{id:4,op:'replay',payload:{record:h.record,replay_id:'second'}}});
  const next=h.messages.find(m=>m.id===4&&!m.event).result;
  assert.equal(next.summary.steps,130);assert.equal(next.cancelled,undefined);
  await h.clearInferenceCache();
});

test("cancellation during restore runs no episode and frees the loaded controller",async()=>{
  const h=workerHarness();let cancelled=false,release,loading;
  const original=h.runtime.PPO.restore;
  h.runtime.PPO.restore=async()=>{loading=true;await new Promise(resolve=>release=resolve);return original();};
  const pending=h.replay({record:h.record},h.runtime,()=>{},()=>cancelled);
  while(!loading)await new Promise(resolve=>setTimeout(resolve,0));
  cancelled=true;release();
  const result=await pending;
  assert.equal(result.cancelled,true);assert.equal(result.frames.length,0);
  assert.equal(h.counts.predictions,0,'Cancellation skips warmup and episode predictions.');
  assert.equal(h.counts.closed,1);assert.equal(h.counts.policiesClosed,1);
});

test("stop while reading model storage prevents a replay from reaching the worker",async()=>{
  const sent=[];
  const scope={window:{},document:{baseURI:'http://example.test/'},URL,console,
    Worker:class{postMessage(message){sent.push(message);}},
    indexedDB:{open:()=>{const request={};queueMicrotask(()=>{request.result={};request.onsuccess();});return request;}}};
  vm.runInNewContext(clientSource,scope);
  const client=new scope.window.BrowserBackend();let release;
  client.records=()=>new Promise(resolve=>release=resolve);
  const pending=client.request('/api/replay',{run_id:'model',replay_id:'loading'});
  await client.request('/api/stop-replay',{replay_id:'loading'});
  release({config:{}});
  assert.equal((await pending).cancelled,true);
  assert.equal(sent.length,0);assert.equal(client.replays.size,0);
});

test("streamed replay sends an initial frame before stepping and preserves full deterministic results",async()=>{
  const harness=workerHarness(),{record,runtime,counts}=harness,events=[];
  const full=await harness.replay({record,seed:23},runtime);
  assert.equal(counts.created,1,"Restoration and replay share one compiled environment.");
  assert.equal(counts.closed,0,"The prepared environment remains in the bounded cache.");
  let resolved=false;
  const result=await harness.replay({record,seed:23,stream:true},runtime,event=>{
    assert.equal(resolved,false);
    if(event.phase==="ready")assert.equal(counts.predictions,131,"The initial frame precedes new steps; the first restore warmed the actor once.");
    events.push(event);
  }).then(value=>{resolved=true;return value;});
  const ready=events.find(event=>event.phase==="ready");
  const chunks=events.filter(event=>event.phase==="simulating");
  assert.ok(chunks.length>=3,"Long episodes arrive in bounded chunks.");
  assert.ok(chunks.every(chunk=>chunk.frames.length<=32&&!chunk.body&&!chunk.model_info));
  assert.equal(JSON.stringify(events.flatMap(event=>event.frames||[])),JSON.stringify(full.frames));
  assert.deepEqual(result.summary,full.summary);
  assert.deepEqual(ready.body,full.body);
  assert.equal(result.frames,undefined,"The final response does not duplicate previously transferred frames or assets.");
  assert.equal(result.body,undefined);
  assert.equal(counts.created,1);
  await harness.clearInferenceCache();
  assert.equal(counts.closed,1);
  assert.equal(counts.policiesClosed,1);
});

test("failed restoration closes its validation environment and status telemetry omits immutable assets",async()=>{
  const harness=workerHarness(),{runtime,record,counts}=harness;
  await assert.rejects(harness.replay({record:{...record,normalizer:{size:99}}},runtime),/dimensions/);
  assert.equal(counts.created,1);
  assert.equal(counts.closed,1);
  const config={body:{visual:{files:[{data:"large unchanged source asset"}]}}};
  harness.setStatus({run_id:"one",config,steps:1,model_info:{action_size:2}});
  harness.publish();
  harness.publish();
  assert.equal(harness.messages[0].status.config,config);
  assert.equal(harness.messages[1].status.config,undefined);
  assert.equal(harness.messages[1].status.model_info,undefined);
  assert.equal(harness.statusSnapshot().config,undefined);
  harness.setStatus({run_id:"two",config,steps:0});
  harness.publish();
  assert.equal(harness.messages[2].status.config,config,"New runs send their own body once.");
});

test("client assembles request-scoped chunks, survives callback errors, and hydrates only the current run",async()=>{
  const logs=[],sent=[];
  const scope={window:{},document:{baseURI:"http://example.test/"},URL,console:{error:(...args)=>logs.push(args)},
    Worker:class{postMessage(message){sent.push(message);}},
    indexedDB:{open:()=>{const request={};queueMicrotask(()=>{request.result={};request.onsuccess();});return request;}}};
  vm.runInNewContext(clientSource,scope);
  const client=new scope.window.BrowserBackend(),received=[];
  const pending=client.request("/api/replay",{body:{}},{onProgress:progress=>{
    received.push(progress.phase);
    if(progress.phase==="ready")throw new Error("Renderer failure");
  }});
  const id=sent[0].id;
  const progress=(request,phase,extra={})=>client.worker.onmessage({data:{event:"replay_progress",id:request,progress:{phase,...extra}}});
  await progress(id+1,"ready",{frames:[{time:999}]});
  await progress(id,"ready",{body:{name:"saved body"},frames:[{time:0}]});
  await progress(id,"simulating",{frames:[{time:.04},{time:.08}]});
  await client.worker.onmessage({data:{id,result:{streamed:true,summary:{seconds:.08}}}});
  const result=await pending;
  assert.equal(JSON.stringify(result.frames),JSON.stringify([{time:0},{time:.04},{time:.08}]));
  assert.equal(result.body.name,"saved body");
  assert.equal(logs.length,1);
  assert.equal(client.pending.size,0);
  assert.deepEqual(received,["loading","ready","simulating"]);
  await client.worker.onmessage({data:{event:"status",status:{run_id:"one",config:{body:{name:"one"}},model_info:{action_size:2}}}});
  assert.equal(client.hydrateStatus({run_id:"one",steps:100}).config.body.name,"one");
  assert.equal(client.hydrateStatus({run_id:"two",steps:0}).config,undefined);
  assert.equal(client.hydrateStatus({run_id:"two",steps:0}).model_info,undefined);
});
