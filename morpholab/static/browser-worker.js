/* Simulation and training stay off the UI thread and entirely on this device. */
"use strict";
let runtime, status = {status:"idle",history:[]}, stopping = false, running = false, operation = false;
const replayRequests=new Map();
const pause = () => new Promise(resolve => setTimeout(resolve, 0));
const send = (event, payload) => postMessage({event,...payload});
// randomUUID is unavailable on plain HTTP; CPU training can still work there.
const runId = () => `browser-${Date.now()}-${Array.from(crypto.getRandomValues(new Uint8Array(8)),
  byte => byte.toString(16).padStart(2,"0")).join("")}`;
const saves = new Map();
let saveId = 0;
const TRAINING_DEFAULTS = Object.freeze({
  total_steps:16384,n_envs:4,target_speed:1,seed:7,
  hidden_size:64,hidden_layers:2,learning_rate:.0003,batch_size:128,epochs:4,
  rollout_steps:128,gamma:.99,entropy_coef:.005,clip_range:.2,
  target_kl:.03,best_checkpoint:true,evaluation_interval:16384,evaluation_episodes:3,
});
async function init() {
  if (!runtime) runtime = (async () => {
    importScripts("./vendor/tf.min.js", "./vendor/tf-backend-webgpu.min.js");
    const physics = await import("./browser-physics.mjs");
    const policy = await import("./browser-policy.mjs");
    const imported = await import("./imported-physics.mjs");
    const tasks = await import("./task-regimes.mjs");
    const curriculum = await import("./learning-curriculum.mjs");
    const morphology = await import("./morphology.mjs");
    const mj = await physics.loadPhysics();
    const hardware = {...await policy.selectBackend(),physics_device:"cpu"};
    return {...physics,...policy,...imported,...tasks,...curriculum,...morphology,mj,hardware};
  })();
  return runtime;
}
function number(value, fallback, min, max, integer=false) {
  const result=value ?? fallback;
  if (!Number.isFinite(result) || result<min || result>max || (integer && !Number.isInteger(result)))
    throw new Error(`Expected ${integer ? "an integer" : "a number"} between ${min} and ${max}.`);
  return result;
}
function configFor(input,r,restoring=false) {
  const curriculumMode=input.curriculum_mode??"speed";
  if(!["speed","staged"].includes(curriculumMode))throw new Error("Curriculum must be speed or staged.");
  const task=input.task==null?(curriculumMode==="staged"?r.validateTask({preset:"velocity"}):null):r.validateTask(input.task);
  if(input.curriculum!==false&&curriculumMode==="staged"&&task?.preset!=="velocity")throw new Error("Staged locomotion curriculum requires the Velocity objective.");
  return {body:validateBody(input.body,r),environment:r.validateEnvironment(input.environment),
    ...(input.morphology!=null?{morphology:r.validateMorphology(input.morphology)}:{}),
    task,curriculum_mode:curriculumMode,
    total_steps:number(input.total_steps,TRAINING_DEFAULTS.total_steps,4096,2000000,true),
    n_envs:number(input.n_envs,TRAINING_DEFAULTS.n_envs,1,32,true),
    target_speed:number(input.target_speed,TRAINING_DEFAULTS.target_speed,.1,3),
    seed:number(input.seed,TRAINING_DEFAULTS.seed,0,2147483647,true),
    hidden_size:number(input.hidden_size,TRAINING_DEFAULTS.hidden_size,restoring ? 4 : 16,256,true),
    hidden_layers:number(input.hidden_layers,TRAINING_DEFAULTS.hidden_layers,1,4,true),
    learning_rate:number(input.learning_rate,TRAINING_DEFAULTS.learning_rate,.00001,.01),
    batch_size:number(input.batch_size,TRAINING_DEFAULTS.batch_size,restoring ? 2 : 16,restoring ? 4096 : 2048,true),
    epochs:number(input.epochs,TRAINING_DEFAULTS.epochs,1,restoring ? 32 : 20,true),
    rollout_steps:number(input.rollout_steps,TRAINING_DEFAULTS.rollout_steps,16,1024,true),
    gamma:number(input.gamma,TRAINING_DEFAULTS.gamma,.8,.9999),
    entropy_coef:number(input.entropy_coef,TRAINING_DEFAULTS.entropy_coef,0,.1),
    clip_range:number(input.clip_range,TRAINING_DEFAULTS.clip_range,.05,.4),
    target_kl:number(input.target_kl,TRAINING_DEFAULTS.target_kl,0,1),
    best_checkpoint:input.best_checkpoint!==false,
    evaluation_interval:number(input.evaluation_interval,TRAINING_DEFAULTS.evaluation_interval,4096,2000000,true),
    evaluation_episodes:number(input.evaluation_episodes,TRAINING_DEFAULTS.evaluation_episodes,1,8,true),
    randomize:input.randomize!==false,curriculum:input.curriculum!==false,device:"auto"};
}
function validateBody(body,r){return body?.kind==="mjcf"?r.validateImportedBody(body):r.validateBody(body);}
function bodyXML(config,r){return config.body.kind==="mjcf"?config.body.xml:r.exportMJCF(config.body,config.environment);}
function makeEnv(config,r,overrides={}){
  const options={target_speed:config.target_speed,seed:config.seed,randomize:config.randomize,task:config.task,...overrides};
  if(config.body.kind==="mjcf"&&config.task?.preset==="walk")throw new Error("Coordinated walking currently supports template bodies only; choose Velocity for imported rigs.");
  if(config.body.kind==="mjcf")return new r.ImportedWalker(r.mj,config.body,config.environment,options);
  const env=new r.BrowserWalker(r.mj,config.body,config.environment,options);
  return config.task==null?env:new r.TaskWalker(env,config.task);
}
const physicsSummaries=new WeakMap();
function physicsSummary(env,r){
  if(physicsSummaries.has(env))return physicsSummaries.get(env);
  const m=env.model,d=env.data,base=env.env||env;
  const tracked=env.trackedId??base._root??base._torsoId;
  if(!d||!Number.isInteger(tracked)||tracked<1)return undefined;
  let root=tracked;
  while(m.body_parentid[root]>0)root=m.body_parentid[root];
  const members=new Set();
  for(let i=1;i<m.nbody;i++){
    let ancestor=i;
    while(m.body_parentid[ancestor]>0)ancestor=m.body_parentid[ancestor];
    if(ancestor===root)members.add(i);
  }
  // Nominal masses survive domain randomization. Only the tracked body's tree
  // contributes: the floor, scenery and other independent robots are excluded.
  const mass=base._base?.body_mass||base._baseMass||m.body_mass;
  let totalMass=0,movingBodies=0,articulatedJoints=0,floatingBase=false;
  for(const body of members){
    totalMass+=mass[body];
    let ancestor=body;
    while(ancestor>0&&!m.body_dofnum[ancestor])ancestor=m.body_parentid[ancestor];
    if(ancestor>0)movingBodies++;
  }
  for(let i=0;i<m.njnt;i++){
    if(!members.has(m.jnt_bodyid[i]))continue;
    if(m.jnt_bodyid[i]!==root)articulatedJoints++;
    if(m.jnt_bodyid[i]===root&&m.jnt_type[i]===r.mj.mjtJoint.mjJNT_FREE.value)floatingBase=true;
  }
  const actuated=new Set();
  for(let i=0;i<(m.nactuator??m.nu);i++){
    // Joint and joint-in-parent transmissions identify a joint directly.
    // Tendon/site transmissions are deliberately not guessed.
    if(m.actuator_trntype[i]!==0&&m.actuator_trntype[i]!==1)continue;
    const joint=m.actuator_trnid[i*2];
    if(joint>=0&&members.has(m.jnt_bodyid[joint]))actuated.add(joint);
  }
  const geoms=[];
  for(let i=0;i<m.ngeom;i++)
    if(members.has(m.geom_bodyid[i])&&m.geom_type[i]!==r.mj.mjtGeom.mjGEOM_PLANE.value)geoms.push(i);
  const collisions=geoms.filter(i=>m.geom_contype[i]||m.geom_conaffinity[i]);
  const minimum=[Infinity,Infinity,Infinity],maximum=[-Infinity,-Infinity,-Infinity];
  for(const i of collisions.length?collisions:geoms){
    // MuJoCo's compiled local AABB handles primitive and mesh geometry alike.
    // Rotating that box gives conservative world extents, not anatomical lengths.
    for(let axis=0;axis<3;axis++){
      let center=d.geom_xpos[i*3+axis],half=0;
      for(let local=0;local<3;local++){
        const rotation=d.geom_xmat[i*9+axis*3+local];
        center+=rotation*m.geom_aabb[i*6+local];
        half+=Math.abs(rotation)*m.geom_aabb[i*6+3+local];
      }
      if(Number.isFinite(center)&&Number.isFinite(half)){
        minimum[axis]=Math.min(minimum[axis],center-half);
        maximum[axis]=Math.max(maximum[axis],center+half);
      }
    }
  }
  const extent=minimum.map((value,i)=>Number.isFinite(value)?maximum[i]-value:0);
  const summary={total_mass:totalMass,tracked_height:d.xpos[tracked*3+2],
    body_height:extent[2],extent,floating_base:floatingBase,
    articulated_joints:articulatedJoints,actuated_joints:actuated.size,moving_bodies:movingBodies,
    bounds_source:collisions.length?"compiled_collision_aabb":"compiled_geometry_aabb"};
  // Keep the initial preview dimensions stable in later telemetry/checkpoints.
  physicsSummaries.set(env,summary);
  return summary;
}
function modelInfo(env,r){
  if(!env.model)return undefined;
  const m=env.model,name=(type,i)=>r.mj.mj_id2name(m,r.mj.mjtObj[type].value,i)||`${type.slice(6).toLowerCase()}_${i}`;
  let observations=env.observationNames;
  if(!observations){
    observations=[...Array.from({length:m.nq-1},(_,i)=>`qpos[${i+1}]`),...Array.from({length:m.nv},(_,i)=>`qvel[${i}]`),'target_speed',
      ...['legs','torso_length','torso_radius','upper_length','lower_length','leg_radius','body_mass','motor_strength','joint_damping'].map(k=>`normalized_morphology:${k}`),
      'normalized_mass_scale','normalized_motor_scale','normalized_friction_scale'];
    if(env.observationSchema==='rlsim-planar-gait-v1')observations.push('gait_phase_sin','gait_phase_cos',...Array.from({length:env.body.legs},(_,i)=>`foot_contact[${i}]`),...Array.from({length:m.nu},(_,i)=>`previous_action[${i}]`));
    if(['rlsim-planar-task-v1','rlsim-planar-gait-v1'].includes(env.observationSchema))observations.push('tracked_position.x','tracked_position.y','tracked_position.z','target_delta.x','target_delta.y','target_delta.z','body_velocity.x','body_velocity.y','body_velocity.z');
  }
  return {bodies:Array.from({length:m.nbody-1},(_,i)=>({name:name('mjOBJ_BODY',i+1)})),
    joints:Array.from({length:m.njnt},(_,i)=>({name:name('mjOBJ_JOINT',i),type:['free','ball','slide','hinge'][m.jnt_type[i]]})),
    actuators:Array.from({length:m.nu},(_,i)=>({name:env.actionNames?.[i]??name('mjOBJ_ACTUATOR',i),normalized_range:[-1,1],
      control_range:m.actuator_ctrllimited[i]?Array.from(m.actuator_ctrlrange.subarray(i*2,i*2+2)):[-1,1]})),
    observation_size:env.observationSize,action_size:env.actionSize,observation_schema:env.observationSchema||'rlsim-planar-v1',
    action_schema:env.actionSchema||'rlsim-actuator-controls-v1',
    observation_names:observations,action_names:env.actionNames||Array.from({length:m.nu},(_,i)=>name('mjOBJ_ACTUATOR',i)),
    timestep:env.dt,tracked_body:env.trackedBody||env.task?.tracked_body||'torso',
    physics_summary:physicsSummary(env,r)};
}
function preview(payload,r){
  const body=validateBody(payload.body,r),environment=r.validateEnvironment(payload.environment),task=payload.task==null?null:r.validateTask(payload.task);
  const env=makeEnv({body,environment,task,target_speed:payload.target_speed||1,seed:payload.seed||7,randomize:false},r);
  try{return {body,environment,task,frame:env.getFrame(),model_info:modelInfo(env,r),action_size:env.actionSize,observation_size:env.observationSize,warnings:body.metadata?.notes||[]};}finally{env.close();}
}
let publishedRun, publishedModelInfo;
function statusSnapshot(includeConfig=false,includeModelInfo=false){
  // Imported visual assets are immutable during a run and can be many MB.
  // Send them once, not with every telemetry sample and status poll.
  const {config,model_info,...snapshot}=status;
  if(includeConfig&&config)snapshot.config=config;
  if(includeModelInfo&&model_info)snapshot.model_info=model_info;
  return snapshot;
}
function publish() {
  const newRun=status.run_id!==publishedRun;
  send("status",{status:statusSnapshot(newRun,newRun||status.model_info!==publishedModelInfo)});
  publishedRun=status.run_id;publishedModelInfo=status.model_info;
}
function save(record) {
  const token=++saveId;
  return new Promise((resolve,reject)=>{
    const timeout=setTimeout(()=>{saves.delete(token);reject(new Error("Local model storage timed out. Keep this tab open and check browser storage permissions."));},30000);
    saves.set(token,{resolve,reject,timeout});send("checkpoint",{token,record});
  });
}
async function evaluateController(config,r,policy,normalizer,cancel=()=>false) {
  const env=makeEnv(config,r,{randomize:false}),episodes=[];
  try {
    for(let i=0;i<config.evaluation_episodes;i++){
      // Separate from the consecutive seeds used by the training environments.
      const seed=(config.seed+1000003+i*997)%2147483647;
      let obs=env.reset(seed),reward=0,error=0,effort=0,steps=0,fell=false,first=env.getFrame().position;
      const limit=env.max_episode_steps||1000;
      for(;steps<limit;){
        if(cancel())return null;
        const inputs=[normalizer.normalize(obs)];
        const action=policy.predict? (await policy.predict(inputs))[0] : (await policy.act(inputs,true)).actions[0];
        const result=env.step(action);obs=result.obs;reward+=result.reward;
        error+=Math.abs(result.info.velocity-config.target_speed);
        effort+=result.info.energy_cost??0;steps++;
        fell ||= Boolean(result.info.fallen??result.terminated);
        if(result.terminated||result.truncated)break;
        if(steps%32===0)await pause();
      }
      const frame=env.getFrame();
      episodes.push({seed,return:reward,seconds:frame.time??steps*env.dt,
        speed_error:error/steps,mean_squared_action:effort/steps,distance:frame.position-first,fell,...((env.motion||env.env?.motion)?{gait:(env.motion||env.env.motion).summary(),effort:(env.effort||env.env.effort).summary(frame.position-first,env.model.body_mass.reduce((a,b)=>a+b,0),config.environment.gravity)}: {})});
    }
    const mean=key=>episodes.reduce((sum,e)=>sum+e[key],0)/episodes.length;
    return {mean_return:mean("return"),seconds:mean("seconds"),speed_error:mean("speed_error"),
      mean_squared_action:mean("mean_squared_action"),distance:mean("distance"),
      survival_rate:episodes.filter(e=>!e.fell).length/episodes.length,episodes};
  }finally{env.close();}
}
function trainingSignature(config){
  const {total_steps,device,...settings}=config;
  const sorted=value=>Array.isArray(value)?value.map(sorted):value&&typeof value==='object'
    ?Object.fromEntries(Object.keys(value).sort().map(key=>[key,sorted(value[key])])):value;
  // A compatibility checksum, not an authenticity/security signature. Avoid
  // duplicating embedded meshes and textures in every training checkpoint.
  const text=JSON.stringify(sorted(settings));let a=2166136261,b=5381;
  for(let i=0;i<text.length;i++){const c=text.charCodeAt(i);a=Math.imul(a^c,16777619)>>>0;b=(Math.imul(b,33)^c)>>>0;}
  return `v1:${text.length}:${a}:${b}`;
}
function continuationFor(record,config){
  const saved=record?.continuation;
  if(saved?.format!=='rlsim-training-state-v1'||record.controller||!record.policy)throw new Error('This model is inference-only. It has no complete training state.');
  number(record.steps,undefined,0,Number.MAX_SAFE_INTEGER-2000000,true);
  if(saved.steps!==record.steps||!Array.isArray(saved.environment_random)||saved.environment_random.length!==config.n_envs)throw new Error('Training state does not match this checkpoint.');
  saved.environment_random.forEach(value=>number(value,undefined,0,0xffffffff,true));
  number(saved.schedule_steps,undefined,4096,2000000,true);
  number(record.policy.randomState,undefined,0,0xffffffff,true);
  if(saved.config_signature!==trainingSignature(config))throw new Error('The saved body or training settings changed. Start a new training run.');
  return saved;
}
async function train(config,r,resume=null) {
  const envs=[],start=performance.now(),returns=[],episodeReturns=new Array(config.n_envs).fill(0);
  const initialSteps=resume?.steps||0,totalSteps=initialSteps+config.total_steps;
  const saved=resume?continuationFor(resume,config):null,scheduleSteps=saved?.schedule_steps??config.total_steps;
  let policy,normalizer,steps=initialSteps,lastPublish=0,lastSave=initialSteps,lastEvaluation=initialSteps,best=resume?.best??null;
  const curriculum=config.curriculum&&config.curriculum_mode==="staged"?new r.LearningCurriculum(config.target_speed):null;
  const episodeStats=Array.from({length:config.n_envs},()=>({steps:0,error:0,fell:false}));
  try {
    for(let i=0;i<config.n_envs;i++) envs.push(makeEnv(config,r,{seed:config.seed+i}));
    if(!envs[0].actionSize)throw new Error("This body has no actuators. Enable automatic motors during import or add actuators to the MJCF.");
    policy=resume?await r.PPO.continue(resume.policy,saved.optimizer):new r.PPO(envs[0].observationSize,envs[0].actionSize,config);
    if(resume&&(policy.obsSize!==envs[0].observationSize||policy.actionSize!==envs[0].actionSize))throw new Error('Training policy dimensions do not match the body.');
    status.model_info=modelInfo(envs[0],r);
    normalizer=resume?r.RunningNormalizer.restore(resume.normalizer):new r.RunningNormalizer(envs[0].observationSize);
    if(resume&&normalizer.size!==envs[0].observationSize)throw new Error('Training normalization does not match the body.');
    if(curriculum&&saved)curriculum.restore(saved.curriculum);
    let observations=envs.map((env,i)=>{
      if(curriculum)env.setLearningStage(curriculum.stage);
      else if(config.curriculum)env.setTrainingProgress(Math.min(1,steps/(scheduleSteps*.6)));
      if(saved){(env.random||env.env?.random).setState(saved.environment_random[i]);return env.reset();}
      return env.reset(config.seed+i);
    });
    if(curriculum)status.curriculum=curriculum.snapshot();
    normalizer.update(observations);
    const checkpoint=async finalStatus=>{
      const record={format:"morpholab-browser-run-v1",id:status.run_id,created_at:status.created_at,
        config,steps,status:finalStatus,hardware:r.hardware,metrics:status.metrics,history:status.history,
        checkpoint_available:true,policy:await policy.serialize(),normalizer:normalizer.serialize(),
        ...(policy.trainingState?{continuation:{format:'rlsim-training-state-v1',steps,schedule_steps:scheduleSteps,config_signature:trainingSignature(config),
          optimizer:await policy.trainingState(),environment_random:envs.map((env,i)=>(env.random||env.env?.random)?.getState?.()??config.seed+i),
          curriculum:curriculum?.serialize()??null},resumable:true}:{}),
        ...(resume?{parent_run_id:resume.id,start_steps:initialSteps}:{}),
        ...(best?{best}:{}),validation:status.validation??null,
        curriculum_state:curriculum?.snapshot()??null,
        model_info:modelInfo(envs[0],r),body_xml:bodyXML(config,r)};
      await save(record);lastSave=steps;
    };
    const validate=async()=>{
      status={...status,phase:"evaluating"};publish();
      const evaluation=await evaluateController(config,r,policy,normalizer,()=>stopping);
      if(evaluation){
        lastEvaluation=steps;
        if(!best || evaluation.mean_return>best.validation.mean_return){
          best={steps,policy:await policy.serialize(),normalizer:normalizer.serialize(),validation:evaluation};
        }
        status={...status,validation:evaluation,best_checkpoint_steps:best.steps,best_validation:best.validation};
      }
      status={...status,phase:"training"};publish();
    };
    while(steps<totalSteps && !stopping) {
      const batch={observations:[],actions:[],oldLogProbs:[],advantages:[],returns:[]};
      const rows=[];
      for(let t=0;t<config.rollout_steps && steps<totalSteps && !stopping;t++) {
        const inputs=normalizer.normalize(observations);
        const sampled=await policy.act(inputs);
        const results=envs.map((env,i)=>env.step(sampled.actions[i]));
        // Terminal states bootstrap only at time limits, never after a fall.
        const timeoutIndices=results.flatMap((result,i)=>result.truncated && !result.terminated ? [i] : []);
        const timeoutValues=timeoutIndices.length ? (await policy.act(timeoutIndices.map(i=>normalizer.normalize(results[i].obs)),true)).values : [];
        const row=[];
        for(let i=0;i<envs.length;i++) {
          const result=results[i],done=result.terminated || result.truncated;
          const stats=episodeStats[i];stats.steps++;stats.error+=Math.abs(result.info.velocity-(curriculum?config.target_speed*curriculum.stage.speed_fraction:config.target_speed));
          stats.fell ||= result.terminated || Boolean(result.info.fallen);
          if(done&&curriculum){curriculum.record({fell:stats.fell,speed_error:stats.error/stats.steps});episodeStats[i]={steps:0,error:0,fell:false};}
          const timeoutIndex=timeoutIndices.indexOf(i);
          row.push({value:sampled.values[i],reward:result.reward,done,
            timeoutValue:timeoutIndex>=0 ? timeoutValues[timeoutIndex] : 0});
          batch.observations.push(inputs[i]);batch.actions.push(sampled.actions[i]);batch.oldLogProbs.push(sampled.logProbs[i]);
          episodeReturns[i]+=result.reward;
          if(done){returns.push(episodeReturns[i]);if(returns.length>100)returns.shift();episodeReturns[i]=0;}
          const nextObservation=config.curriculum&&!curriculum
            ? envs[i].setTrainingProgress(Math.min(1,(steps+envs.length)/(scheduleSteps*.6)))
            : result.obs;
          observations[i]=done ? envs[i].reset() : nextObservation;
        }
        rows.push(row);normalizer.update(observations);steps+=envs.length;
        const now=performance.now();
        if(now-lastPublish>350) {
          status={...status,status:stopping?"stopping":"training",steps,elapsed:(now-start)/1000,frame:envs[0].getFrame(),
            metrics:{mean_reward:returns.length ? returns.reduce((a,b)=>a+b,0)/returns.length : null,
              speed:results[0].info.velocity,fps:(steps-initialSteps)/((now-start)/1000)}};
          publish();lastPublish=now;
        }
        if(t%8===0)await pause();
      }
      if(batch.actions.length >= 2 && !stopping) {
        const next=(await policy.act(normalizer.normalize(observations),true)).values;
        const gae=new Array(envs.length).fill(0);
        batch.advantages=new Array(batch.actions.length);batch.returns=new Array(batch.actions.length);
        for(let t=rows.length-1;t>=0;t--)for(let i=0;i<envs.length;i++) {
          const row=rows[t][i],index=t*envs.length+i;
          const nextValue=row.done ? row.timeoutValue : (t===rows.length-1 ? next[i] : rows[t+1][i].value);
          const delta=row.reward+config.gamma*nextValue-row.value;
          gae[i]=delta+(row.done ? 0 : config.gamma*.95*gae[i]);
          batch.advantages[index]=gae[i];batch.returns[index]=gae[i]+row.value;
        }
        const losses=await policy.update(batch);
        if(curriculum?.advance()){
          observations=envs.map((env,i)=>{
            env.setLearningStage(curriculum.stage);episodeReturns[i]=0;episodeStats[i]={steps:0,error:0,fell:false};
            return env.reset();
          });
          normalizer.update(observations);
        }
        if(curriculum)status.curriculum=curriculum.snapshot();
        const mean=returns.length ? returns.reduce((a,b)=>a+b,0)/returns.length : null;
        status={...status,steps,elapsed:(performance.now()-start)/1000,metrics:{...status.metrics,...losses,mean_reward:mean},
          history:[...status.history,{step:steps,reward:mean,mean_reward:mean}].slice(-1000),frame:envs[0].getFrame()};
        publish();
        if(config.best_checkpoint && steps-lastEvaluation>=config.evaluation_interval){
          await validate();await checkpoint("stopped");
        }
        if(steps-lastSave>=16384)await checkpoint("stopped");
      }
      await pause();
    }
    if(config.best_checkpoint && !stopping && lastEvaluation!==steps)await validate();
    const elapsed=(performance.now()-start)/1000;
    status={...status,status:"stopping",steps,elapsed,
      metrics:{...status.metrics,fps:(steps-initialSteps)/Math.max(elapsed,.001)}};publish();
    const finished=stopping ? "stopped" : "completed";
    await checkpoint(finished);
    status={...status,status:finished,checkpoint_available:true};publish();
  } catch(error) {
    status={...status,status:"failed",steps,error:error.message || String(error)};publish();
  } finally {
    for(const env of envs)env.close();policy?.dispose();running=false;
  }
}
let inferenceCache=null;
async function clearInferenceCache(){
  const cached=inferenceCache;inferenceCache=null;
  if(cached){cached.model.env?.close();await cached.model.policy.dispose();}
}
function selectedCheckpoint(record,selection="best"){
  if(!["best","latest"].includes(selection))throw new Error("Checkpoint must be best or latest.");
  // Optimizer slots are for training only; keep them out of the inference key.
  const {continuation,...inferenceRecord}=record;record=inferenceRecord;
  if(selection==="best"&&record.best){
    if(!record.best.policy||!record.best.normalizer)throw new Error("The best checkpoint is incomplete.");
    return {...record,policy:record.best.policy,normalizer:record.best.normalizer,
      checkpoint_steps:record.best.steps,validation:record.best.validation,best:undefined,checkpoint:"best"};
  }
  return {...record,best:undefined,checkpoint:"latest",checkpoint_steps:record.steps};
}
async function restore(record,r,replayInput=null,cancelled=()=>false) {
  if(!record || record.format!=="morpholab-browser-run-v1")throw new Error("Select an RLsim browser model JSON file.");
  // Early browser records stored network options only inside the policy.
  // Fill those gaps before applying defaults; keep their trained architecture.
  const config=configFor({...record.policy?.config,...record.config},r,true);
  if(replayInput?.proportions!=null)config.body=r.proportionBody(config,replayInput.proportions,r.validateBody);
  const environment=replayInput?r.validateEnvironment(replayInput.environment||config.environment):config.environment;
  const target_speed=replayInput?number(replayInput.target_speed,config.target_speed||1,.1,3):config.target_speed;
  const seed=replayInput?number(replayInput.seed,10007,0,2147483647,true):config.seed;
  const env=makeEnv({...config,environment,target_speed,seed},r,{randomize:false});
  let retained=false;
  const result=values=>{
    retained=Boolean(replayInput);
    return {...values,...(retained?{env,environment,target_speed,seed}:{})};
  };
  try {
    if(record.controller?.format==='rlsim-onnx-v1'){
      if(!globalThis.ort)importScripts('./vendor/ort/ort.webgpu.min.js');
      const expectedSchema=env.observationSchema||'rlsim-planar-v1';
      if(record.controller.observation_schema&&record.controller.observation_schema!==expectedSchema)throw new Error(`Controller observation schema must be ${expectedSchema}.`);
      if(record.controller.action_schema&&record.controller.action_schema!==(env.actionSchema||'rlsim-actuator-controls-v1'))throw new Error("Controller action schema does not match this body's control mode.");
      const {ONNXPolicy}=await import('./onnx-policy.mjs');
      const policy=await ONNXPolicy.restore(record.controller,env.observationSize,env.actionSize,r.hardware);
      try{
        const normalizer=record.normalizer?r.RunningNormalizer.restore(record.normalizer):{normalize:value=>value};
        if(record.normalizer&&record.normalizer.size!==env.observationSize)throw new Error('Normalization dimensions do not match the body.');
        if(!cancelled())await policy.act([normalizer.normalize(env.reset(config.seed))],true);
        return result({config,policy,normalizer,model_info:modelInfo(env,r)});
      }catch(error){await policy.dispose();throw error;}
    }
    if(record.policy?.obsSize!==env.observationSize || record.policy?.actionSize!==env.actionSize || record.normalizer?.size!==env.observationSize)
      throw new Error("Model dimensions do not match the saved body.");
    const normalizer=r.RunningNormalizer.restore(record.normalizer);
    const policy=await (replayInput&&r.InferenceActor?r.InferenceActor:r.PPO).restore(record.policy);
    try{if(replayInput&&policy.predict&&!cancelled())await policy.predict([normalizer.normalize(env.reset(seed))]);}
    catch(error){await policy.dispose();throw error;}
    return result({config,normalizer,policy,model_info:modelInfo(env,r)});
  } finally {if(!retained)env.close();}
}
async function replay(input,r,progress=()=>{},cancelled=()=>false) {
  let model,env,cacheHit=false,cached=false,succeeded=false,record;
  try {
    if(cancelled())return {cancelled:true,frames:[]};
    if(input.stream)progress({phase:"loading",stage:input.record?"controller":"physics"});
    if(input.record){
      record=selectedCheckpoint(input.record,input.checkpoint);
      // Content identity prevents stale reuse after an import or overwritten
      // checkpoint, including callers that reuse an id. Keep only one entry.
      const key=JSON.stringify([record,input.environment??record.config.environment,input.target_speed??record.config.target_speed,input.proportions??null]);
      if(inferenceCache?.key===key){model=inferenceCache.model;cacheHit=true;}
      else{
        await clearInferenceCache();
        if(cancelled())return {cancelled:true,frames:[]};
        model=await restore(record,r,input,cancelled);
        inferenceCache={key,model};
      }
      cached=true;env=model.env;
    }
    if(cancelled())return {cancelled:true,frames:[]};
    const body=model?.config.body || validateBody(input.body,r);
    const environment=model?.environment||r.validateEnvironment(input.environment);
    const target_speed=model?.target_speed??number(input.target_speed,1,.1,3);
    const seed=number(input.seed,10007,0,2147483647,true);
    env ||= makeEnv({body,environment,task:input.task??null,target_speed,seed,randomize:false},r);
    let obs=env.reset(seed),steps=0,totalReward=0,effort=0,speedError=0,fell=false;
    const first=env.getFrame(),frames=input.stream?[]:[first],initial=first.position,total_steps=env.max_episode_steps||1000;
    const header={source:model?"trained":"untrained",body,environment,target_speed,seed,
      ...(model?{checkpoint:record.checkpoint,checkpoint_steps:record.checkpoint_steps,
        validation:record.validation??null,inference_cache_hit:cacheHit}:{}),
      inference_device:model?(model.policy.inference_device||r.hardware.training_device):null,fallback_reason:model?.policy.fallback_reason||null,
      model_info:model?.model_info||modelInfo(env,r),task:model?model.config.task:input.task??null};
    let last=first,lastPublish=performance.now(),pendingFrames=[];
    const flush=()=>{
      if(pendingFrames.length){progress({phase:"simulating",frames:pendingFrames,steps,total_steps});pendingFrames=[];lastPublish=performance.now();}
    };
    if(input.stream){
      progress({phase:"ready",...header,frames:[first],steps:0,total_steps});
      // Let the initial frame reach the viewer before computing the episode.
      await pause();
    }
    for(;steps<total_steps;steps++) {
      if(cancelled())break;
      const inputs=model?[model.normalizer.normalize(obs)]:null;
      const action=model ? (model.policy.predict?await model.policy.predict(inputs):(await model.policy.act(inputs,true)).actions)[0] : new Array(env.actionSize).fill(0);
      if(cancelled())break;
      const result=env.step(action);obs=result.obs;totalReward+=result.reward;effort+=result.info.energy_cost;
      speedError+=Math.abs((result.info.velocity??0)-target_speed);
      fell ||= Boolean(result.info.fallen??result.terminated);
      if(steps%2===1 || result.terminated || result.truncated){
        last=env.getFrame();
        if(input.stream)pendingFrames.push(last);else frames.push(last);
      }
      if(result.terminated || result.truncated){steps++;break;}
      if(input.stream&&(pendingFrames.length>=32 || performance.now()-lastPublish>=100))flush();
      if(steps%16===0)await pause();
    }
    if(cancelled() && steps && last.time!==env.getFrame().time){
      last=env.getFrame();if(input.stream)pendingFrames.push(last);else frames.push(last);
    }
    if(input.stream)flush();
    const distance=last.position-initial;
    // Streamed requests have already transferred the header and every frame.
    succeeded=!cancelled();
    return {...(input.stream?{streamed:true}: {...header,frames}),
      ...(cancelled()?{cancelled:true}:{}),
      summary:{steps,seconds:last.time,distance,mean_speed:last.time?distance/last.time:0,speed_error:steps?speedError/steps:0,reward:totalReward,mean_squared_action:steps?effort/steps:0,fell,seed,...((env.motion||env.env?.motion)?{gait:(env.motion||env.env.motion).summary(),effort:(env.effort||env.env.effort).summary(distance,env.model.body_mass.reduce((a,b)=>a+b,0),environment.gravity)}: {})}};
  } finally {
    if(cached){if(!succeeded)await clearInferenceCache();}
    else{env?.close();await model?.policy.dispose();}
  }
}
onmessage=async ({data})=>{
  if(data.op==="saved"){
    const pending=saves.get(data.token);if(pending){clearTimeout(pending.timeout);saves.delete(data.token);data.error ? pending.reject(new Error(data.error)) : pending.resolve();}return;
  }
  const {id,op,payload={}}=data;
  // Cancellation bypasses the operation lock and works while init/restore awaits.
  if(op==='stop_replay'){
    const request=replayRequests.get(payload.replay_id);
    if(request)request.cancelled=true;
    postMessage({id,result:{stopping:Boolean(request)}});return;
  }
  const replayRequest=op==='replay'?{cancelled:false}:null;
  if(replayRequest)replayRequests.set(payload.replay_id??id,replayRequest);
  try {
    const r=await init();let result;
    if(op==="config") result={presets:r.PRESETS,hardware:r.hardware,environment_defaults:{gravity:9.81,friction:1},defaults:{...TRAINING_DEFAULTS}};
    else if(op==="status")result=statusSnapshot();
    else if(op==="stop"){if(running){stopping=true;status={...status,status:"stopping"};publish();}result=status;}
    else {
      if(running || operation)throw new Error("Wait for the current operation to finish, or stop training first.");
      operation=true;
      try {
        if(op==="preview"){
          result=preview(payload,r);
        } else if(op==='check_rig'){
          const {checkPhysicalRig}=await import('./rig-checks.mjs');
          result=await checkPhysicalRig(r.mj,payload.body,payload.environment,pause);
        } else if(op==="import_body"){
          const body=await r.preparePhysicsFiles(r.mj,payload.files,payload.options);
          result=preview({body,environment:payload.environment,task:payload.task},r);
        } else if(op==="interface"){
          const config=configFor(payload,r),resultPreview=preview(config,r);
          result={format:'rlsim-controller-interface-v1',config,model_info:resultPreview.model_info,
            input:{type:'float32',shape:[1,resultPreview.observation_size],normalization:'none (embed preprocessing in ONNX or include normalizer in manifest)'},
            output:{type:'float32',shape:[1,resultPreview.action_size],
              semantics:resultPreview.model_info.action_schema==='rlsim-joint-targets-v1'
                ? 'normalized template joint targets [-1,1]; hip = 0.45 + 0.8*a rad, knee = -0.9 + 0.8*a rad; mass-scaled PD control at every physics substep'
                : resultPreview.model_info.action_schema==='rlsim-joint-target-v1'
                ? 'normalized joint targets [-1,1]; zero maps to clamp(0,lower,upper), each sign scales toward its respective control bound; ball targets are exponential-map radians'
                : 'normalized actuator controls in [-1,1], in action_names order'},
            manifest_template:{format:'rlsim-onnx-manifest-v1',model:'controller.onnx',config,
              observation_schema:resultPreview.model_info.observation_schema,action_schema:resultPreview.model_info.action_schema,
              input_name:null,output_name:null,normalizer:null}};
        } else if(op==="train"||op==='resume'){
          await clearInferenceCache();
          const resume=op==='resume'?payload.record:null;
          const config=configFor(resume?{...resume.config,total_steps:payload.additional_steps}:payload,r);
          if(config.morphology?.enabled)throw new Error('Training across body proportions currently requires the native trainer. Browser inference is supported.');
          if(resume)continuationFor(resume,config);
          stopping=false;running=true;
          status={status:"starting",history:resume?.history??[],steps:resume?.steps??0,total_steps:(resume?.steps||0)+config.total_steps,config,hardware:r.hardware,
            ...(resume?{parent_run_id:resume.id,start_steps:resume.steps,best_checkpoint_steps:resume.best?.steps,best_validation:resume.best?.validation}:{}),
            run_id:runId(),created_at:new Date().toISOString(),checkpoint_available:false};
          result=status;publish();setTimeout(()=>train(config,r,resume),0);
        } else if(op==="replay")result=await replay(payload,r,progress=>send("replay_progress",{id,progress}),()=>replayRequest.cancelled);
        else if(op==="validate"){
          const model=await restore(payload,r);
          const inference_device=model.policy.inference_device||r.hardware.training_device;
          await model.policy.dispose();
          if(payload.continuation){
            const saved=continuationFor(payload,model.config);
            if(model.config.curriculum&&model.config.curriculum_mode==='staged')new r.LearningCurriculum(model.config.target_speed).restore(saved.curriculum);
            const trainingPolicy=await r.PPO.continue(payload.policy,saved.optimizer);trainingPolicy.dispose();
          }
          if(payload.best){
            const best=selectedCheckpoint(payload,"best");
            number(best.checkpoint_steps,0,0,Number.MAX_SAFE_INTEGER,true);
            const checked=await restore(best,r);await checked.policy.dispose();
            if(!Number.isFinite(payload.best.validation?.mean_return))throw new Error("Best checkpoint needs a finite validation return.");
          }
          result={...payload,config:model.config,model_info:model.model_info,inference_device,id:runId(),
            created_at:new Date().toISOString(),status:"stopped",steps:number(payload.steps,0,0,Number.MAX_SAFE_INTEGER,true),
            checkpoint_available:true,body_xml:bodyXML(model.config,r)};
        } else throw new Error(`Unknown browser operation: ${op}`);
      } finally {operation=false;}
    }
    postMessage({id,result});
  }catch(error){postMessage({id,error:error.message || String(error)});}
  finally{if(replayRequest)replayRequests.delete(payload.replay_id??id);}
};
