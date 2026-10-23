/** Portable task definitions and a small, non-executable reward-expression language. */
export const TASK_DEFAULTS = Object.freeze({preset:"velocity",episode_seconds:20,target:[2,0,.8],
  tracked_body:"",terminate_on_fall:true,min_height_ratio:.35,max_tilt_degrees:64});
export const DEFAULT_EXPRESSION="1.5*speed_tracking + 0.3*forward_progress + 0.15*alive + 0.15*upright + 0.04*energy + 0.02*smoothness + fall";
const clip=(v,a,b)=>Math.max(a,Math.min(b,v));
const FUNCTIONS={abs:Math.abs,min:Math.min,max:Math.max,sqrt:Math.sqrt,exp:Math.exp,
  log:Math.log,sin:Math.sin,cos:Math.cos,tanh:Math.tanh,pow:Math.pow,clamp:clip};
export const REWARD_VARIABLES=Object.freeze(["speed_tracking","forward_progress","alive","upright",
  "energy","smoothness","fall","target_progress","target_distance","target_success","height_tracking",
  "x","y","z","vx","vy","vz","speed","height","initial_height","target_speed","distance","time","dt","pi"]);
const variableNames=new Set(REWARD_VARIABLES);
const precedence={"||":1,"&&":2,"==":3,"!=":3,"<":4,">":4,"<=":4,">=":4,"+":5,"-":5,"*":6,"/":6,"%":6,"^":7,"**":7};
const compiled=new Map();
export function compileReward(source) {
  if(typeof source!=="string" || !source.trim() || source.length>2000)throw new Error("Reward expression must contain 1–2,000 characters.");
  if(compiled.has(source))return compiled.get(source);
  const tokens=[];let cursor=0;
  while(cursor<source.length){
    const rest=source.slice(cursor),space=/^\s+/.exec(rest);if(space){cursor+=space[0].length;continue;}
    const number=/^(?:\d+\.?\d*|\.\d+)(?:e[+-]?\d+)?/i.exec(rest);
    const name=/^[A-Za-z_][A-Za-z_0-9]*/.exec(rest);
    const operator=/^(?:\*\*|<=|>=|==|!=|&&|\|\||[+\-*/%^(),?:!<>])/.exec(rest);
    const value=number?.[0] || name?.[0] || operator?.[0];
    if(!value)throw new Error(`Invalid reward syntax near “${rest.slice(0,16)}”.`);
    tokens.push({value,type:number?"number":name?"name":"operator"});cursor+=value.length;
    if(tokens.length>512)throw new Error("Reward expression is too complex.");
  }
  let index=0,depth=0;
  const take=value=>tokens[index]?.value===value ? (++index,true) : false;
  const need=value=>{if(!take(value))throw new Error(`Expected “${value}” in reward expression.`);};
  function parse(minimum=0){
    if(++depth>64)throw new Error("Reward expression is nested too deeply.");
    let node;const token=tokens[index++];
    if(!token)throw new Error("Incomplete reward expression.");
    if(token.type==="number") {const n=Number(token.value);if(!Number.isFinite(n))throw new Error("Reward constants must be finite.");node={kind:"number",value:n};}
    else if(token.value==="("){node=parse();need(")");}
    else if(["+","-","!"].includes(token.value))node={kind:"unary",op:token.value,arg:parse(8)};
    else if(token.type==="name") {
      if(take("(")){
        if(!Object.hasOwn(FUNCTIONS,token.value))throw new Error(`Unknown reward function: ${token.value}.`);
        const args=[];if(!take(")")){do{args.push(parse());}while(take(","));need(")");}
        if(!args.length || args.length>8)throw new Error("Reward functions require 1–8 arguments.");
        node={kind:"call",name:token.value,args};
      }else{
        if(!variableNames.has(token.value))throw new Error(`Unknown reward variable: ${token.value}.`);
        node={kind:"variable",name:token.value};
      }
    }else throw new Error(`Unexpected token “${token.value}”.`);
    while(index<tokens.length){
      const op=tokens[index].value,p=precedence[op];if(p===undefined || p<minimum)break;
      index++;node={kind:"binary",op,left:node,right:parse(p+(["^","**"].includes(op)?0:1))};
    }
    if(minimum===0 && take("?")){const yes=parse();need(":");node={kind:"conditional",condition:node,yes,no:parse()};}
    depth--;return node;
  }
  const tree=parse();if(index!==tokens.length)throw new Error(`Unexpected token “${tokens[index].value}”.`);
  function evaluate(node,variables){
    if(node.kind==="number")return node.value;
    if(node.kind==="variable")return variables[node.name];
    if(node.kind==="call")return FUNCTIONS[node.name](...node.args.map(arg=>evaluate(arg,variables)));
    if(node.kind==="conditional")return evaluate(node.condition,variables)?evaluate(node.yes,variables):evaluate(node.no,variables);
    const a=evaluate(node.kind==="unary"?node.arg:node.left,variables);
    if(node.kind==="unary")return node.op==="-"?-a:node.op==="!"?Number(!a):a;
    if(node.op==="&&")return a?Number(Boolean(evaluate(node.right,variables))):0;
    if(node.op==="||")return a?1:Number(Boolean(evaluate(node.right,variables)));
    const b=evaluate(node.right,variables);
    switch(node.op){case "+":return a+b;case "-":return a-b;case "*":return a*b;case "/":return a/b;
      case "%":return a%b;case "^":case "**":return a**b;case "<":return Number(a<b);case ">":return Number(a>b);
      case "<=":return Number(a<=b);case ">=":return Number(a>=b);case "==":return Number(a===b);case "!=":return Number(a!==b);}
  }
  const result=variables=>{const value=evaluate(tree,variables);if(!Number.isFinite(value))throw new Error("Custom reward produced a non-finite value. Check divisions and function domains.");return value;};
  if(compiled.size>=64)compiled.delete(compiled.keys().next().value);compiled.set(source,result);return result;
}
export function validateTask(input){
  if(input==null)input={};
  if(typeof input!=="object" || Array.isArray(input))throw new Error("Task settings must be an object.");
  const allowed=new Set([...Object.keys(TASK_DEFAULTS),"expression","weights","foot_drift_weight","mechanical_work_weight","foot_only_support"]);
  for(const key of Object.keys(input))if(!allowed.has(key))throw new Error(`Unknown task setting: ${key}.`);
  const task={...TASK_DEFAULTS,...input,target:[...(input.target || TASK_DEFAULTS.target)]};
  if(!["velocity","walk","balance","target","custom"].includes(task.preset))throw new Error("Unknown task preset.");
  for(const [key,lo,hi] of [["episode_seconds",1,120],["min_height_ratio",0,1],["max_tilt_degrees",0,180]])
    if(!Number.isFinite(task[key]) || task[key]<lo || task[key]>hi)throw new Error(`${key} must be between ${lo} and ${hi}.`);
  if(task.target.length!==3 || !task.target.every(x=>Number.isFinite(x)&&Math.abs(x)<=1000))throw new Error("Task target must contain three finite coordinates between −1,000 and 1,000.");
  if(typeof task.tracked_body!=="string" || task.tracked_body.length>200)throw new Error("Invalid tracked body name.");
  if(typeof task.terminate_on_fall!=="boolean")throw new Error("terminate_on_fall must be true or false.");
  if(task.mechanical_work_weight!=null&&(task.preset!=="walk"||!Number.isFinite(task.mechanical_work_weight)||task.mechanical_work_weight<0||task.mechanical_work_weight>2))throw new Error("Mechanical work weight requires Coordinated walking and a value from 0 to 2.");
  if(task.foot_only_support!=null&&(task.preset!=="walk"||typeof task.foot_only_support!=="boolean"))throw new Error("Foot-only support requires Coordinated walking and a boolean.");
  if(task.mechanical_work_weight==null)delete task.mechanical_work_weight;
  if(task.foot_only_support==null)delete task.foot_only_support;
  if(task.foot_drift_weight!=null&&(task.preset!=="walk"||!Number.isFinite(task.foot_drift_weight)||task.foot_drift_weight<0||task.foot_drift_weight>5))throw new Error("Foot drift weight requires Coordinated walking and a value from 0 to 5.");
  if(task.foot_drift_weight==null)delete task.foot_drift_weight;
  if(task.preset==="walk"&&(!["","torso"].includes(task.tracked_body)||task.weights!=null))throw new Error("Coordinated walking uses the template torso and its built-in reward weights.");
  if(task.preset==="custom"){task.expression=input.expression || DEFAULT_EXPRESSION;compileReward(task.expression);}
  else delete task.expression;
  if(task.weights){
    if(typeof task.weights!=="object" || Array.isArray(task.weights))throw new Error("Reward weights must be an object.");
    for(const [key,value] of Object.entries(task.weights))if(!REWARD_VARIABLES.slice(0,11).includes(key) || !Number.isFinite(value) || Math.abs(value)>100)
      throw new Error(`Invalid reward weight: ${key}.`);
    task.weights={...task.weights};
  }
  return task;
}
export function evaluateTask(task,state){
  const velocity=typeof state.velocity==="number"?[state.velocity,0,0]:state.velocity;
  const [x,y,z]=state.position,[vx,vy,vz]=velocity;
  const distance=Math.hypot(...task.target.map((v,i)=>v-state.position[i]));
  const fallen=state.height<Math.max(.01,state.initial_height*task.min_height_ratio)
    || state.upright<Math.cos(task.max_tilt_degrees*Math.PI/180);
  const target=state.target_speed || 1,denominator=Math.max(Math.abs(target),.5);
  const terms={speed_tracking:Math.exp(-2*((vx-target)/denominator)**2),forward_progress:clip(vx/denominator,-1,1),
    alive:fallen?0:1,upright:fallen?0:Math.max(state.upright,0),energy:-state.energy,smoothness:-state.smoothness,
    fall:fallen?-1:0,target_progress:clip(((state.previous_distance??distance)-distance)/state.dt,-10,10),
    target_distance:-distance,target_success:distance<.2?1:0,
    height_tracking:Math.exp(-2*((state.height-state.initial_height)/Math.max(Math.abs(state.initial_height),.1))**2)};
  const weights=task.weights || (task.preset==="balance" ? {alive:.2,upright:1,height_tracking:.5,energy:.03,smoothness:.01,fall:1}
    : task.preset==="target" ? {alive:.05,upright:.1,target_progress:1,target_distance:.05,target_success:1,energy:.02,smoothness:.01,fall:1}
    : {speed_tracking:1.5,forward_progress:.3,alive:.15,upright:.15,energy:.04,smoothness:.02,fall:1});
  const components=Object.fromEntries(Object.entries(weights).map(([key,value])=>[key,value*terms[key]]));
  const reward=task.preset==="custom"?compileReward(task.expression)({...terms,x,y,z,vx,vy,vz,speed:Math.hypot(...velocity),
    height:state.height,initial_height:state.initial_height,target_speed:target,distance,time:state.elapsed,dt:state.dt,pi:Math.PI})
    :Object.values(components).reduce((a,b)=>a+b,0);
  if(!Number.isFinite(reward))throw new Error("The task produced a non-finite reward.");
  return {reward,terminated:task.terminate_on_fall&&fallen,components,terms,distance,fallen};
}

/** Add configurable tasks to the existing planar environment without changing legacy checkpoints. */
export class TaskWalker {
  constructor(env,task){
    this.env=env;this.task=validateTask(task);this.body=env.body;this.dt=env.dt;
    this.actionSize=env.actionSize;this.observationSize=env.observationSize+9;
    this.observationSchema=env.coordinated?"rlsim-planar-gait-v1":"rlsim-planar-task-v1";
    this.actionSchema=env.coordinated?"rlsim-joint-targets-v1":"rlsim-actuator-controls-v1";
    this.max_episode_steps=Math.max(1,Math.round(this.task.episode_seconds/this.dt));
    this.trackedId=this.task.tracked_body ? env.mj.mj_name2id(env.model,env.mj.mjtObj.mjOBJ_BODY.value,this.task.tracked_body) : env._torsoId;
    if(this.trackedId<1){env.close();throw new Error(`Tracked body “${this.task.tracked_body}” was not found.`);}
    this.reset();
  }
  get model(){return this.env.model;}get data(){return this.env.data;}get mj(){return this.env.mj;}
  _position(){return Array.from(this.env.data.xpos.subarray(this.trackedId*3,this.trackedId*3+3));}
  _augment(obs){const result=new Float32Array(this.observationSize);result.set(obs);result.set([...this.position,...this.task.target.map((v,i)=>v-this.position[i]),...this.velocity],obs.length);return result;}
  reset(seed){const obs=this.env.reset(seed);this.steps=0;this.position=this._position();this.velocity=[0,0,0];this.initial_height=this.position[2];
    const offset=this.trackedId*9;this.initialUp=Array.from(this.env.data.xmat.subarray(offset+6,offset+9));
    this.distance=Math.hypot(...this.task.target.map((v,i)=>v-this.position[i]));return this._augment(obs);}
  setTrainingProgress(p){return this._augment(this.env.setTrainingProgress(p));}
  setLearningStage(stage){
    this._requestedTask??=this.task;this.task={...this._requestedTask,preset:stage.preset};
    this.env.setLearningStage(stage);
  }
  step(action){const result=this.env.step(action),previous=this.position;this.position=this._position();this.velocity=this.position.map((v,i)=>(v-previous[i])/this.dt);this.steps++;
    const offset=this.trackedId*9;const upright=this.initialUp.reduce((sum,value,i)=>sum+value*this.env.data.xmat[offset+6+i],0);
    const outcome=evaluateTask(this.task,{velocity:this.velocity,position:this.position,upright,height:this.position[2],initial_height:this.initial_height,
      target_speed:this.env.target_speed,energy:result.info.energy_cost,smoothness:result.info.smoothness_cost,elapsed:this.steps*this.dt,previous_distance:this.distance,dt:this.dt});
    if(result.info.nonfoot_contact){outcome.fallen=true;outcome.terminated=this.task.terminate_on_fall;outcome.reward-=2;outcome.components.nonfoot_contact=-2;}
    if(this.env.coordinated&&this.task.mechanical_work_weight!=null){
      const mass=this.env.model.body_mass.reduce((a,b)=>a+b,0);
      const cost=-this.task.mechanical_work_weight*result.info.motor_work_j/(mass*this.env.environment.gravity*Math.max(this.env.target_speed,.1)*this.dt);
      outcome.reward+=cost;outcome.components.mechanical_work=cost;
    }
    this.env._fallen=outcome.fallen;
    if(this.env.coordinated&&!outcome.fallen){Object.assign(outcome.components,this.env.walkingComponents);outcome.reward+=Object.values(this.env.walkingComponents).reduce((a,b)=>a+b,0);}
    this.distance=outcome.distance;
    return {obs:this._augment(result.obs),reward:outcome.reward,terminated:outcome.terminated,truncated:this.steps>=this.max_episode_steps,
      info:{...result.info,velocity:this.velocity[0],height:this.position[2],reward:outcome.reward,reward_components:outcome.components,task_distance:this.distance,fallen:outcome.fallen}};
  }
  getFrame(){return {...this.env.getFrame(),...(this.task.preset==="target"||this.task.preset==="custom"?{target:this.task.target}: {})};}
  close(){this.env.close();}
}
