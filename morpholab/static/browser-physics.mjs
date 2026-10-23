import {EffortMeter} from "./effort.mjs";
import {MotionTracker,phaseTargets,positionControls,walkingComponents} from "./motion.mjs";
/**
 * Local MuJoCo locomotion environment, matching environment.py.
 * Physics runs in WebAssembly on the visitor's CPU. Training chooses its own
 * accelerator; this module has no Python server or network API dependency.
 */

export const PHYSICS_TIMESTEP = 0.004;
export const FRAME_SKIP = 5;
export const EPISODE_STEPS = 1000;

const BODY_DEFAULTS = {
  name: "Quadruped", legs: 4, torso_length: 0.8, torso_radius: 0.14,
  upper_length: 0.42, lower_length: 0.42, leg_radius: 0.055,
  body_mass: 4, motor_strength: 60, joint_damping: 1.5,
};
const BODY_LIMITS = {
  legs: [2, 6], torso_length: [0.3, 1.8], torso_radius: [0.07, 0.3],
  upper_length: [0.18, 0.9], lower_length: [0.18, 0.9],
  leg_radius: [0.025, 0.1], body_mass: [1, 20],
  motor_strength: [5, 150], joint_damping: [0.1, 5],
};
const ENV_DEFAULTS = {gravity: 9.81, friction: 1};
const ENV_LIMITS = {gravity: [0.5, 20], friction: [0.05, 3]};

export const PRESETS = {
  quadruped: {...BODY_DEFAULTS},
  biped: {
    ...BODY_DEFAULTS, name: "Biped", legs: 2, torso_length: 0.45,
    torso_radius: 0.16, upper_length: 0.46, lower_length: 0.46,
    body_mass: 3.5, motor_strength: 45,
  },
  hexapod: {
    ...BODY_DEFAULTS, name: "Hexapod", legs: 6, torso_length: 1.15,
    torso_radius: 0.16, upper_length: 0.34, lower_length: 0.36,
    body_mass: 6, motor_strength: 45,
  },
};

let physicsPromise;
export async function loadPhysics() {
  if (!physicsPromise) {
    physicsPromise = import("./vendor/mujoco/mujoco.js")
      .then(({default: loadMujoco}) => loadMujoco())
      .catch(error => {
        physicsPromise = undefined;
        throw error;
      });
  }
  return physicsPromise;
}

function validatedObject(input, defaults, limits, label) {
  if (input == null) input = {};
  if (typeof input !== "object" || Array.isArray(input)) {
    throw new Error(`${label} must be an object`);
  }
  for (const key of Object.keys(input)) {
    if (!Object.hasOwn(defaults, key)) throw new Error(`Unknown ${label} setting: ${key}`);
  }
  const result = {...defaults, ...input};
  for (const [key, [low, high]] of Object.entries(limits)) {
    if (!Number.isFinite(result[key]) || result[key] < low || result[key] > high) {
      throw new Error(`${key} must be a finite number between ${low} and ${high}`);
    }
  }
  return result;
}

export function validateBody(body = {}) {
  const result = validatedObject(body, BODY_DEFAULTS, BODY_LIMITS, "body");
  if (![2, 4, 6].includes(result.legs)) throw new Error("legs must be 2, 4, or 6");
  if (typeof result.name !== "string" || [...result.name].length < 1 || [...result.name].length > 60) {
    throw new Error("Body name must contain 1 to 60 characters");
  }
  if (/[\u0000-\u0008\u000b\u000c\u000e-\u001f\ufffe\uffff]/u.test(result.name)) {
    throw new Error("Body name contains an invalid XML character");
  }
  return result;
}

export function validateEnvironment(environment = {}) {
  return validatedObject(environment, ENV_DEFAULTS, ENV_LIMITS, "environment");
}

function escapeXML(value) {
  return value.replaceAll("&", "&amp;").replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;").replaceAll('"', "&quot;").replaceAll("'", "&apos;");
}

export function exportMJCF(body = {}, environment = {}) {
  body = validateBody(body);
  environment = validateEnvironment(environment);
  const footRadius = body.leg_radius * 1.3;
  const standingHeight = (body.upper_length + body.lower_length) * Math.cos(0.1) + footRadius + 0.025;
  const pairs = body.legs / 2;
  const legMass = body.body_mass * 0.4 / body.legs;
  const legs = [], motors = [], joints = [];
  for (let pair = 0; pair < pairs; pair++) {
    const hipX = pairs === 1 ? 0 : -0.38 * body.torso_length + pair * 0.76 * body.torso_length / (pairs - 1);
    for (let sideIndex = 0; sideIndex < 2; sideIndex++) {
      const index = pair * 2 + sideIndex;
      const side = sideIndex === 0 ? -1 : 1;
      const hipY = side * (body.torso_radius + body.leg_radius * 0.5);
      const color = side > 0 ? "0.31 0.83 0.78 1" : "0.23 0.54 0.62 1";
      legs.push(`
        <body name="upper_${index}" pos="${hipX.toFixed(8)} ${hipY.toFixed(8)} 0">
          <joint name="hip_${index}" type="hinge" axis="0 1 0"
                 range="-1.3 1.3" damping="${body.joint_damping}" />
          <geom name="upper_geom_${index}" type="capsule"
                fromto="0 0 0 0 0 ${-body.upper_length}"
                size="${body.leg_radius}" mass="${legMass * 0.55}" rgba="${color}" />
          <body name="lower_${index}" pos="0 0 ${-body.upper_length}">
            <joint name="knee_${index}" type="hinge" axis="0 1 0"
                   range="-2.1 0.45" damping="${body.joint_damping}" />
            <geom name="lower_geom_${index}" type="capsule"
                  fromto="0 0 0 0 0 ${-body.lower_length}"
                  size="${body.leg_radius * 0.85}" mass="${legMass * 0.4}" rgba="${color}" />
            <geom name="foot_${index}" type="sphere" pos="0 0 ${-body.lower_length}"
                  size="${footRadius}" mass="${legMass * 0.05}" rgba="0.85 0.95 0.90 1" />
          </body>
        </body>`);
      for (const joint of [`hip_${index}`, `knee_${index}`]) {
        motors.push(`<motor name="${joint}_motor" joint="${joint}" gear="${body.motor_strength}"
                            ctrllimited="true" ctrlrange="-1 1" />`);
      }
      joints.push(0.1, -0.2);
    }
  }
  return `<mujoco model="${escapeXML(body.name)}">
    <compiler angle="radian" autolimits="true" />
    <option timestep="${PHYSICS_TIMESTEP}" gravity="0 0 ${-environment.gravity}"
            integrator="implicitfast" iterations="50" />
    <default>
      <joint armature="0.025" limited="true" />
      <geom contype="1" conaffinity="0" friction="1.0 0.005 0.0001"
            condim="3" solref="0.012 1" solimp="0.9 0.95 0.001" />
    </default>
    <worldbody>
      <light pos="0 -3 5" dir="0 0 -1" />
      <geom name="floor" type="plane" size="200 5 0.1" contype="0" conaffinity="1"
            priority="1" friction="${environment.friction} 0.005 0.0001" rgba="0.12 0.15 0.18 1" />
      <body name="torso" pos="0 0 ${standingHeight}">
        <joint name="root_x" type="slide" axis="1 0 0" limited="false" damping="0" armature="0" />
        <joint name="root_z" type="slide" axis="0 0 1" limited="false" damping="0" armature="0" />
        <joint name="root_pitch" type="hinge" axis="0 1 0" limited="false" damping="0" armature="0" />
        <geom name="torso_geom" type="capsule"
              fromto="${-body.torso_length / 2} 0 0 ${body.torso_length / 2} 0 0"
              size="${body.torso_radius}" mass="${body.body_mass * 0.6}" rgba="0.88 0.93 0.83 1" />
        ${legs.join("")}
      </body>
    </worldbody>
    <actuator>${motors.join("")}</actuator>
    <keyframe><key name="standing" qpos="${[0, 0, 0, ...joints].join(" ")}" /></keyframe>
  </mujoco>`;
}

const clip = (value, low, high) => Math.min(high, Math.max(low, value));
function seededRandom(seed) {
  if (!Number.isSafeInteger(seed) || seed < 0 || seed > 0xffffffff) {
    throw new Error("seed must be an integer between 0 and 4294967295");
  }
  let state = seed >>> 0;
  // Mulberry32: repeatable browser episodes, independent of Math.random().
  const next = () => {
    state = (state + 0x6d2b79f5) >>> 0;
    let t = state;
    t = Math.imul(t ^ t >>> 15, t | 1);
    t ^= t + Math.imul(t ^ t >>> 7, t | 61);
    return ((t ^ t >>> 14) >>> 0) / 4294967296;
  };
  next.getState=()=>state;
  next.setState=s=>{if(!Number.isInteger(s)||s<0||s>0xffffffff)throw new Error("Invalid environment random state.");state=s>>>0;};
  return next;
}

export class BrowserWalker {
  constructor(module, body = {}, environment = {}, options = {}) {
    this.mj = module;
    this.body = validateBody(body);
    this.environment = validateEnvironment(environment);
    const speed = options.target_speed ?? 1;
    if (!Number.isFinite(speed) || speed < 0.1 || speed > 5) {
      throw new Error("target_speed must be finite and between 0.1 and 5 m/s");
    }
    this.requested_target_speed = speed;
    this.target_speed = speed;
    this.training_progress = 1;
    this.randomize = Boolean(options.randomize);
    this.random = seededRandom(options.seed ?? 7);
    this.dt = PHYSICS_TIMESTEP * FRAME_SKIP;
    this.max_episode_steps = EPISODE_STEPS;
    this._closed = false;
    this._morphology = Object.entries(BODY_LIMITS).map(([key, [low, high]]) => 2 * (this.body[key] - low) / (high - low) - 1);
    try {
      this.model = module.MjModel.from_xml_string(exportMJCF(this.body, this.environment));
      this.data = new module.MjData(this.model);
      this._torsoId = module.mj_name2id(this.model, module.mjtObj.mjOBJ_BODY.value, "torso");
      this._baseMass = this.model.body_mass.slice();
      this._baseInertia = this.model.body_inertia.slice();
      this._baseFriction = this.model.geom_friction.slice();
      this._baseGear = this.model.actuator_gear.slice();
      this._domainScales = new Float64Array([1, 1, 1]);
      this._previousAction = new Float64Array(this.model.nu);
      this._action = new Float64Array(this.model.nu);
      this._observationSize = this.model.nq - 1 + this.model.nv + 1 + this._morphology.length + 3;
      this.coordinated=options.task?.preset==='walk';
      this.footDriftWeight=options.task?.foot_drift_weight??.2;
      this.footOnlySupport=options.task?.foot_only_support??false;
      this.effort=new EffortMeter();
      this.motion=new MotionTracker(this.body);this.phaseOffset=0;
      this._feetIds=Array.from({length:this.body.legs},(_,i)=>module.mj_name2id(this.model,module.mjtObj.mjOBJ_GEOM.value,`foot_${i}`));
      this._footIdSet=new Set(this._feetIds);
      if(this.coordinated)this._observationSize+=2+this.body.legs+this.model.nu;
      this._actionSize = this.model.nu;
      this.reset();
    } catch (error) {
      this.close();
      throw error;
    }
  }

  get observationSize() { return this._observationSize; }
  get actionSize() { return this._actionSize; }

  _assertOpen() {
    if (this._closed) throw new Error("Physics environment is closed");
  }

  setTrainingProgress(progress) {
    this._assertOpen();
    if (!Number.isFinite(progress)) throw new Error("training progress must be finite");
    this.training_progress = clip(progress, 0, 1);
    this.target_speed = this.requested_target_speed * (0.25 + 0.75 * this.training_progress);
    return this._observation();
  }
  setLearningStage(stage) {
    this.target_speed=this.requested_target_speed*stage.speed_fraction;
    this.randomization_strength=stage.randomization;
  }

  _observation() {
    const obs = new Float32Array(this.observationSize);
    const qpos = this.data.qpos, qvel = this.data.qvel;
    obs.set(qpos.subarray(1));
    obs.set(qvel, qpos.length - 1);
    let offset = qpos.length - 1 + qvel.length;
    obs[offset++] = this.target_speed;
    obs.set(this._morphology, offset);
    offset += this._morphology.length;
    obs[offset++] = (this._domainScales[0] - 1) / 0.2;
    obs[offset++] = (this._domainScales[1] - 1) / 0.2;
    obs[offset++] = (this._domainScales[2] - 1) / 0.4;
    if(this.coordinated){const {phase}=phaseTargets(this.data.time,this.phaseOffset,this.body.legs,this.motion.length);obs.set([Math.sin(2*Math.PI*phase),Math.cos(2*Math.PI*phase),...this.motion.contacts.map(Number),...this._previousAction],offset);}
    return obs;
  }

  _randomizeModel() {
    this._domainScales.set(this.randomize
      ? [.2,.2,.4].map(range=>1+(2*this.random()-1)*range*(this.randomization_strength??1))
      : [1, 1, 1]);
    const [mass, motor, friction] = this._domainScales;
    for (const [view, base, scale] of [
      [this.model.body_mass, this._baseMass, mass],
      [this.model.body_inertia, this._baseInertia, mass],
      [this.model.actuator_gear, this._baseGear, motor],
      [this.model.geom_friction, this._baseFriction, friction],
    ]) {
      for (let i = 0; i < base.length; i++) view[i] = base[i] * scale;
    }
    this.mj.mj_setConst(this.model, this.data);
  }

  reset(seed) {
    this._assertOpen();
    if (seed !== undefined && seed !== null) this.random = seededRandom(seed);
    this._randomizeModel();
    this.mj.mj_resetDataKeyframe(this.model, this.data, 0);
    const qpos = this.data.qpos, qvel = this.data.qvel;
    for (let i = 2; i < qpos.length; i++) qpos[i] += -0.015 + this.random() * 0.03;
    for (let i = 0; i < qvel.length; i++) qvel[i] = -0.015 + this.random() * 0.03;
    this.mj.mj_forward(this.model, this.data);
    this.phaseOffset=this.coordinated?this.random():0;
    this.motion.reset(this._feet());
    this.effort.reset();
    this._previousAction.fill(0);
    this._steps = 0;
    this._fallen = false;
    this.last_info = this._stateInfo(this.data.qvel[0]);
    return this._observation();
  }

  _feet(){return this._feetIds.map(id=>[this.data.geom_xpos[id*3],this.data.geom_xpos[id*3+1],this.data.geom_xpos[id*3+2]-this.body.leg_radius*1.3]);}
  _stateInfo(velocity) {
    return {
      position: this.data.qpos[0], velocity, target_speed: this.target_speed,
      height: this.data.xpos[this._torsoId * 3 + 2], pitch: this.data.qpos[2],
      fallen: this._fallen, step: this._steps,
      mass_scale: this._domainScales[0], motor_scale: this._domainScales[1],
      friction_scale: this._domainScales[2],
    };
  }

  step(action) {
    this._assertOpen();
    if (action == null || action.length !== this.actionSize) {
      throw new Error(`Expected ${this.actionSize} actions`);
    }
    for (let i = 0; i < this.actionSize; i++) {
      if (!Number.isFinite(action[i])) throw new Error("Actions must contain only finite numbers");
      this._action[i] = clip(action[i], -1, 1);
    }
    const xBefore = this.data.qpos[0];
    this.data.ctrl.set(this._action);
    let stepWork=0,nonfootContact=false;
    for (let i = 0; i < FRAME_SKIP; i++) {
      if(this.coordinated)this.data.ctrl.set(positionControls(this._action,this.data.qpos,this.data.qvel,this.body));
      this.mj.mj_step(this.model, this.data);
      stepWork+=this.effort.observe(this.data.actuator_force,this.data.actuator_velocity,this.data.qfrc_actuator.subarray(3),this.data.ctrl,PHYSICS_TIMESTEP);
      if(this.footOnlySupport){
        const contacts=this.data.contact;
        try{for(let j=0;j<contacts.size();j++){
          const c=contacts.get(j);
          try{if(c.efc_address>=0&&c.dist<=0&&(c.geom1===0||c.geom2===0)&&!this._footIdSet.has(c.geom1===0?c.geom2:c.geom1))nonfootContact=true;}
          finally{c.delete();}
        }}finally{contacts.delete();}
      }
    }
    this.mj.mj_forward(this.model, this.data);
    this._steps++;
    const velocity = (this.data.qpos[0] - xBefore) / this.dt;
    const height = this.data.xpos[this._torsoId * 3 + 2];
    const upright = Math.cos(this.data.qpos[2]);
    const finite = this.data.qpos.every(Number.isFinite) && this.data.qvel.every(Number.isFinite);
    this._fallen = !finite || height < Math.max(this.body.torso_radius * 1.4,
      (this.body.upper_length + this.body.lower_length) * 0.35) || upright < 0.45;
    const speedError = (velocity - this.target_speed) / Math.max(this.target_speed, 0.5);
    const speedTracking = finite ? Math.exp(-2 * speedError ** 2) : 0;
    const progress = finite ? clip(velocity / Math.max(this.target_speed, 0.5), -1, 1) : 0;
    let energy = 0, smoothness = 0;
    for (let i = 0; i < this.actionSize; i++) {
      energy += this._action[i] ** 2;
      smoothness += (this._action[i] - this._previousAction[i]) ** 2;
    }
    energy /= this.actionSize;
    smoothness /= this.actionSize;
    const feet=this._feet(),slip=this.motion.update(feet,this.dt,this.data.qpos[2],smoothness);
    this.walkingComponents=this.coordinated?walkingComponents(this.body,feet,this.motion.contacts,this.data.time,this.phaseOffset,velocity,this.target_speed,height,this.data.qpos[2],this.data.qvel[1],this.data.qvel[2],slip,smoothness,this.footDriftWeight):null;
    const components = {
      speed_tracking: 1.5 * speedTracking, forward_progress: 0.3 * progress,
      alive: this._fallen ? 0 : 0.15,
      upright: this._fallen ? 0 : 0.15 * Math.max(upright, 0),
      energy: -0.04 * energy, smoothness: -0.02 * smoothness,
      fall: this._fallen ? -1 : 0,
    };
    const reward = Object.values(components).reduce((a, b) => a + b, 0);
    this._previousAction.set(this._action);
    this.last_info = {
      ...this._stateInfo(velocity), reward_components: components,
      speed_error: velocity - this.target_speed, energy_cost: energy,
      smoothness_cost: smoothness, reward, motor_work_j:stepWork, nonfoot_contact:nonfootContact,
    };
    return {
      obs: this._observation(), reward, terminated: this._fallen,
      truncated: this._steps >= this.max_episode_steps, info: {...this.last_info},
    };
  }

  getFrame() {
    this._assertOpen();
    const types = this.mj.mjtGeom;
    const names = new Map([
      [types.mjGEOM_CAPSULE.value, "capsule"], [types.mjGEOM_SPHERE.value, "sphere"],
      [types.mjGEOM_BOX.value, "box"], [types.mjGEOM_PLANE.value, "plane"],
    ]);
    const geomTypes = this.model.geom_type, positions = this.data.geom_xpos;
    const matrices = this.data.geom_xmat, sizes = this.model.geom_size, colors = this.model.geom_rgba;
    const geoms = [];
    for (let i = 0; i < this.model.ngeom; i++) {
      const type = names.get(geomTypes[i]);
      if (!type) continue;
      geoms.push({
        type, pos: Array.from(positions.subarray(i * 3, i * 3 + 3)),
        mat: Array.from(matrices.subarray(i * 9, i * 9 + 9)),
        size: Array.from(sizes.subarray(i * 3, i * 3 + 3)),
        rgba: Array.from(colors.subarray(i * 4, i * 4 + 4)),
      });
    }
    return {
      time: this.data.time, position: this.data.qpos[0], velocity: this.data.qvel[0],
      torso_position: Array.from(this.data.xpos.subarray(this._torsoId * 3, this._torsoId * 3 + 3)),
      target_speed: this.target_speed, fallen: this._fallen, geoms,
    };
  }

  close() {
    if (this._closed) return;
    this._closed = true;
    if (this.data) this.data.delete();
    if (this.model) this.model.delete();
    this.data = null;
    this.model = null;
  }
}
