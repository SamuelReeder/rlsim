/**
 * Explainable starting configurations, not a hyperparameter search or a
 * prediction that a body can walk. All numerical tiers below are RLsim design
 * choices; they have not been benchmarked as optimal.
 *
 * PPO clipping, minibatches and rollout sizes:
 * https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html
 * Evaluation across seeds and environment-specific tuning:
 * https://stable-baselines3.readthedocs.io/en/master/guide/rl_tips.html
 *
 * No imports, browser globals, mutation, network calls or training side effects.
 */

const count = value => Number.isInteger(value) && value >= 0 ? value : null;
const positive = value => Number.isFinite(value) && value > 0 ? value : null;
const clamp = (value, low, high) => Math.min(high, Math.max(low, value));
const array = value => Array.isArray(value) ? value : [];
const first = (...values) => values.find(value => value !== null && value !== undefined) ?? null;
const round = (value, digits = 2) => Number(value.toFixed(digits));
const validPosition = value => Array.isArray(value) && value.length === 3 && value.every(Number.isFinite);
const distance = (a, b) => Math.hypot(...a.map((value, i) => value - b[i]));

function legLength(body, rig) {
  if (body.kind !== "mjcf" && [2, 4, 6].includes(body.legs)
      && positive(body.upper_length) && positive(body.lower_length)) {
    return {length: body.upper_length + body.lower_length, source: "template leg lengths"};
  }
  // Only named, connected humanoid chains qualify. An arbitrary mesh's AABB
  // may contain hair, wings or machinery and is not a trustworthy leg length.
  if (rig?.profile !== "humanoid") return null;
  const bodies = array(rig.bodies), byId = new Map(bodies.map(part => [part.id, part]));
  const names = part => String(part?.label || "").toLowerCase().replace(/[^a-z]/g, "");
  const feet = bodies.filter(part => /^(?:left|right|l|r)?(?:foot|ankle)(?:left|right|l|r)?$/.test(names(part)));
  const legs = [];
  for (const foot of feet) {
    const shin = byId.get(foot.parent), thigh = byId.get(shin?.parent);
    if (!shin || !thigh || !/lowerleg|shin|calf/.test(names(shin)) || !/upperleg|upleg|thigh/.test(names(thigh))) continue;
    if (![thigh, shin, foot].every(part => validPosition(part.position))) continue;
    const upper = distance(thigh.position, shin.position), lower = distance(shin.position, foot.position);
    if (upper < .05 || lower < .05 || upper / lower < .3 || upper / lower > 3
        || thigh.position[2] <= foot.position[2] + .1) continue;
    legs.push(upper + lower);
  }
  if (legs.length !== 2 || Math.max(...legs) / Math.min(...legs) > 1.3) return null;
  const length = (legs[0] + legs[1]) / 2;
  return length >= .15 && length <= 5 ? {length, source: "both connected humanoid leg chains"} : null;
}

function dimensions(body, modelInfo, task) {
  const rig = body.physical_rig, parts = array(rig?.bodies);
  const template = body.kind !== "mjcf" && [2, 4, 6].includes(body.legs);
  const graphControls = parts.length ? parts.reduce((total, part) => total +
    (positive(part.joint?.torque) ? ({hinge: 1, ball: 3, free: rig.profile === "rigid" ? 6 : 0}[part.joint?.type] || 0) : 0), 0) : null;
  const actions = first(count(modelInfo.action_size), count(modelInfo.nu),
    Array.isArray(modelInfo.actuators) ? modelInfo.actuators.length : null,
    template ? body.legs * 2 : null, graphControls);
  // Built-in state: qpos except root x, qvel, speed, morphology and scales;
  // the optional task wrapper adds nine position/target/velocity values.
  const observations = first(count(modelInfo.observation_size),
    template ? 18 + body.legs * 4 + (task == null ? 0 : 9) : null);
  const physics = modelInfo.physics_summary || {};
  const profile = template ? "template" : rig?.profile || (body.kind === "mjcf" ? "imported" : "unknown");
  const bodyCount = first(count(physics.moving_bodies),
    Array.isArray(modelInfo.bodies) ? modelInfo.bodies.length : null,
    parts.length || null, template ? 1 + body.legs * 2 : null);
  const graphMass = parts.length && parts.every(part => positive(part.mass))
    ? parts.reduce((total, part) => total + part.mass, 0) : null;
  const floating = typeof physics.floating_base === "boolean" ? physics.floating_base
    : template ? false : parts.some(part => part.parent === null && part.joint?.type === "free")
      || array(modelInfo.joints).some(joint => joint.type === "free");
  return {
    profile, actions, observations, body_count: bodyCount,
    total_mass: first(positive(physics.total_mass), graphMass, template ? positive(body.body_mass) : null),
    body_height: positive(physics.body_height),
    tracked_height: positive(physics.tracked_height),
    floating_base: floating,
    articulated_joints: count(physics.articulated_joints),
    dimensions_source: count(modelInfo.action_size) !== null ? "compiled preview" : template ? "template" : graphControls !== null ? "physical rig" : "unavailable",
  };
}

/**
 * resources accepts browser hints {hardwareConcurrency, deviceMemory} (GB).
 * hardware.training_device must be the actual tested backend, not merely the
 * availability of navigator.gpu. Physics remains on CPU for every backend.
 */
export function recommendTraining({
  body = {}, modelInfo = {}, environment = {}, task = null,
  hardware = {}, engine = "browser", resources = {},
} = {}) {
  body ||= {}; modelInfo ||= {}; environment ||= {}; hardware ||= {}; resources ||= {};
  const analysis = dimensions(body, modelInfo, task);
  const settings = {}, reasons = [], warnings = [];
  const result = summary => ({version: 1, settings, reasons, warnings, summary, analysis});
  const set = (key, value, reason) => {
    settings[key] = value;
    reasons.push({key, value, reason});
  };
  const warn = message => { if (!warnings.includes(message)) warnings.push(message); };
  const preset = task?.preset || "velocity";
  const parts = array(body.physical_rig?.bodies);
  analysis.task = preset;
  analysis.recommendable = false;
  if (engine === "server" && body.kind === "mjcf") {
    warn("Imported bodies train in This device mode. Native server training currently supports the built-in body templates.");
    return result("Switch to This device mode to recommend settings for this imported body.");
  }
  if (parts.some(part => !positive(part.mass) || !part.joint || !validPosition(part.position))) {
    warn("The editable physical rig has invalid mass, joint or rest-position data. Repair the rig and preview it before training.");
    return result("Fix the physical rig before choosing training settings.");
  }
  if (analysis.actions === 0) {
    warn("This body has no active motor controls. Add actuators or enable joint torque before training; a larger network cannot move passive joints.");
    return result("No trainable motor controls were found.");
  }
  if (analysis.actions === null) {
    warn("Preview the body first so RLsim can inspect its actual motor controls and observations.");
    return result("A compiled body preview is needed before recommending settings.");
  }
  if (analysis.observations === 0) {
    warn("This body reports no policy observations. Check its compiled controller interface before training.");
    return result("A valid observation interface is needed before recommending settings.");
  }
  if (analysis.observations === null) warn("Observation dimensions are unavailable. This is a provisional setup; preview the body and recommend again before training.");
  const humanoid = analysis.profile === "humanoid";
  const biped = analysis.profile === "template" && body.legs === 2;
  const high = humanoid || analysis.actions >= 24 || (analysis.observations || 0) >= 128
    || (analysis.body_count || 0) >= 16 || (analysis.floating_base && analysis.actions >= 12);
  const medium = !high && (biped || analysis.actions > 8 || (analysis.observations || 0) > 64 || (analysis.body_count || 0) > 10);
  const veryLarge = analysis.actions >= 48 || (analysis.observations || 0) >= 192;
  const complexity = high ? "complex" : medium ? "moderate" : "simple";
  const width = high ? 256 : medium ? 128 : 64;
  const device = String(hardware.training_device || hardware.selected_device || "").toLowerCase();
  const gpu = ["webgpu", "cuda", "mps"].includes(device);
  const memory = positive(resources.deviceMemory ?? resources.device_memory);
  const cores = positive(resources.hardwareConcurrency ?? resources.hardware_concurrency);
  const limited = (memory !== null && memory <= 2) || (cores !== null && cores <= 2);
  const native = engine === "server";
  // These are environment copies collected together, not CPU worker threads.
  let envs = gpu ? high ? 4 : 8 : high ? 2 : 4;
  if (native && gpu) envs = 8;
  if (limited) envs = high ? 1 : 2;
  const rollout = preset === "target" ? 1024 : high || medium ? 512 : 256;
  const samples = envs * rollout;
  const batch = Math.min(samples, gpu && !limited ? 512 : high && !limited ? 256 : 128);
  let budget = veryLarge ? 1572864 : high ? 1048576 : medium ? 524288 : 262144;
  if (preset === "balance") budget /= 2;
  budget = Math.min(2000000, Math.max(131072, Math.ceil(budget / samples) * samples));
  // Round down only if the app's 2M cap prevented a whole rollout.
  budget = Math.floor(budget / samples) * samples;
  const stateDescription = `${analysis.actions} controls${analysis.observations !== null ? ` and ${analysis.observations} observations` : ""}`;
  set("hidden_size", width, `${stateDescription} and ${complexity} body coordination suggest ${width} units per layer as a starting capacity.`);
  set("hidden_layers", 2, "Two hidden layers keep the policy expressive without adding depth that has not been tested on this body.");
  set("total_steps", budget, `${complexity === "simple" ? "A simple" : complexity === "moderate" ? "A moderately complex" : "A complex"} body needs repeated rollout updates; this budgets ${budget / samples} updates${preset === "balance" ? " for the simpler Balance objective" : ""}. It is a trial budget, not a predicted time to success.`);
  set("n_envs", envs, `${envs} environment copies balance experience collection and ${limited ? "limited device resources" : gpu ? "GPU policy batches" : "CPU training cost"}. Physics still runs on CPU; these copies are not parallel physics workers.`);
  set("rollout_steps", rollout, `Collect ${rollout} steps per copy (${samples} transitions per update)${preset === "target" ? " to cover more of the route to a target" : high || medium ? " before updating the more coupled controller" : " before each policy update"}.`);
  set("batch_size", batch, `${batch}-sample minibatches divide the ${samples}-transition rollout evenly${gpu && !limited ? " and give the GPU larger training batches" : " while keeping update memory modest"}.`);
  set("learning_rate", high ? .00015 : .0003, high
    ? "Use a smaller learning rate for the coupled, high-dimensional controller to make initial updates more conservative."
    : "Start with PPO's conventional 0.0003 learning rate; body size alone does not identify a better value.");
  set("clip_range", high ? .15 : .2, high
    ? "A narrower PPO clip range encourages smaller policy changes; clipping alone does not guarantee stability."
    : "A 0.2 PPO clip range is a conventional starting point for limiting policy changes.");
  set("epochs", medium ? 6 : 4, high
    ? "Four passes over each rollout limit repeated updates on the same data; the optional KL safeguard can stop excessive updates earlier."
    : medium ? "Six passes reuse a moderate controller's rollout without the much longer updates of a high epoch count."
      : "Four passes keep initial updates and browser feedback responsive.");
  set("gamma", .99, "Keep a conventional 0.99 discount factor; anatomy alone does not determine the task's useful planning horizon.");
  set("entropy_coef", .005, "Keep the existing small exploration bonus. More motor controls do not by themselves justify adding more action noise.");
  set("randomize", false, "First establish behavior with fixed mass, motor strength and floor friction. Enable randomization in later trials to test and train robustness.");
  set("curriculum", ["velocity","walk"].includes(preset), ["velocity","walk"].includes(preset)
    ? "Use your selected locomotion curriculum: Gradual speed ramps the command with elapsed steps; Balance → walking waits for stable episodes."
    : "Locomotion curricula are intended for Velocity and coordinated walking; leave them off for Balance, Target and custom rewards.");
  const leg = legLength(body, body.physical_rig);
  if (["velocity","walk"].includes(preset) && leg) {
    const gravity = positive(environment.gravity) || 9.81;
    const speed = round(clamp(.25 * Math.sqrt(gravity * leg.length), .1, 1.2));
    set("target_speed", speed, `Start at ${speed} m/s using a cautious 0.25 × √(gravity × leg length) scale from ${leg.source} (${round(leg.length)} m). This engineering heuristic is not an optimal gait speed.`);
  }
  if (humanoid) warn("A free-standing humanoid may need more than this budget, and the app's 2,000,000-step limit may still be insufficient. Try a separate Balance run to check controllability; it does not transfer a policy into walking.");
  else if (high) warn("Complex bodies may need more samples than this trial, and even the app's 2,000,000-step limit may be insufficient. Evaluate progress rather than treating the budget as a success threshold.");
  else if (biped) warn("This biped must learn support and movement together. A separate Balance trial can help check the setup before attempting velocity tracking.");
  if (analysis.profile === "rigid") warn("Rigid conversion uses direct forces and torques on one body. It can learn movement, but articulated walking requires a skeleton with powered joints.");
  else if (analysis.floating_base && analysis.articulated_joints === 0) warn("This free-moving body has no articulated joints. Its controls may move the body directly, but articulated walking requires powered limb joints.");
  if (analysis.body_height !== null && (analysis.body_height < .05 || analysis.body_height > 10)) warn(`The compiled body is ${round(analysis.body_height, 3)} metres tall. Check the import units and intended scale before training; very small or large bodies may need different physical settings.`);
  if (preset === "custom") warn("A custom reward can change which settings work well. These recommendations only account for controller size and compute cost; inspect reward behavior separately.");
  if (positive(task?.episode_seconds) && task.episode_seconds < 5) warn("Episodes shorter than five seconds may end before a sustained gait or target-reaching behavior develops. Review the episode duration for your task.");
  if (Number.isFinite(environment.friction) && environment.friction < .25) warn("The floor has low friction. Foot slip can limit locomotion even with well-chosen training settings.");
  if (Number.isFinite(environment.gravity) && (environment.gravity < 3 || environment.gravity > 15)) warn("Gravity is far from Earth gravity; review motor strength and contact behavior before committing to a long run.");
  const masses = parts.map(part => part.mass).filter(positive);
  if (masses.length > 1 && Math.max(...masses) / Math.min(...masses) > 100) warn("The rig spans more than a 100× mass ratio between parts. Inspect the small parts and mass distribution if contacts or joints become unstable.");
  const powered = parts.filter(part => positive(part.joint?.torque) && ["ball", "hinge"].includes(part.joint?.type));
  if (powered.length && powered.every(part => part.joint.damping === 0)) warn("Every powered joint has zero damping. Review joint damping if the physical body oscillates; training parameters cannot repair unstable mechanics.");
  if (!device) warn("A tested training device was not reported, so environment and batch recommendations use conservative CPU assumptions.");
  if (!gpu && budget >= 524288) warn("This trial has a large sample budget for CPU training and may take substantial time. Review early learning progress before committing to the full run; no runtime or successful gait has been predicted.");
  if (limited && high) warn("This body is large for the reported device resources. The setup reduces environment copies, but training may still be slow.");
  if (!leg && preset === "velocity") warn("Leg length could not be established reliably, so the requested target speed is left unchanged.");
  analysis.recommendable = true;
  Object.assign(analysis, {
    complexity, training_device: device || "unknown", gpu, limited_resources: limited,
    transitions_per_update: samples, planned_updates: budget / samples,
    leg_length: leg?.length ?? null, leg_length_source: leg?.source ?? null,
  });
  return result(`Recommended starting setup for ${stateDescription}: ${width} × 2 network, ${budget.toLocaleString("en-US")} training steps. Validate results across several seeds; this is not an optimum or a guarantee of walking.`);
}
