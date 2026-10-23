import assert from "node:assert/strict";
import test from "node:test";
import {recommendTraining} from "../morpholab/static/training-advisor.mjs";
import {PRESETS} from "../morpholab/static/browser-physics.mjs";

const cpu = {training_device: "cpu"};
const gpu = {training_device: "webgpu"};
const velocity = {preset: "velocity", episode_seconds: 20};
const compiled = (actions, observations, extra = {}) => ({
  action_size: actions, observation_size: observations,
  physics_summary: {total_mass: 60, body_height: 1.7, tracked_height: .8,
    moving_bodies: 20, floating_base: true, articulated_joints: 19, ...extra},
});
const part = (id, label, parent, position, type = "ball", mass = 3) => ({
  id, label, parent, position, mass,
  joint: {type, torque: 50, damping: 1},
});
function female() {
  return {kind: "mjcf", physical_rig: {profile: "humanoid", bodies: [
    part("pelvis", "Body_1", null, [0, 0, .85], "free", 15),
    ...[-1, 1].flatMap((side, i) => {
      const suffix = i ? "R" : "L";
      return [
        part(`thigh${suffix}`, `UpperLeg${suffix}`, "pelvis", [0, .1 * side, .8]),
        part(`shin${suffix}`, `LowerLeg${suffix}`, `thigh${suffix}`, [0, .1 * side, .4], "hinge"),
        part(`foot${suffix}`, `Foot${suffix}`, `shin${suffix}`, [0, .1 * side, .05]),
      ];
    }),
  ]}};
}
const bounds = {
  total_steps: [4096, 2000000, true], n_envs: [1, 32, true],
  hidden_size: [16, 256, true], hidden_layers: [1, 4, true],
  learning_rate: [.00001, .01], batch_size: [16, 2048, true],
  epochs: [1, 20, true], rollout_steps: [16, 1024, true],
  gamma: [.8, .9999], entropy_coef: [0, .1], clip_range: [.05, .4],
  target_speed: [.1, 3],
};
function valid(result) {
  for (const [key, value] of Object.entries(result.settings)) {
    if (["randomize", "curriculum"].includes(key)) { assert.equal(typeof value, "boolean"); continue; }
    assert.ok(bounds[key], `Unsupported training key: ${key}`);
    const [lo, hi, integer] = bounds[key];
    assert.ok(Number.isFinite(value) && value >= lo && value <= hi, `${key}=${value}`);
    if (integer) assert.ok(Number.isInteger(value), `${key} should be an integer`);
    assert.ok(result.reasons.some(reason => reason.key === key && reason.value === value && reason.reason.length > 20));
  }
  if (!result.analysis.recommendable) return;
  const config = result.settings, sampleCount = config.n_envs * config.rollout_steps;
  assert.ok(config.batch_size <= sampleCount);
  assert.equal(sampleCount % config.batch_size, 0);
  assert.equal(config.total_steps % sampleCount, 0);
  assert.ok(config.total_steps / sampleCount >= 32);
}

test("simple templates get bounded configurations with trustworthy dimension and speed estimates", () => {
  const result = recommendTraining({body: PRESETS.quadruped, task: velocity, hardware: cpu});
  valid(result);
  assert.equal(result.analysis.actions, 8);
  assert.equal(result.analysis.observations, 43);
  assert.equal(result.analysis.profile, "template");
  assert.equal(result.analysis.total_mass, 4);
  assert.equal(result.settings.hidden_size, 64);
  assert.equal(result.settings.total_steps, 262144);
  assert.equal(result.settings.target_speed, .72);
  assert.equal(result.settings.randomize, false);
  assert.equal(result.settings.curriculum, true);
  assert.equal(result.settings.hidden_layers, 2);
});

test("a complex female humanoid gets greater capacity and a longer, conservative trial", () => {
  const result = recommendTraining({body: female(), modelInfo: compiled(50, 189), task: velocity, hardware: gpu});
  valid(result);
  assert.equal(result.settings.hidden_size, 256);
  assert.equal(result.settings.total_steps, 1572864);
  assert.equal(result.settings.learning_rate, .00015);
  assert.equal(result.settings.clip_range, .15);
  assert.equal(result.settings.epochs, 4);
  assert.equal(result.analysis.dimensions_source, "compiled preview");
  assert.ok(result.settings.target_speed < 1);
  assert.ok(result.warnings.some(warning => /Balance.*does not transfer/.test(warning)));
});

test("generic robots use actual controls and dimensions, never infer leg length from an AABB", () => {
  const result = recommendTraining({body: {kind: "mjcf"}, modelInfo: compiled(12, 79),
    task: velocity, hardware: cpu});
  valid(result);
  assert.equal(result.analysis.actions, 12);
  assert.equal(result.analysis.complexity, "complex");
  assert.ok(!Object.hasOwn(result.settings, "target_speed"));
  assert.ok(result.warnings.some(warning => /target speed is left unchanged/.test(warning)));
});

test("known zero actuators and passive generated graphs do not suggest a trainable setup", () => {
  const noMotor = recommendTraining({body: {kind: "mjcf"}, modelInfo: compiled(0, 25), hardware: cpu});
  assert.deepEqual(noMotor.settings, {});
  assert.equal(noMotor.analysis.recommendable, false);
  assert.ok(noMotor.warnings.some(warning => /no active motor/.test(warning)));
  const body = female();
  for (const item of body.physical_rig.bodies) item.joint.torque = 0;
  assert.deepEqual(recommendTraining({body}).settings, {});
  const noObservations = recommendTraining({body: {kind: "mjcf"}, modelInfo: compiled(6, 0)});
  assert.deepEqual(noObservations.settings, {});
  assert.match(noObservations.warnings[0], /no policy observations/);
});

test("speed suggestions scale with leg dimensions and gravity but require two credible humanoid chains", () => {
  const body = {...PRESETS.quadruped, upper_length: .8, lower_length: .8};
  const normal = recommendTraining({body, task: velocity, environment: {gravity: 9.81}, hardware: cpu});
  const lighter = recommendTraining({body, task: velocity, environment: {gravity: 2}, hardware: cpu});
  assert.ok(normal.settings.target_speed > lighter.settings.target_speed);
  const broken = female();
  broken.physical_rig.bodies.find(part => part.id === "footL").parent = "pelvis";
  const result = recommendTraining({body: broken, modelInfo: compiled(50, 189), task: velocity, hardware: cpu});
  assert.ok(!Object.hasOwn(result.settings, "target_speed"));
});

test("task, reward, seed, body and environment are never changed and recommendations are deterministic", () => {
  const input = {body: female(), modelInfo: compiled(50, 189), hardware: cpu,
    environment: {gravity: 4, friction: .8}, task: {preset: "custom", expression: "energy", episode_seconds: 20}};
  const before = structuredClone(input), first = recommendTraining(input), second = recommendTraining(input);
  valid(first);
  assert.deepEqual(input, before);
  assert.deepEqual(first, second);
  for (const key of ["body", "task", "seed", "environment", "device", "target_speed", "expression", "weights"]) {
    assert.ok(!Object.hasOwn(first.settings, key), key);
  }
  assert.equal(first.settings.curriculum, false);
  assert.ok(first.warnings.some(warning => /custom reward/i.test(warning)));
});

test("Balance and Target retain their goal while the actual speed curriculum is disabled", () => {
  for (const preset of ["balance", "target"]) {
    const result = recommendTraining({body: PRESETS.biped, task: {preset}, hardware: gpu});
    valid(result);
    assert.equal(result.settings.curriculum, false);
    assert.ok(!Object.hasOwn(result.settings, "target_speed"));
    assert.equal(result.settings.rollout_steps, preset === "target" ? 1024 : 512);
  }
});

test("the verified device affects batches, not network requirements or sample budget", () => {
  const config = {body: female(), modelInfo: compiled(50, 189), task: velocity};
  const onCPU = recommendTraining({...config, hardware: cpu});
  const onGPU = recommendTraining({...config, hardware: gpu});
  valid(onCPU); valid(onGPU);
  assert.equal(onCPU.settings.hidden_size, onGPU.settings.hidden_size);
  assert.equal(onCPU.settings.total_steps, onGPU.settings.total_steps);
  assert.ok(onCPU.settings.n_envs < onGPU.settings.n_envs);
  assert.ok(onCPU.settings.batch_size < onGPU.settings.batch_size);
  const limited = recommendTraining({...config, hardware: gpu, resources: {hardwareConcurrency: 2, deviceMemory: 2}});
  valid(limited);
  assert.equal(limited.settings.n_envs, 1);
  assert.equal(limited.settings.batch_size, 128);
});

test("missing metadata is conservative, and device availability is not a tested backend", () => {
  const unavailable = recommendTraining({body: {kind: "mjcf"}, hardware: {cuda_available: true}});
  assert.deepEqual(unavailable.settings, {});
  assert.match(unavailable.summary, /compiled body preview/);
  const inferred = recommendTraining({body: female(), task: velocity, hardware: {cuda_available: true}});
  valid(inferred);
  assert.equal(inferred.analysis.gpu, false);
  assert.equal(inferred.analysis.training_device, "unknown");
  assert.ok(inferred.warnings.some(warning => /provisional setup/.test(warning)));
  assert.ok(inferred.warnings.some(warning => /conservative CPU/.test(warning)));
  assert.deepEqual(recommendTraining().settings, {});
});

test("native template recommendations obey shared limits and imported bodies explain the engine mismatch", () => {
  const native = recommendTraining({body: PRESETS.hexapod, hardware: {training_device: "cuda"}, engine: "server", task: velocity});
  valid(native);
  assert.equal(native.settings.n_envs, 8);
  const unsupported = recommendTraining({body: female(), modelInfo: compiled(50, 189), hardware: cpu, engine: "server"});
  assert.deepEqual(unsupported.settings, {});
  assert.match(unsupported.warnings[0], /Imported bodies train in This device mode/);
});

test("rigid movement, short episodes, low friction and invalid physical data have distinct guidance", () => {
  const body = {kind: "mjcf", physical_rig: {profile: "rigid",
    bodies: [part("root", "Body", null, [0, 0, .3], "free")]}};
  const result = recommendTraining({body, task: {preset: "velocity", episode_seconds: 2},
    environment: {friction: .1}, hardware: cpu});
  valid(result);
  assert.equal(result.analysis.actions, 6);
  assert.ok(result.warnings.some(warning => /direct forces and torques/.test(warning)));
  assert.ok(result.warnings.some(warning => /shorter than five seconds/.test(warning)));
  assert.ok(result.warnings.some(warning => /low friction/.test(warning)));
  body.physical_rig.bodies[0].mass = -1;
  assert.deepEqual(recommendTraining({body, modelInfo: compiled(6, 30)}).settings, {});
});

test("authored free bodies with no limb joints and unusual scales receive review guidance", () => {
  const result = recommendTraining({body: {kind: "mjcf"}, task: velocity, hardware: cpu,
    modelInfo: compiled(6, 30, {articulated_joints: 0, body_height: 30, moving_bodies: 1})});
  valid(result);
  assert.ok(result.warnings.some(warning => /no articulated joints/.test(warning)));
  assert.ok(result.warnings.some(warning => /30 metres tall.*import units/.test(warning)));
  assert.ok(!Object.hasOwn(result.settings, "target_speed"));
  const small = recommendTraining({body: {kind: "mjcf"}, task: velocity, hardware: cpu,
    modelInfo: compiled(6, 30, {body_height: .02})});
  assert.ok(small.warnings.some(warning => /0.02 metres tall/.test(warning)));
});

test("all body, device and task combinations fit shared browser/native PPO constraints", () => {
  const bodies = [PRESETS.quadruped, PRESETS.biped, PRESETS.hexapod, female()];
  for (const body of bodies) for (const preset of ["velocity", "balance", "target", "custom"])
    for (const hardware of [cpu, gpu, {}]) for (const resources of [{}, {deviceMemory: 1}]) {
      valid(recommendTraining({body, modelInfo: body.kind === "mjcf" ? compiled(50, 189) : {},
        task: {preset}, hardware, resources}));
    }
});
