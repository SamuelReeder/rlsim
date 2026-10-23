import assert from "node:assert/strict";
import {readFile} from "node:fs/promises";
import vm from "node:vm";
import * as tf from "@tensorflow/tfjs";
import {PPO, RunningNormalizer, selectBackend} from "../morpholab/static/browser-policy.mjs";

globalThis.tf = tf;
const initialTensors = tf.memory().numTensors;
const backend = await selectBackend();
assert.equal(backend.training_device, "cpu");
assert.equal(backend.execution_location, "browser");
assert.match(backend.fallback_reason, /WebGPU/);
assert.equal(tf.memory().numTensors, initialTensors, "Backend probe releases its tensors.");

const normalizer = new RunningNormalizer(2);
normalizer.update([[1, 10], [3, 14]]);
const normalized = normalizer.normalize([[1, 10], [3, 14]]);
assert.ok(Math.abs(normalized[0][0] + 1) < 0.001);
assert.ok(Math.abs(normalized[1][1] - 1) < 0.01);
assert.deepEqual(RunningNormalizer.restore(normalizer.serialize()).normalize([1, 10]), normalized[0]);
assert.throws(() => normalizer.update([[NaN, 1]]), /finite/);

// One-step continuous-control bandit: a real PPO update should learn to choose
// +0.8, using sampled actions and rewards rather than supervised target actions.
const policy = new PPO(2, 1, {seed: 42, hidden_size: 16, learning_rate: 0.003, epochs: 4, batch_size: 64, entropy_coef: 0});
const observations = Array.from({length: 128}, () => [1, 0.25]);
const before = await policy.act([[1, 0.25]], true);
const beforeSerialized = await policy.serialize();
for (let step = 0; step < 12; step++) {
  const rollout = await policy.act(observations);
  const rewards = rollout.actions.map(action => -((action[0] - 0.8) ** 2));
  const result = await policy.update({
    observations,
    actions: rollout.actions,
    oldLogProbs: rollout.logProbs,
    advantages: rewards.map((reward, i) => reward - rollout.values[i]),
    returns: rewards,
  });
  assert.ok(Object.values(result).every(Number.isFinite));
}
const after = await policy.act([[1, 0.25]], true);
assert.ok(Math.abs(after.actions[0][0] - 0.8) < Math.abs(before.actions[0][0] - 0.8) * 0.4,
  `PPO should improve the reward: mean ${before.actions[0][0]} -> ${after.actions[0][0]}.`);
const serialized = await policy.serialize();
assert.notDeepEqual(serialized.weights, beforeSerialized.weights);
const restored = await PPO.restore(JSON.parse(JSON.stringify(serialized)));
assert.deepEqual(await restored.act([[1, 0.25]], true), after, "Export and import preserve deterministic inference.");
const predictionTensors=tf.memory().numTensors,predictionRandomState=restored.randomState;
for(let i=0;i<50;i++)assert.deepEqual(await restored.predict([[1,.25]]),after.actions,
  "Actor-only playback matches the full deterministic policy.");
assert.equal(restored.randomState,predictionRandomState,"Playback must not advance stochastic training state.");
assert.equal(tf.memory().numTensors,predictionTensors,"Repeated actor-only playback releases temporary tensors.");
await assert.rejects(restored.predict([[NaN,0]]),/finite/);
await assert.rejects(restored.predict([]),/nonempty/);
assert.deepEqual(await restored.act([[1, 0.25]]), await policy.act([[1, 0.25]]), "Export preserves sampling state.");
const legacy = structuredClone(serialized);
delete legacy.config.hidden_layers;
const legacyRestored = await PPO.restore(legacy);
assert.equal(legacyRestored.config.hidden_layers, 2);
assert.deepEqual(await legacyRestored.act([[1, 0.25]], true), after, "Legacy two-layer checkpoints retain their predictions.");
legacyRestored.dispose();

const corrupt = structuredClone(serialized);
corrupt.weights.actor_w1.values[0] = Infinity;
await assert.rejects(PPO.restore(corrupt), /finite/);
restored.dispose();
policy.dispose();

const architectures = [];
for (const layers of [1, 3, 4]) {
  const custom = new PPO(2, 1, {seed: 42, hidden_size: 16, hidden_layers: layers,
    learning_rate: .003, epochs: 4, batch_size: 64, entropy_coef: 0});
  const initial = (await custom.act([[1, .25]], true)).actions[0][0];
  for (let step = 0; step < 12; step++) {
    const rollout = await custom.act(observations);
    const rewards = rollout.actions.map(action => -((action[0] - .8) ** 2));
    await custom.update({observations, actions: rollout.actions, oldLogProbs: rollout.logProbs,
      advantages: rewards.map((reward, i) => reward - rollout.values[i]), returns: rewards});
  }
  const final = await custom.act([[1, .25]], true);
  assert.ok(Math.abs(final.actions[0][0] - .8) < Math.abs(initial - .8) * .4,
    `A ${layers}-layer policy must learn the bandit task.`);
  const saved = await custom.serialize();
  assert.equal(saved.config.hidden_layers, layers);
  assert.equal(Object.keys(saved.weights).length, 4 * layers + 5);
  assert.ok(saved.weights[`actor_w${layers}`]);
  assert.equal(saved.weights[`actor_w${layers + 1}`], undefined);
  const imported = await PPO.restore(JSON.parse(JSON.stringify(saved)));
  assert.deepEqual(await imported.act([[1, .25]], true), final);
  assert.deepEqual(await imported.predict([[1,.25]]),final.actions,
    "Every supported policy depth preserves its deterministic actions in the playback path.");
  architectures.push({layers, initial, learned: final.actions[0][0]});
  imported.dispose();
  custom.dispose();
}
assert.throws(() => new PPO(2, 1, {hidden_layers: 0}), /integer/);
assert.throws(() => new PPO(2, 1, {hidden_layers: 5}), /integer/);

// Exercise the actual worker's option validation, legacy migration, rollout
// boundaries, and GAE against a deterministic tiny environment.
const workerSource = await readFile(new URL("../morpholab/static/browser-worker.js", import.meta.url), "utf8");
const worker = {
  onmessage: null, performance, setTimeout, clearTimeout,
  postMessage: message => {
    if (message.event === "checkpoint") worker.onmessage({data: {op: "saved", token: message.token}});
  },
};
vm.runInNewContext(`${workerSource}\nglobalThis.internals={configFor,train,restore};`, worker);
const runtime = {validateBody: value => value, validateEnvironment: value => value};
const customOptions = {body: {}, environment: {}, hidden_size: 32, hidden_layers: 3,
  learning_rate: .001, batch_size: 32, epochs: 2, rollout_steps: 16, gamma: .9,
  entropy_coef: .01, clip_range: .15};
const configured = worker.internals.configFor(customOptions, runtime);
for (const key of Object.keys(customOptions)) assert.deepEqual(configured[key], customOptions[key]);
for (const [key, bad] of Object.entries({hidden_size: 15, hidden_layers: 5,
  learning_rate: .011, batch_size: 15, epochs: 21, rollout_steps: 15, gamma: 1,
  entropy_coef: -.001, clip_range: .01})) {
  assert.throws(() => worker.internals.configFor({...customOptions, [key]: bad}, runtime));
}
let actualOptions;
const updates = [];
class ToyPolicy {
  constructor(_obs, _action, options) { actualOptions = options; }
  async act(batch) { return {actions: batch.map(() => [0]), values: batch.map(() => 0), logProbs: batch.map(() => 0)}; }
  async update(batch) { updates.push(JSON.parse(JSON.stringify(batch))); return {loss: 0}; }
  async serialize() { return {config: actualOptions}; }
  dispose() {}
}
class ToyWalker {
  observationSize = 1;
  actionSize = 1;
  reset() { return [0]; }
  step() { return {obs: [0], reward: 1, terminated: false, truncated: false, info: {velocity: 0}}; }
  getFrame() { return {}; }
  close() {}
}
const toyRuntime = {...runtime, BrowserWalker: ToyWalker, PPO: ToyPolicy,
  RunningNormalizer: class { update() {} normalize(value) { return value; } serialize() { return {}; } },
  exportMJCF: () => "<mujoco/>", hardware: {}, mj: {}};
await worker.internals.train({...configured, total_steps: 4, n_envs: 1, rollout_steps: 2, curriculum: false,best_checkpoint:false}, toyRuntime);
assert.equal(actualOptions.hidden_layers, 3);
assert.equal(actualOptions.learning_rate, .001);
assert.equal(updates.length, 2, "Configured rollout length determines PPO update boundaries.");
for (const update of updates) {
  assert.equal(update.observations.length, 2);
  assert.ok(Math.abs(update.advantages[0] - (1 + .9 * .95)) < 1e-12, "GAE uses configured gamma.");
  assert.equal(update.advantages[1], 1);
}
const legacyRuntime = {...toyRuntime, PPO: {restore: async () => ({dispose() {}})},
  RunningNormalizer: {restore: () => ({})}};
const migrated = await worker.internals.restore({format: "morpholab-browser-run-v1",
  config: {body: {}, environment: {}},
  policy: {obsSize: 1, actionSize: 1, config: {hidden_size: 8, learning_rate: .001, epochs: 3}},
  normalizer: {size: 1}}, legacyRuntime);
assert.equal(migrated.config.hidden_size, 8, "Import retains older small hidden widths.");
assert.equal(migrated.config.hidden_layers, 2, "Old checkpoints default to two hidden layers.");
assert.equal(migrated.config.learning_rate, .001);
assert.equal(migrated.config.epochs, 3);

assert.equal(tf.memory().numTensors, initialTensors, "Training and failed imports release all tensors.");
console.log(JSON.stringify({backend: backend.training_device, initial_action: before.actions[0][0],
  learned_action: after.actions[0][0], architectures, worker_options_and_gae: "passed",
  tensors_remaining: tf.memory().numTensors, result: "passed"}));
