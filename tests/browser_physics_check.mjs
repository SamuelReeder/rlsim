/**
 * Integration checks against the real MuJoCo WebAssembly engine.
 * Run: node tests/browser_physics_check.mjs
 */
import assert from "node:assert/strict";
import test from "node:test";
import loadMujoco from "@mujoco/mujoco";
import {
  BrowserWalker, PRESETS, validateBody, validateEnvironment, exportMJCF,
  PHYSICS_TIMESTEP, FRAME_SKIP, EPISODE_STEPS,
} from "../morpholab/static/browser-physics.mjs";

const mujoco = await loadMujoco();

function withEnvironment(body, environment, options, callback) {
  const env = new BrowserWalker(mujoco, body, environment, options);
  try { return callback(env); }
  finally { env.close(); }
}

for (const [name, body] of Object.entries(PRESETS)) {
  test(`${name}: real MuJoCo contacts and finite, rendered rollouts`, () => {
    withEnvironment(body, {}, {seed: 18}, env => {
      assert.equal(env.actionSize, body.legs * 2);
      assert.equal(env.observationSize, 18 + body.legs * 4);
      const action = new Float32Array(env.actionSize);
      let contacts = 0, episodes = 0;
      for (let step = 0; step < 350; step++) {
        for (let j = 0; j < action.length; j++) action[j] = 0.08 * Math.sin(step / 13 + j);
        const result = env.step(action);
        assert.equal(result.obs.length, env.observationSize);
        assert.ok(result.obs.every(Number.isFinite));
        assert.ok(Number.isFinite(result.reward));
        assert.ok(Math.abs(result.reward - Object.values(result.info.reward_components).reduce((a, b) => a + b, 0)) < 1e-12);
        contacts += env.data.ncon;
        if (result.terminated || result.truncated) { env.reset(); episodes++; }
      }
      assert.ok(contacts > 0, "Expected actual floor contacts");
      const frame = env.getFrame();
      assert.equal(frame.geoms.length, 2 + body.legs * 3);
      assert.equal(frame.geoms[0].type, "plane");
      assert.equal(frame.geoms[1].type, "capsule");
      for (const geom of frame.geoms) {
        assert.equal(geom.pos.length, 3);
        assert.equal(geom.mat.length, 9);
        assert.equal(geom.size.length, 3);
        assert.equal(geom.rgba.length, 4);
        assert.ok([...geom.pos, ...geom.mat, ...geom.size, ...geom.rgba].every(Number.isFinite));
      }
      assert.ok(episodes >= 0);
    });
  });
}

test("reset seed reproduces full randomized rollout; new episodes advance the generator", () => {
  withEnvironment(PRESETS.quadruped, {}, {randomize: true}, env => {
    const first = env.reset(893);
    const firstInfo = {...env.last_info};
    const actions = new Float32Array(env.actionSize).fill(0.1);
    const step = env.step(actions);
    const frame = env.getFrame();
    assert.deepEqual(env.reset(893), first);
    assert.deepEqual(env.last_info, firstInfo);
    assert.deepEqual(env.step(actions), step);
    assert.deepEqual(env.getFrame(), frame);
    assert.notDeepEqual(env.reset(), first);
    const info = env.last_info;
    assert.ok(info.mass_scale >= 0.8 && info.mass_scale <= 1.2);
    assert.ok(info.motor_scale >= 0.8 && info.motor_scale <= 1.2);
    assert.ok(info.friction_scale >= 0.6 && info.friction_scale <= 1.4);
    env.randomize = false;
    const fixed = env.reset(893);
    assert.deepEqual(Array.from(fixed.slice(-3)), [0, 0, 0]);
    assert.deepEqual(env.model.body_mass, env._baseMass);
    assert.deepEqual(env.model.actuator_gear, env._baseGear);
    assert.deepEqual(env.model.geom_friction, env._baseFriction);
  });
});

test("gravity and friction modify the compiled world and actual trajectories", () => {
  function rollout(environment) {
    return withEnvironment(PRESETS.quadruped, environment, {seed: 7}, env => {
      assert.equal(env.model.geom_friction[0], environment.friction);
      const action = new Float32Array(env.actionSize);
      let result;
      for (let t = 0; t < 35; t++) {
        for (let j = 0; j < action.length; j++) action[j] = 0.15 * Math.sin(t * 0.2 + j);
        result = env.step(action);
      }
      return Array.from(result.obs);
    });
  }
  const earth = rollout({gravity: 9.81, friction: 1});
  const lowGravity = rollout({gravity: 1.62, friction: 1});
  const slippery = rollout({gravity: 9.81, friction: 0.05});
  assert.notDeepEqual(earth, lowGravity);
  assert.notDeepEqual(earth, slippery);
});

test("control timing, curriculum, action clipping, and episode limit", () => {
  withEnvironment(PRESETS.biped, {}, {target_speed: 2}, env => {
    env.setTrainingProgress(0);
    assert.equal(env.target_speed, 0.5);
    env.setTrainingProgress(0.5);
    assert.equal(env.target_speed, 1.25);
    env.setTrainingProgress(2);
    assert.equal(env.target_speed, 2);
    env.reset(7);
    env.step(new Float32Array(env.actionSize).fill(2));
    const clipped = env.getFrame();
    env.reset(7);
    env.step(new Float32Array(env.actionSize).fill(1));
    assert.deepEqual(env.getFrame(), clipped);
    assert.ok(Math.abs(env.data.time - PHYSICS_TIMESTEP * FRAME_SKIP) < 1e-12);
    assert.equal(env.max_episode_steps, EPISODE_STEPS);
    // A one-step time horizon tests truncation independently of falling.
    env.max_episode_steps = 1;
    env.reset();
    const result = env.step(new Float32Array(env.actionSize));
    assert.equal(result.truncated, true);
    assert.equal(result.terminated, false);
    assert.throws(() => env.step([0]), /Expected/);
    assert.throws(() => env.step(new Float32Array(env.actionSize).fill(NaN)), /finite/);
    assert.throws(() => env.setTrainingProgress(NaN), /finite/);
  });
});

test("body and environment validation rejects invalid values and XML injection", () => {
  for (const body of [{legs: 3}, {body_mass: NaN}, {upper_length: Infinity}, {motor_strength: 151}, {unknown: 1}]) {
    assert.throws(() => validateBody(body));
  }
  for (const environment of [{gravity: 0}, {friction: 4}, {gravity: NaN}, {unknown: 1}]) {
    assert.throws(() => validateEnvironment(environment));
  }
  const body = {...PRESETS.biped, name: 'A"><body name="injected"/>&'};
  const xml = exportMJCF(body);
  assert.ok(xml.includes("&quot;&gt;&lt;body"));
  withEnvironment(body, {}, {}, env => {
    assert.equal(env.model.nbody, 2 + body.legs * 2);
    assert.equal(mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_BODY.value, "injected"), -1);
  });
});

test("WASM handles are explicitly released and closed environments reject access", () => {
  const env = new BrowserWalker(mujoco, PRESETS.quadruped);
  const model = env.model, data = env.data;
  env.close();
  assert.ok(model.isDeleted());
  assert.ok(data.isDeleted());
  assert.throws(() => env.step(new Float32Array(8)), /closed/);
  assert.throws(() => env.reset(), /closed/);
  assert.throws(() => env.getFrame(), /closed/);
  env.close();
});
