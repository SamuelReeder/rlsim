/** Real MuJoCo WASM tests for conventional robot imports and 3D dynamics. */
import assert from "node:assert/strict";
import {readFile} from "node:fs/promises";
import test from "node:test";
import loadMujoco from "@mujoco/mujoco";
import {preparePhysicsFiles, validateImportedBody, ImportedWalker} from "../morpholab/static/imported-physics.mjs";

const mj = await loadMujoco();
const bytes = text => new TextEncoder().encode(text);
const fixture = async name => new Uint8Array(await readFile(new URL(`../examples/imports/${name}`, import.meta.url)));
const pendulum = await preparePhysicsFiles(mj, [{path: "pendulum.xml", data: await fixture("pendulum.xml")}]);
const freeBody = await preparePhysicsFiles(mj, [{path: "free-body.xml", data: await fixture("free-body.xml")}]);
const armFiles = [
  {path: "robot/serial-arm.urdf", data: await fixture("serial-arm.urdf")},
  {path: "robot/meshes/box.obj", data: await fixture("meshes/box.obj")},
];
const arm = await preparePhysicsFiles(mj, armFiles, {root_body: "forearm"});
const keepRunning = {preset: "balance", terminate_on_fall: false};

function using(body, options, callback, environment = {}) {
  const env = new ImportedWalker(mj, body, environment, options);
  try { return callback(env); } finally { env.close(); }
}

test("authored MJCF keeps motor torque, ranges, joint limits and keyframe", () => {
  using(pendulum, {seed: 17, task: keepRunning}, env => {
    assert.equal(env.actionSize, 1);
    assert.equal(env.observationSize, 13);
    assert.equal(env.observationNames.length, env.observationSize);
    assert.equal(env.observationSchema, "rlsim-general-v1");
    assert.equal(env.model.actuator_gear[0], 12);
    assert.deepEqual(Array.from(env.model.actuator_ctrlrange), [-.4, .8]);
    assert.deepEqual(Array.from(env.model.jnt_range), [-1.8, 1.8]);
    assert.ok(Math.abs(env.data.qpos[0] - .2) <= .005);
    env.step([1]);
    assert.ok(Math.abs(env.data.ctrl[0] - .8) < 1e-12);
    env.step([-1]);
    assert.equal(env.data.ctrl[0], -.4);
    env.step([0]);
    assert.ok(Math.abs(env.data.ctrl[0] - .2) < 1e-12);
    assert.ok(Math.abs(env.data.time - .06) < 1e-12);
    env.reset(17);
    for (let step = 0; step < 30; step++) env.step([1]);
    const positive = env.data.qpos[0];
    env.reset(17);
    for (let step = 0; step < 30; step++) env.step([-1]);
    assert.ok(Math.abs(env.data.qpos[0] - positive) > .1, "Authored motor must affect motion");
  });
});

test("URDF package mesh and inertials survive portable canonical MJCF export", () => {
  assert.equal(arm.metadata.source_format, "urdf");
  assert.equal(arm.metadata.actuators, 2);
  assert.equal(arm.assets[0].path, "robot/meshes/box.obj");
  const portable = validateImportedBody(JSON.parse(JSON.stringify(arm)));
  using(portable, {task: keepRunning}, env => {
    assert.equal(env.model.nq, 2, "Authored fixed-base arm must stay fixed");
    assert.equal(env.model.nbody, 4);
    assert.equal(env.trackedBody, "forearm");
    assert.equal(env.actionSize, 2);
    assert.deepEqual(Array.from(env.model.body_mass).slice(1), [3, 1, .6]);
    assert.deepEqual(Array.from(env.model.jnt_range), [-1.5, 1.5, -2, 2]);
    assert.deepEqual(Array.from(env.model.jnt_actfrcrange), [-40, 40, -25, 25]);
    const first = env.getFrame(), second = env.getFrame();
    const mesh = first.geoms.find(geom => geom.type === "mesh");
    assert.ok(mesh && first.meshes[mesh.mesh]);
    assert.equal(first.meshes[mesh.mesh].vertices.length, 24);
    assert.equal(first.meshes[mesh.mesh].faces.length, 36);
    assert.equal(second.meshes, undefined, "Mesh bytes are sent once per reset");
    assert.ok(first.bodies.some(body => body.name === "forearm" && body.quat.length === 4));
    const initial = Array.from(env.data.qpos);
    for (let i = 0; i < 25; i++) {
      const result = env.step([.1, -.1]);
      assert.ok(result.obs.every(Number.isFinite));
      assert.ok(Number.isFinite(result.reward));
    }
    assert.notDeepEqual(Array.from(env.data.qpos), initial);
  });
});

test("URDF root can explicitly become a floating six-DOF body", async () => {
  const floating = await preparePhysicsFiles(mj, armFiles, {floating_base: true});
  using(floating, {task: keepRunning}, env => {
    assert.equal(env.model.nq, 9); // free 7 + two hinge coordinates
    assert.equal(env.model.nv, 8);
    assert.equal(env.model.nu, 2);
    assert.equal(env.model.jnt_type[0], mj.mjtJoint.mjJNT_FREE.value);
  });
});

test("free 3D body preserves rotations, moves laterally and makes real contacts", () => {
  using(freeBody, {task: keepRunning, seed: 12}, env => {
    assert.equal(env.actionSize, 3, "Ball joint needs three independent axis motors");
    assert.equal(env.model.nq, 11);
    assert.equal(env.model.nv, 9);
    const start = env.getFrame();
    env.data.qvel[1] = .5; // world-y velocity is unconstrained
    env.data.qvel[3] = .4; // roll angular velocity is unconstrained
    let contacts = 0;
    for (let i = 0; i < 120; i++) {
      const result = env.step([0, 0, 0]);
      contacts += env.data.ncon;
      assert.ok(result.obs.every(Number.isFinite));
    }
    const end = env.getFrame();
    assert.ok(Math.abs(end.torso_position[1] - start.torso_position[1]) > .02);
    assert.notDeepEqual(end.bodies[1].quat, start.bodies[1].quat);
    assert.ok(contacts > 0);
    assert.ok(end.torso_position[2] < start.torso_position[2]);
  });
});

test("seeded randomization is reproducible and task controls target/tracked body", () => {
  using(arm, {randomize: true, task: {
    preset: "target", tracked_body: "upper", target: [1, 2, 3],
    episode_seconds: 1, terminate_on_fall: false,
  }}, env => {
    const obs = env.reset(29);
    const frame = env.getFrame();
    const step = env.step([.1, .1]);
    assert.equal(env.trackedBody, "upper");
    const at = env.model.nq + env.model.nv + 1;
    assert.deepEqual(Array.from(obs.slice(at, at + 3)), Array.from(Float32Array.from([1, 2, 3].map((x, i) => x - frame.torso_position[i]))));
    assert.deepEqual(env.reset(29), obs);
    assert.deepEqual(env.step([.1, .1]), step);
    assert.ok(env.last_info.mass_scale >= .8 && env.last_info.mass_scale <= 1.2);
    assert.equal(env.max_episode_steps, 50);
    env.setTrainingProgress(0);
    assert.equal(env.target_speed, .25);
    assert.equal(env.setTrainingProgress(1).length, env.observationSize);
  });
});

test("local relative includes and mesh directories resolve without network", async () => {
  const body = await preparePhysicsFiles(mj, [
    {path: "bundle/main.xml", data: bytes('<mujoco><compiler meshdir="meshes"/><asset><mesh name="box" file="box.obj"/></asset><include file="parts/body.xml"/></mujoco>')},
    {path: "bundle/parts/body.xml", data: bytes('<mujoco><worldbody><body name="floating" pos="0 0 1"><freejoint/><geom type="mesh" mesh="box" mass="1"/></body></worldbody></mujoco>')},
    {path: "bundle/meshes/box.obj", data: await fixture("meshes/box.obj")},
  ], {entry: "bundle/main.xml"});
  using(body, {task: keepRunning}, env => {
    assert.equal(env.model.nmesh, 1);
    assert.equal(env.model.nq, 7);
    assert.equal(env.actionSize, 0, "A free joint alone is not a fabricated motor");
    assert.ok(env.step([]).obs.every(Number.isFinite));
  });
});

test("disabled auto motors allow preview and floor creation avoids duplicates", async () => {
  const noMotors = await preparePhysicsFiles(mj, [{path: "free.xml", data: await fixture("free-body.xml")}], {auto_actuators: false, add_floor: false});
  using(noMotors, {task: keepRunning}, env => {
    assert.equal(env.actionSize, 0);
    assert.equal(env.getFrame().geoms.some(g => g.type === "plane"), false);
  });
  const floor = await preparePhysicsFiles(mj, [{path: "floor.xml", data: bytes('<mujoco><worldbody><geom type="plane" size="10 10 1" friction=".4 .005 .001"/><body pos="0 0 1"><freejoint/><geom type="sphere" size=".2" friction=".7 .1 .01"/></body></worldbody></mujoco>')}]);
  using(floor, {task: keepRunning}, env => {
    assert.equal(env.getFrame().geoms.filter(g => g.type === "plane").length, 1);
    assert.equal(env.model.geom_friction[0], .2);
    assert.equal(env.model.geom_friction[3], .7, "Authored non-floor friction is preserved");
    assert.equal(env.model.opt.gravity[2], -3);
  }, {gravity: 3, friction: .2});
});

test("missing/ambiguous/external assets and invalid XML fail with useful errors", async () => {
  await assert.rejects(preparePhysicsFiles(mj, [armFiles[0]]), /Missing asset/);
  await assert.rejects(preparePhysicsFiles(mj, [{path: "../outside.xml", data: bytes("<mujoco/>")}]), /leaves/);
  await assert.rejects(preparePhysicsFiles(mj, [{path: "bad.xml", data: bytes('<mujoco><asset><mesh file="https://example.com/model.obj"/></asset></mujoco>')}]), /locally/);
  await assert.rejects(preparePhysicsFiles(mj, [{path: "bad.xml", data: bytes('<!DOCTYPE x [<!ENTITY x SYSTEM "secret">]><mujoco/>')}], {entry: "bad.xml"}), /entities/);
  assert.throws(() => validateImportedBody({...pendulum, assets: [{path: "mesh.obj", data: "%%%"}]}), /base64/);
});

test("inherited joint types, existing default floor and heightfield geometry remain correct", async () => {
  const inherited = await preparePhysicsFiles(mj, [{
    path: "defaults.mjcf",
    data: bytes('<mujoco><default><default class="floor"><geom type="plane" size="10 10 .1"/></default><joint type="ball"/></default><worldbody><geom class="floor"/><body pos="0 0 1"><joint name="ball"/><geom type="sphere" size=".1"/></body></worldbody></mujoco>'),
  }]);
  using(inherited, {task: keepRunning}, env => {
    assert.equal(env.actionSize, 3);
    assert.equal(env.getFrame().geoms.filter(g => g.type === "plane").length, 1);
  });
  const terrain = await preparePhysicsFiles(mj, [{
    path: "terrain.xml", data: bytes('<mujoco><asset><hfield name="terrain" nrow="2" ncol="2" size="1 1 .3 .1" elevation="0 1 0 1"/></asset><worldbody><geom type="hfield" hfield="terrain"/></worldbody></mujoco>'),
  }], {add_floor: false});
  using(terrain, {task: keepRunning}, env => {
    const frame = env.getFrame(), ground = frame.geoms.find(g => g.type === "hfield");
    assert.equal(frame.meshes[ground.mesh].vertices.length, 12);
    assert.equal(frame.meshes[ground.mesh].faces.length, 6);
    assert.deepEqual(Array.from(env.model.hfield_data), [0, 1, 0, 1]);
  });
});

test("closed imported environments release both WASM handles", () => {
  const env = new ImportedWalker(mj, pendulum);
  const model = env.model, data = env.data;
  env.close(); env.close();
  assert.ok(model.isDeleted());
  assert.ok(data.isDeleted());
  assert.throws(() => env.reset(), /closed/);
});
