import test from "node:test";
import assert from "node:assert/strict";
import "../morpholab/static/camera-follow.js";

const create = () => {
  const camera = {follow: true, target: [0, 0, .75]};
  return {camera, follow: new globalThis.RLsimCameraFollow(camera)};
};
const frame = (time, x, y, z) => ({
  time, geoms: [], bodies: [{name: "pelvis", pos: [x, y, z]}],
  torso_position: [99, 99, 99],
});
const body = {root_body: "pelvis"};

test("training samples and episode resets cannot shake the camera or alter physics", () => {
  const {camera, follow} = create();
  follow.setMode("training");
  for (let i = 0; i < 40; i++) {
    const snapshot = frame((i % 7) / 10, i % 2 ? 4 : -3, i / 2, i % 2 ? .1 : 2);
    const original = structuredClone(snapshot);
    follow.setFrame(snapshot, body);
    follow.advance(i * 350);
    assert.deepEqual(camera.target, [0, 0, .75]);
    assert.deepEqual(snapshot, original);
  }
});

test("inference follows XY smoothly while the horizon ignores bounce and falls", () => {
  const {camera, follow} = create();
  follow.setMode("inference");
  follow.setFrame(frame(0, 0, 0, 1), body);
  follow.advance(0);
  follow.setFrame(frame(.02, 1, 2, .1), body);
  assert.deepEqual(camera.target, [0, 0, .75]);
  assert.equal(follow.advance(100), true);
  assert.ok(camera.target[0] > 0 && camera.target[0] < 1);
  assert.ok(camera.target[1] > 0 && camera.target[1] < 2);
  assert.equal(camera.target[2], .75);
  for (let i = 2; i < 50; i++) follow.advance(i * 100);
  assert.ok(Math.abs(camera.target[0] - 1) < 1e-4);
  assert.equal(follow.advance(5000), false);
});

test("reset, seek and follow toggle re-anchor without panning across missing motion", () => {
  const {camera, follow} = create();
  follow.setMode("inference");
  follow.setFrame(frame(2, 10, 3, 1), body);
  follow.setFrame(frame(0, 0, 0, .2), body);
  assert.deepEqual(camera.target, [0, 0, .75]);
  follow.setFrame(frame(8, -4, -2, 1), body);
  assert.deepEqual(camera.target, [-4, -2, .75]);
  camera.follow = false;
  follow.setFrame(frame(8.02, 12, 5, 1), body);
  follow.advance(100);
  assert.deepEqual(camera.target, [-4, -2, .75]);
  camera.follow = true;
  follow.setFrame(frame(8.04, 12, 5, 1), body);
  assert.deepEqual(camera.target, [12, 5, .75]);
});

test("camera damping is based on elapsed time, not the display refresh rate", () => {
  const targetAfter = hz => {
    const {camera, follow} = create();
    follow.setMode("inference");
    follow.setFrame(frame(0, 0, 0, 1), body);
    follow.advance(0);
    follow.setFrame(frame(.02, 3, -1, 2), body);
    for (let i = 1; i <= hz; i++) follow.advance(i * 1000 / hz);
    return camera.target;
  };
  const low = targetAfter(30), high = targetAfter(120);
  low.forEach((v, i) => assert.ok(Math.abs(v - high[i]) < 1e-9));
});

test("built-in bodies use torso position and malformed roots cannot poison camera", () => {
  const {camera, follow} = create();
  follow.setFrame({time: 0, torso_position: [2, 3, .6]}, {});
  assert.deepEqual(camera.target, [2, 3, .75]);
  follow.setFrame(frame(.1, NaN, 2, 1), body);
  follow.advance(20);
  assert.deepEqual(camera.target, [2, 3, .75]);
});

test("reset framing moves with an imported root while preserving its rest offset", () => {
  const imported = {
    root_body: "pelvis",
    visual: {bounds: {min: [-1, -2, 0], max: [1, 2, 2]},
      bind_root: [1,0,0,0,0,1,0,0,0,0,1,0,.5,-.2,.7,1]},
  };
  assert.deepEqual(globalThis.RLsimCameraFollow.visualCenter(imported, frame(1, 8.5, 2.8, .2)), [8, 3, .5]);
  assert.deepEqual(globalThis.RLsimCameraFollow.visualCenter(imported, null), [0, 0, 1]);
  assert.equal(globalThis.RLsimCameraFollow.visualCenter({}, null), null);
});
