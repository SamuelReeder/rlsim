"use strict";

// Shared by the WebGL and canvas renderers. Only the viewing camera is filtered;
// simulation poses, contacts, and saved replay data remain untouched.
globalThis.RLsimCameraFollow = class {
  static visualCenter(body, frame) {
    const visual = body?.visual;
    if (!visual?.bounds) return null;
    const center = visual.bounds.min.map((value, axis) => (value + visual.bounds.max[axis]) / 2);
    const root = frame?.bodies?.find(item => item.name === (body.root_body || visual.root_body))?.pos;
    const rest = visual.bind_root?.slice(12, 15);
    if (root?.length === 3 && rest?.length === 3 && [...root, ...rest].every(Number.isFinite)) {
      return center.map((value, axis) => value + root[axis] - rest[axis]);
    }
    return center;
  }

  constructor(camera) {
    this.camera = camera;
    this.mode = "preview";
    this.goal = null;
    this.lastTime = null;
    this.lastDraw = null;
    this.wasFollowing = camera.follow;
  }

  reset() {
    this.goal = null;
    this.lastTime = null;
    this.lastDraw = null;
    this.wasFollowing = this.camera.follow;
  }

  setMode(mode) {
    if (mode === this.mode) return;
    this.mode = mode;
    this.reset();
  }

  setFrame(frame, body) {
    const camera = this.camera;
    if (!camera.follow || this.mode === "training") {
      this.reset();
      return;
    }
    const root = frame.bodies?.find(item => item.name === body?.root_body)?.pos;
    const position = root || frame.torso_position || [frame.position || 0, 0, .8];
    if (!position.slice(0, 2).every(Number.isFinite)) return;
    const time = Number.isFinite(frame.time) ? frame.time : null;
    const restart = time !== null && this.lastTime !== null &&
      (time < this.lastTime - 1e-6 || time - this.lastTime > .5);
    const first = !this.goal || !this.wasFollowing;
    this.goal = position.slice(0, 2);
    if (first || restart || this.mode === "preview") {
      camera.target[0] = position[0];
      camera.target[1] = position[1];
    }
    // Keep the ground/horizon steady through the root's walking bounce or fall.
    // The user's framing (including its height) changes only through reset().
    this.lastTime = time;
    this.wasFollowing = true;
  }

  advance(now = globalThis.performance.now()) {
    const camera = this.camera;
    const dt = this.lastDraw === null ? 1 / 60 : Math.min(.1, Math.max(0, (now - this.lastDraw) / 1000));
    this.lastDraw = now;
    if (!camera.follow || this.mode === "training" || !this.goal) {
      this.wasFollowing = camera.follow;
      return false;
    }
    const alpha = -Math.expm1(-dt / .18);
    let moved = false;
    for (let axis = 0; axis < 2; axis++) {
      const delta = this.goal[axis] - camera.target[axis];
      if (Math.abs(delta) <= 1e-5) continue;
      camera.target[axis] += Math.abs(delta) < 1e-4 ? delta : delta * alpha;
      moved = true;
    }
    return moved;
  }
};
