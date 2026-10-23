import assert from "node:assert/strict";
import test from "node:test";
import loadMujoco from "@mujoco/mujoco";
import {importVisualFiles, rebuildPhysicalRig} from "../morpholab/static/scene-runtime.mjs";
import {validateImportedBody, ImportedWalker} from "../morpholab/static/imported-physics.mjs";
import {tinyRigGLB} from "./scene-fixtures.mjs";

globalThis.ProgressEvent ||= class {
  constructor(type, options) { this.type = type; Object.assign(this, options); }
};
const mj = await loadMujoco();

test("editable physics survives the worker/checkpoint JSON boundary and compiles after an edit", async () => {
  const imported = await importVisualFiles([{path: "character.glb", data: tinyRigGLB()}]);
  const portable = validateImportedBody(JSON.parse(JSON.stringify(imported)));
  assert.deepEqual(portable.physical_rig, imported.physical_rig);
  assert.notEqual(portable.physical_rig, imported.physical_rig);
  const editedRig = structuredClone(portable.physical_rig);
  const root = editedRig.bodies.find(body => body.id === editedRig.root);
  root.mass *= 2;
  const edited = validateImportedBody(rebuildPhysicalRig(portable, editedRig));
  assert.deepEqual(edited.visual.files, imported.visual.files, "Physics edits must preserve source assets.");
  assert.deepEqual(imported.physical_rig, portable.physical_rig, "Editing must not mutate an existing saved body.");
  const env = new ImportedWalker(mj, edited, {}, {task: {preset: "balance", terminate_on_fall: false}});
  try {
    const rootId = mj.mj_name2id(env.model, mj.mjtObj.mjOBJ_BODY.value, editedRig.root);
    assert.equal(env.trackedBody, editedRig.root);
    assert.ok(Math.abs(env.model.body_mass[rootId] - root.mass) < 1e-5);
    for (let i = 0; i < 25; i++) {
      assert.ok(env.step(new Array(env.actionSize).fill(0)).obs.every(Number.isFinite));
    }
  } finally { env.close(); }
});

test("legacy body bundles remain valid, while corrupt editable graphs fail at import", async () => {
  const body = await importVisualFiles([{path: "character.glb", data: tinyRigGLB()}]);
  const legacy = structuredClone(body);
  delete legacy.physical_rig;
  assert.equal(validateImportedBody(legacy).xml, body.xml);
  const mismatch = structuredClone(body);
  mismatch.root_body = "wrong_root";
  assert.throws(() => validateImportedBody(mismatch), /rig root/i);
  const cycle = structuredClone(body);
  cycle.physical_rig.bodies[0].parent = cycle.physical_rig.bodies[0].id;
  assert.throws(() => validateImportedBody(cycle));
  const invalidMass = structuredClone(body);
  invalidMass.physical_rig.bodies[0].mass = -1;
  assert.throws(() => validateImportedBody(invalidMass));
});
