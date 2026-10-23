import assert from "node:assert/strict";
import test from "node:test";
import * as THREE from "three";
import loadMujoco from "@mujoco/mujoco";
import {importVisualFiles, loadVisualScene, VisualBinding, disposeScene, decodeAsset} from "../morpholab/static/scene-runtime.mjs";
import {tinyRigGLB, tinyOBJ, tinyMTL, tinySTL, tinyFBX} from "./scene-fixtures.mjs";

// Three's file progress events are present in browsers, but not Node 18.
globalThis.ProgressEvent ||= class ProgressEvent { constructor(type, options) { this.type = type; Object.assign(this, options); } };
const mj = await loadMujoco();
const glb = [{path: "character.glb", data: tinyRigGLB()}];
const obj = [{path: "cube.obj", data: tinyOBJ}, {path: "cube.mtl", data: tinyMTL}];
function compiled(body) {
  const model = mj.MjModel.from_xml_string(body.xml), data = new mj.MjData(model);
  mj.mj_forward(model, data);
  return {model, data, close() { data.delete(); model.delete(); }};
}
function frame(model, data) {
  return {geoms: [], bodies: Array.from({length: model.nbody - 1}, (_, index) => {
    const id = index + 1;
    return {name: mj.mj_id2name(model, mj.mjtObj.mjOBJ_BODY.value, id),
      pos: Array.from(data.xpos.slice(id * 3, id * 3 + 3)),
      quat: Array.from(data.xquat.slice(id * 4, id * 4 + 4))};
  })};
}
function skinVertex(scene, index) {
  let mesh;
  scene.traverse(object => { if (object.isSkinnedMesh) mesh ||= object; });
  return mesh.getVertexPosition(index, new THREE.Vector3()).applyMatrix4(mesh.matrixWorld);
}

test("rigged GLB produces articulated MuJoCo joints and preserves the actual mesh", async () => {
  const body = await importVisualFiles(glb);
  assert.equal(body.kind, "mjcf");
  assert.equal(body.visual.mode, "skinned");
  assert.equal(Object.keys(body.visual.bone_bodies).length, 3);
  assert.deepEqual(decodeAsset(body.visual.files[0].data), glb[0].data);
  const env = compiled(body);
  try {
    assert.equal(env.model.nu, 6);
    assert.equal(env.model.njnt, 3); // free root plus both deformation joints, including the leaf
    assert.ok(env.model.body_mass.reduce((sum, mass) => sum + mass, 0) > 9.99);
    env.data.ctrl.fill(.2);
    for (let i = 0; i < 100; i++) mj.mj_step(env.model, env.data);
    assert.ok(env.data.qpos.every(Number.isFinite));
    assert.ok(env.data.qvel.some(value => Math.abs(value) > .01), "Real forces must move the articulated model.");
  } finally { env.close(); }
});

test("skin bones and vertices follow physics transforms without animation playback", async () => {
  const body = await importVisualFiles(glb), loaded = await loadVisualScene(glb);
  const binding = new VisualBinding(loaded.scene, body.visual), env = compiled(body);
  try {
    binding.setFrame(frame(env.model, env.data));
    const before = skinVertex(loaded.scene, 2);
    const scales = binding.bones.map(item => item.bone.scale.clone());
    env.data.qpos[0] += .4;
    mj.mj_forward(env.model, env.data);
    binding.setFrame(frame(env.model, env.data));
    const translated = skinVertex(loaded.scene, 2);
    assert.ok(Math.abs(translated.x - before.x - .4) < 1e-6, "Skin follows root translation.");
    const jointId = mj.mj_name2id(env.model, mj.mjtObj.mjOBJ_JOINT.value, "joint_bone_001");
    const address = env.model.jnt_qposadr[jointId];
    env.data.qpos[address] = Math.cos(.35);
    env.data.qpos[address + 2] = Math.sin(.35);
    mj.mj_forward(env.model, env.data);
    const state = frame(env.model, env.data);
    binding.setFrame(state);
    const after = skinVertex(loaded.scene, 2);
    assert.ok(after.distanceTo(translated) > .25, "The rendered skinned vertex must respond to simulated joint motion.");
    for (const [i, {bone, bodyName}] of binding.bones.entries()) {
      const expected = state.bodies.find(item => item.name === bodyName);
      assert.ok(bone.getWorldPosition(new THREE.Vector3()).distanceTo(new THREE.Vector3(...expected.pos)) < 1e-6);
      assert.deepEqual(bone.scale.toArray(), scales[i].toArray());
    }
  } finally { binding.dispose(); env.close(); }
});

test("65-bone character compiles without replacing its skeleton with a preset", async () => {
  const body = await importVisualFiles([{path: "many-bones.glb", data: tinyRigGLB(65)}]);
  assert.equal(Object.keys(body.visual.bone_paths).length, 65);
  const env = compiled(body);
  try { assert.equal(env.model.nu, 64 * 3); assert.equal(env.model.nbody, 66); }
  finally { env.close(); }
});

for (const [name, files] of [
  ["OBJ + MTL", obj], ["STL", [{path: "triangle.stl", data: tinySTL}]], ["FBX", [{path: "triangle.fbx", data: tinyFBX}]],
]) {
  test(`${name} retains its geometry and creates an explicitly rigid six-control body`, async () => {
    const loaded = await loadVisualScene(files);
    let meshes = 0;
    loaded.scene.traverse(object => { if (object.isMesh) meshes++; });
    assert.ok(meshes > 0);
    disposeScene(loaded.scene);
    const body = await importVisualFiles(files, {mass: 3, scale: 2});
    assert.equal(body.visual.mode, "rigid");
    assert.match(body.metadata.notes[0], /no articulated limbs/);
    const env = compiled(body);
    try {
      assert.equal(env.model.nu, 6);
      assert.equal(env.model.njnt, 1);
      assert.ok(Math.abs(env.model.body_mass[1] - 3) < 1e-6);
    } finally { env.close(); }
  });
}

test("rigid visual follows the free body's position and orientation", async () => {
  const body = await importVisualFiles(obj), loaded = await loadVisualScene(obj);
  const binding = new VisualBinding(loaded.scene, body.visual);
  const initial = binding.group.matrix.clone();
  const root = new THREE.Matrix4().fromArray(body.visual.bind_root);
  const position = new THREE.Vector3(), quaternion = new THREE.Quaternion(), scale = new THREE.Vector3();
  root.decompose(position, quaternion, scale);
  position.x += 2;
  binding.setFrame({bodies: [{name: body.root_body, pos: position.toArray(), quat: [1,0,0,0]}]});
  assert.ok(Math.abs(binding.group.matrix.elements[12] - initial.elements[12] - 2) < 1e-8);
  binding.dispose();
});

test("missing dependencies and remote URLs fail locally, and rigid meshes cannot claim a skeleton", async () => {
  await assert.rejects(importVisualFiles([{path: "cube.obj", data: tinyOBJ}]), /Missing local asset/);
  await assert.rejects(importVisualFiles(obj, {rig: "skeleton"}), /no articulated skeleton/);
  const source = JSON.stringify({asset: {version:"2.0"}, scene:0, scenes:[{nodes:[0]}],
    nodes:[{mesh:0}], meshes:[{primitives:[{attributes:{POSITION:0}}]}],
    buffers:[{uri:"https://example.invalid/private.bin",byteLength:36}],
    bufferViews:[{buffer:0,byteOffset:0,byteLength:36}],
    accessors:[{bufferView:0,componentType:5126,type:"VEC3",count:3,min:[0,0,0],max:[1,1,1]}]});
  let externalFetches = 0;
  const originalFetch = globalThis.fetch;
  globalThis.fetch = (...args) => { if (String(args[0]).startsWith("http")) externalFetches++; return originalFetch(...args); };
  try {
    await assert.rejects(importVisualFiles([{path:"remote.gltf",data:new TextEncoder().encode(source)}]), /External asset URLs/);
    assert.equal(externalFetches, 0);
  } finally { globalThis.fetch = originalFetch; }
});
