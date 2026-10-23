import * as THREE from "three";
import {GLTFLoader} from "three/addons/loaders/GLTFLoader.js";
import {FBXLoader} from "three/addons/loaders/FBXLoader.js";
import {OBJLoader} from "three/addons/loaders/OBJLoader.js";
import {MTLLoader} from "three/addons/loaders/MTLLoader.js";
import {STLLoader} from "three/addons/loaders/STLLoader.js";
import {TGALoader} from "three/addons/loaders/TGALoader.js";
import "./camera-follow.js";
import {recoverRestPose, inferPhysicalRig, rebuildPhysicalRig} from "./rig-inference.mjs";
export {rebuildPhysicalRig, validatePhysicalRig} from "./rig-inference.mjs";

const VISUAL_EXTENSIONS = new Set(["glb", "gltf", "fbx", "obj", "stl"]);
const decoder = new TextDecoder();
const clamp = (value, min, max) => Math.max(min, Math.min(max, value));
const extension = name => name.split(".").at(-1).toLowerCase();
const directory = name => name.includes("/") ? name.slice(0, name.lastIndexOf("/") + 1) : "";
const vector = value => new THREE.Vector3().fromArray(value);

// Saved runs cross the worker boundary as new objects. Compare their display
// inputs before disposing GPU resources or decoding the same source files again.
const sameData = (a, b) => {
  if (a === b) return true;
  if (!a || !b || typeof a !== "object" || typeof b !== "object") return false;
  const keys = Object.keys(a);
  return keys.length === Object.keys(b).length && keys.every(key =>
    Object.hasOwn(b, key) && sameData(a[key], b[key]));
};
const sameDisplayBody = (a, b) => Boolean(a && b &&
  a.kind === b.kind && a.xml === b.xml && a.root_body === b.root_body &&
  sameData(a.assets, b.assets) && sameData(a.visual, b.visual) &&
  // Built-in bodies do not carry XML; their geometry comes from their dimensions.
  (a.kind === "mjcf" || sameData(a, b)));

function assetPath(value) {
  let path = String(value);
  try { path = decodeURIComponent(path); } catch { /* A literal percent sign is a valid local filename. */ }
  path = path.replaceAll("\\", "/").replace(/[?#].*$/, "");
  if (/^[a-z][a-z0-9+.-]*:/i.test(path) || path.startsWith("//") || path.startsWith("/")) {
    throw new Error(`External asset URLs are unsupported: ${value}. Include the referenced file locally.`);
  }
  const parts = [];
  for (const part of path.split("/")) {
    if (!part || part === ".") continue;
    if (part === "..") {
      if (!parts.length) throw new Error(`Asset path leaves the selected folder: ${value}`);
      parts.pop();
    } else parts.push(part);
  }
  return parts.join("/");
}

function bytes(data) {
  if (data instanceof Uint8Array) return data;
  if (data instanceof ArrayBuffer) return new Uint8Array(data);
  if (ArrayBuffer.isView(data)) return new Uint8Array(data.buffer, data.byteOffset, data.byteLength);
  throw new Error("Asset contents must be bytes.");
}

export function encodeAsset(data) {
  const value = bytes(data);
  let binary = "";
  for (let offset = 0; offset < value.length; offset += 16384) {
    binary += String.fromCharCode(...value.subarray(offset, offset + 16384));
  }
  return btoa(binary);
}

export function decodeAsset(data) {
  const binary = atob(data);
  return Uint8Array.from(binary, character => character.charCodeAt(0));
}

function mimeType(path) {
  return ({png:"image/png",jpg:"image/jpeg",jpeg:"image/jpeg",webp:"image/webp",gif:"image/gif",
    svg:"image/svg+xml",ktx2:"image/ktx2",gltf:"model/gltf+json",glb:"model/gltf-binary"})[extension(path)]
    || "application/octet-stream";
}

function filesMap(files) {
  if (!Array.isArray(files) || !files.length) throw new Error("Select a 3D model and its supporting files.");
  const result = new Map();
  let total = 0;
  for (const file of files) {
    const path = assetPath(file.path), data = bytes(file.data);
    if (!path || result.has(path)) throw new Error(`Duplicate or empty asset path: ${path}`);
    total += data.byteLength;
    if (total > 100 * 1024 * 1024) throw new Error("The selected 3D assets exceed 100 MB.");
    result.set(path, data);
  }
  return result;
}

function chooseEntry(files, selected) {
  if (selected) {
    const path = assetPath(selected);
    if (!files.has(path) || !VISUAL_EXTENSIONS.has(extension(path))) throw new Error("Select a supported 3D model entry file.");
    return path;
  }
  const candidates = [...files.keys()].filter(path => VISUAL_EXTENSIONS.has(extension(path)));
  if (candidates.length !== 1) {
    throw new Error(candidates.length
      ? "The selection contains multiple models. Choose the model entry file."
      : "Select a GLB, glTF, FBX, OBJ, or STL model.");
  }
  return candidates[0];
}

function findLocal(files, requested, entry) {
  const path = assetPath(requested);
  if (files.has(path)) return path;
  const relative = assetPath(directory(entry) + path);
  if (files.has(relative)) return relative;
  const base = path.split("/").at(-1);
  const matches = [...files.keys()].filter(candidate => candidate.split("/").at(-1) === base);
  if (matches.length === 1) return matches[0];
  throw new Error(`Missing local asset: ${requested}. Include its texture, material, or binary file.`);
}

/** Load only the supplied local bundle. Animation clips are never played. */
export async function loadVisualScene(inputFiles, options = {}) {
  const files = filesMap(inputFiles), entry = chooseEntry(files, options.entry), format = extension(entry);
  const manager = new THREE.LoadingManager(), objectURLs = new Map();
  manager.addHandler(/\.tga(?:[?#].*)?$/i, new TGALoader(manager));
  let pending = 0, loadingError;
  const waiters = [];
  const originalStart = manager.itemStart.bind(manager), originalEnd = manager.itemEnd.bind(manager);
  manager.itemStart = url => { pending++; originalStart(url); };
  manager.itemEnd = url => {
    originalEnd(url);
    if (--pending === 0) for (const resolve of waiters.splice(0)) resolve();
  };
  manager.onError = url => { loadingError ||= new Error(`Could not load local 3D asset: ${url}`); };
  manager.setURLModifier(url => {
    if (/^(data|blob):/i.test(url)) return url;
    const path = findLocal(files, url, entry);
    if (!objectURLs.has(path)) objectURLs.set(path, URL.createObjectURL(new Blob([files.get(path)], {type: mimeType(path)})));
    return objectURLs.get(path);
  });
  let scene;
  try {
    const data = files.get(entry);
    const buffer = data.buffer.slice(data.byteOffset, data.byteOffset + data.byteLength);
    if (format === "glb" || format === "gltf") {
      const model = await new GLTFLoader(manager).parseAsync(format === "glb" ? buffer : decoder.decode(data), directory(entry));
      scene = model.scene;
      const preparation=model.userData?.rlsim_rig_fidelity;
      if(preparation?.format==='rlsim-rig-fidelity-v1')scene.userData.rlsim_preparation=preparation;
    } else if (format === "fbx") {
      scene = new FBXLoader(manager).parse(buffer, directory(entry));
    } else if (format === "obj") {
      const text = decoder.decode(data), loader = new OBJLoader(manager);
      const libraries = [...text.matchAll(/^\s*mtllib\s+(.+?)\s*$/gm)].map(match => match[1]);
      if (libraries.length) {
        // Merge material definitions while keeping each library's texture base.
        const creators = libraries.map(library => {
          const path = findLocal(files, directory(entry) + library, entry);
          const creator = new MTLLoader(manager).parse(decoder.decode(files.get(path)), directory(path));
          creator.preload();
          return creator;
        });
        loader.setMaterials({create(name) {
          const creator = creators.find(value => Object.hasOwn(value.materialsInfo, name));
          return creator ? creator.create(name) : new THREE.MeshStandardMaterial({color: 0x9ba5ab});
        }});
      }
      scene = loader.parse(text);
    } else {
      const geometry = new STLLoader(manager).parse(buffer);
      scene = new THREE.Group();
      scene.add(new THREE.Mesh(geometry, new THREE.MeshStandardMaterial({
        color: geometry.hasColors ? 0xffffff : 0x9ba5ab, vertexColors: Boolean(geometry.hasColors), roughness: .8,
      })));
    }
    if (pending) await new Promise(resolve => waiters.push(resolve));
    if (loadingError) throw loadingError;
    let vertices = 0, meshCount = 0;
    scene.traverse(object => {
      if (object.isMesh) {
        meshCount++;
        vertices += object.geometry?.attributes?.position?.count || 0;
        object.frustumCulled = false;
      }
      // File cameras/lights do not replace the simulation's inspection camera.
      if (object.isLight || object.isCamera) object.visible = false;
    });
    if (!meshCount || !vertices) throw new Error("This file contains no renderable mesh geometry.");
    if (vertices > 2_000_000) throw new Error("This model exceeds two million vertices. Export a smaller mesh.");
    scene.updateMatrixWorld(true);
    return {scene, entry, format, files};
  } catch (error) {
    if (scene) disposeScene(scene);
    throw error;
  } finally {
    for (const url of objectURLs.values()) URL.revokeObjectURL(url);
  }
}

export function disposeScene(scene) {
  const resources = new Set();
  scene.traverse(object => {
    if (object.geometry) resources.add(object.geometry);
    for (const material of Array.isArray(object.material) ? object.material : [object.material]) {
      if (!material) continue;
      resources.add(material);
      for (const value of Object.values(material)) if (value?.isTexture) resources.add(value);
    }
  });
  for (const resource of resources) resource.dispose();
}

function objectAtPath(root, path) {
  return path === "" ? root : path.split("/").reduce((object, index) => object?.children[Number(index)], root);
}

function finiteOption(value, fallback, min, max, name) {
  const result = value ?? fallback;
  if (!Number.isFinite(result) || result < min || result > max) throw new Error(`${name} must be between ${min} and ${max}.`);
  return result;
}

/**
 * Build a real MuJoCo body from a conventional visual model. Rigged models use
 * their actual skeleton hierarchy; unrigged meshes get an explicitly rigid proxy.
 */
export async function importVisualFiles(inputFiles, options = {}) {
  const scale = finiteOption(options.scale, 1, .00001, 1000, "Scale");
  const mass = finiteOption(options.mass, 10, .01, 10000, "Mass");
  const torque = finiteOption(options.motor_strength, 60, .1, 10000, "Motor strength");
  const rig = options.rig || "auto";
  if (!["auto", "skeleton", "rigid"].includes(rig)) throw new Error("Choose automatic, skeleton, or rigid-body physics.");
  const loaded = await loadVisualScene(inputFiles, options);
  const {scene, entry, format, files} = loaded;
  try {
    const rest = recoverRestPose(scene);
    // FBX UnitScaleFactor is in centimetres per file unit. GLTF uses metres.
    const unitScale = format === "fbx" ? (scene.userData.unitScaleFactor || 1) / 100 : 1;
    const factor = scale * unitScale;
    const orientation = new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(1, 0, 0), Math.PI / 2);
    const transform = new THREE.Matrix4().compose(new THREE.Vector3(), orientation, new THREE.Vector3(factor, factor, factor));
    const wrapper = new THREE.Group();
    wrapper.matrixAutoUpdate = false;
    wrapper.matrix.copy(transform);
    wrapper.add(scene);
    wrapper.updateMatrixWorld(true);
    let bounds = new THREE.Box3().setFromObject(wrapper, true);
    if (bounds.isEmpty() || ![...bounds.min.toArray(), ...bounds.max.toArray()].every(Number.isFinite)) {
      throw new Error("This model has invalid geometry bounds.");
    }
    const center = bounds.getCenter(new THREE.Vector3());
    const size = bounds.getSize(new THREE.Vector3());
    const height = Math.max(size.z, .01), clearance = Math.max(height * .015, .005);
    transform.setPosition(-center.x, -center.y, -bounds.min.z + clearance);
    wrapper.matrix.copy(transform);
    wrapper.updateMatrixWorld(true);
    bounds = new THREE.Box3().setFromObject(wrapper, true);
    const fitted = inferPhysicalRig(scene, {mass, torque, height, rig, profile: options.profile || "auto", bounds});
    if (fitted.lift) {
      transform.elements[14] += fitted.lift;
    }
    if (fitted.rig.profile === "humanoid") {
      const heading = new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(0,0,1), fitted.heading);
      const rotation = new THREE.Matrix4().makeRotationFromQuaternion(heading);
      transform.premultiply(rotation);
      for (const physical of fitted.rig.bodies) {
        physical.position = vector(physical.position).applyQuaternion(heading).toArray();
        const q = physical.quaternion;
        const aligned = heading.clone().multiply(new THREE.Quaternion(q[1], q[2], q[3], q[0]));
        physical.quaternion = [aligned.w, aligned.x, aligned.y, aligned.z];
      }
      for (const binding of Object.values(fitted.bindings)) binding.rest_world = new THREE.Matrix4().fromArray(binding.rest_world).premultiply(rotation).toArray();
      fitted.rig.repairs.push("Aligned inferred humanoid forward with the task's +X direction.");
    }
    wrapper.matrix.copy(transform); wrapper.updateMatrixWorld(true);
    bounds = new THREE.Box3().setFromObject(wrapper, true);
    if (format === "obj" || format === "stl") fitted.rig.warnings.push("Source coordinates are treated as metres and Y-up; adjust import scale if needed.");
    if (format === "fbx") fitted.rig.warnings.push(`FBX unit conversion: ${unitScale} metres per source unit.`);
    const name = entry.split("/").at(-1).replace(/\.[^.]+$/, "");
    const body = {
      kind: "mjcf", name, assets: [],
      visual: {
        entry, format, files: [...files].map(([path, data]) => ({path, data: encodeAsset(data)})),
        mode: fitted.rig.profile === "rigid" ? "rigid" : "skinned", transform: transform.toArray(),
        rest_locals: rest.locals, source_bindings: fitted.bindings,
        bounds: {min: bounds.min.toArray(), max: bounds.max.toArray()}, scale, unit_scale: unitScale,
      },
      metadata: {source_format: format,...(scene.userData.rlsim_preparation?{auto_rig:structuredClone(scene.userData.rlsim_preparation)}:{})},
    };
    return rebuildPhysicalRig(body, fitted.rig);
  } finally {
    disposeScene(scene);
  }
}

/** Apply simulator body transforms to a loaded visual; useful without WebGL too. */
export class VisualBinding {
  constructor(scene, visual) {
    this.scene = scene;
    this.visual = visual;
    for (const [path, rest] of Object.entries(visual.rest_locals || {})) {
      const bone = objectAtPath(scene, path);
      if (!bone?.isBone) throw new Error(`The visual skeleton is missing bone ${path}.`);
      bone.position.fromArray(rest.position); bone.quaternion.fromArray(rest.quaternion); bone.scale.fromArray(rest.scale);
      bone.updateMatrix();
    }
    scene.updateMatrixWorld(true);
    this.group = new THREE.Group();
    this.group.matrixAutoUpdate = false;
    this.baseTransform = new THREE.Matrix4().fromArray(visual.transform);
    this.inverseBindRoot = new THREE.Matrix4().fromArray(visual.bind_root).invert();
    this.group.matrix.copy(this.baseTransform);
    this.group.add(scene);
    this.group.updateMatrixWorld(true);
    this.bones = [];
    for (const [path, bodyName] of Object.entries(visual.bone_paths || {})) {
      const bone = objectAtPath(scene, path);
      if (!bone) throw new Error(`The visual skeleton is missing bone ${path}.`);
      this.bones.push({bone, bodyName, offset: new THREE.Matrix4().fromArray(visual.bone_offsets?.[path] || new THREE.Matrix4().toArray()), scale: bone.scale.clone(), worldScale: bone.getWorldScale(new THREE.Vector3())});
    }
    if (!this.bones.length && visual.mode === "skinned") {
      scene.traverse(bone => {
        const bodyName = visual.bone_bodies?.[bone.name];
        if (bodyName) this.bones.push({bone, bodyName, offset: new THREE.Matrix4(), scale: bone.scale.clone(), worldScale: bone.getWorldScale(new THREE.Vector3())});
      });
    }
    // Parents must be updated first, including files whose skin joint list is unordered.
    const depth = object => { let count = 0; for (; object.parent; object = object.parent) count++; return count; };
    this.bones.sort((a, b) => depth(a.bone) - depth(b.bone));
  }

  setFrame(frame) {
    const bodies = new Map((frame.bodies || []).map(body => [body.name, body]));
    const root = bodies.get(this.visual.root_body);
    if (root) {
      const q = root.quat;
      const world = new THREE.Matrix4().compose(vector(root.pos),
        new THREE.Quaternion(q[1], q[2], q[3], q[0]), new THREE.Vector3(1, 1, 1));
      this.group.matrix.copy(world).multiply(this.inverseBindRoot).multiply(this.baseTransform);
    }
    this.group.updateMatrixWorld(true);
    for (const {bone, bodyName, offset, scale, worldScale} of this.bones) {
      const state = bodies.get(bodyName);
      if (!state) continue;
      const q = state.quat;
      const desired = new THREE.Matrix4().compose(vector(state.pos),
        new THREE.Quaternion(q[1], q[2], q[3], q[0]), new THREE.Vector3(1, 1, 1)).multiply(offset).scale(worldScale);
      const local = bone.parent.matrixWorld.clone().invert().multiply(desired);
      local.decompose(bone.position, bone.quaternion, new THREE.Vector3());
      bone.scale.copy(scale);
      bone.updateMatrix();
      bone.updateWorldMatrix(false, false);
    }
    this.group.updateMatrixWorld(true);
    this.scene.traverse(object => { if (object.isSkinnedMesh) object.skeleton.update(); });
  }

  dispose() { disposeScene(this.scene); this.group.removeFromParent(); }
}

function geometryFor(geom, meshes) {
  const size = geom.size || [1, 1, 1];
  if (geom.mesh || geom.type === "mesh" || geom.type === "hfield") {
    const data = typeof geom.mesh === "string" ? meshes.get(geom.mesh) : geom.mesh;
    if (!data) return null;
    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute("position", new THREE.Float32BufferAttribute(data.vertices, 3));
    if (data.faces?.length) geometry.setIndex(new THREE.Uint32BufferAttribute(data.faces, 1));
    geometry.computeVertexNormals();
    return geometry;
  }
  if (geom.type === "plane") return new THREE.PlaneGeometry(size[0] > 0 ? 2 * size[0] : 2000, size[1] > 0 ? 2 * size[1] : 2000);
  if (geom.type === "box") return new THREE.BoxGeometry(2 * size[0], 2 * size[1], 2 * size[2]);
  if (geom.type === "capsule") return new THREE.CapsuleGeometry(size[0], 2 * size[1], 4, 12).rotateX(Math.PI / 2);
  if (geom.type === "cylinder") return new THREE.CylinderGeometry(size[0], size[0], 2 * size[1], 20).rotateX(Math.PI / 2);
  if (geom.type === "sphere") return new THREE.SphereGeometry(size[0], 20, 12);
  if (geom.type === "ellipsoid") return new THREE.SphereGeometry(1, 20, 12).scale(...size);
  return null;
}

function segmentGridLines(grid) {
  // Some software WebGL drivers drop long lines that cross the near plane.
  // Short, contiguous segments preserve the same metre grid and depth testing.
  const geometry = grid.geometry, position = geometry.getAttribute("position"), color = geometry.getAttribute("color");
  const positions = [], colors = [];
  for (let index = 0; index < position.count; index += 2) {
    const a = new THREE.Vector3().fromBufferAttribute(position, index);
    const b = new THREE.Vector3().fromBufferAttribute(position, index + 1);
    const pieces = Math.max(1, Math.ceil(a.distanceTo(b) / 2));
    for (let part = 0; part < pieces; part++) for (const fraction of [part / pieces, (part + 1) / pieces]) {
      positions.push(...a.clone().lerp(b, fraction).toArray());
      colors.push(color.getX(index), color.getY(index), color.getZ(index));
    }
  }
  geometry.setAttribute("position", new THREE.Float32BufferAttribute(positions, 3));
  geometry.setAttribute("color", new THREE.Float32BufferAttribute(colors, 3));
}

export class SceneView {
  constructor(canvas) {
    this.canvas = canvas;
    this.renderer = new THREE.WebGLRenderer({canvas, antialias: true, alpha: false});
    this.renderer.setClearColor(0xf7f7f5);
    this.renderer.setPixelRatio(Math.min(globalThis.devicePixelRatio || 1, 2));
    this.renderer.outputColorSpace = THREE.SRGBColorSpace;
    this.scene = new THREE.Scene();
    this.viewCamera = new THREE.PerspectiveCamera(42, 1, .01, 2000);
    this.viewCamera.up.set(0, 0, 1);
    this.camera = {azimuth: -1.15, elevation: .30, distance: 4.2, follow: true, target: [0, 0, .45]};
    this.followCamera = new globalThis.RLsimCameraFollow(this.camera);
    this.scene.add(new THREE.HemisphereLight(0xffffff, 0xb5b1a7, 2));
    const light = new THREE.DirectionalLight(0xffffff, 2.5);
    light.position.set(-3, -4, 8); this.scene.add(light);
    this.ground = new THREE.Mesh(new THREE.PlaneGeometry(2000, 2000),
      new THREE.MeshStandardMaterial({color: 0xf1f1ee, roughness: 1}));
    this.ground.position.z = -.002;
    this.scene.add(this.ground);
    this.grid = new THREE.GridHelper(60, 60, 0xb9bbb7, 0xd8d9d4);
    segmentGridLines(this.grid);
    this.grid.rotation.x = Math.PI / 2;
    this.grid.position.z = .003;
    this.grid.material.depthWrite = false;
    this.grid.renderOrder = 1;
    this.scene.add(this.grid);
    this.physics = new THREE.Group(); this.scene.add(this.physics);
    this.environment = new THREE.Group(); this.scene.add(this.environment);
    this.targetMarker = new THREE.Group();
    const markerMaterial = new THREE.MeshBasicMaterial({color: 0xde752f, depthTest: false});
    this.targetMarker.add(new THREE.Mesh(new THREE.SphereGeometry(.055, 12, 8), markerMaterial));
    for (let axis = 0; axis < 3; axis++) {
      const ring = new THREE.Mesh(new THREE.TorusGeometry(.2, .008, 5, 40), markerMaterial);
      if (axis === 1) ring.rotation.x = Math.PI / 2;
      if (axis === 2) ring.rotation.y = Math.PI / 2;
      this.targetMarker.add(ring);
    }
    this.targetMarker.traverse(object => { object.renderOrder = 3; });
    this.targetMarker.visible = false;
    this.scene.add(this.targetMarker);
    this.geomObjects = new Map(); this.meshes = new Map();
    this.frame = null; this.dirty = true; this.bodyVersion = 0; this.visual = null; this.showCollision = false;
    this.body = null; this.width = this.height = 0;
    this.resizeObserver = new ResizeObserver(() => this.resize());
    this.resizeObserver.observe(canvas);
    let pointer;
    canvas.addEventListener("pointerdown", event => {
      pointer = [event.clientX, event.clientY]; canvas.setPointerCapture(event.pointerId);
    });
    canvas.addEventListener("pointermove", event => {
      if (!pointer) return;
      this.camera.azimuth -= (event.clientX - pointer[0]) * .008;
      this.camera.elevation = clamp(this.camera.elevation + (event.clientY - pointer[1]) * .007, .025, 1.5);
      pointer = [event.clientX, event.clientY]; this.dirty = true;
    });
    for (const type of ["pointerup", "pointercancel", "lostpointercapture"]) canvas.addEventListener(type, () => { pointer = null; });
    canvas.addEventListener("wheel", event => {
      event.preventDefault();
      this.camera.distance = clamp(this.camera.distance * Math.exp(event.deltaY * .001), .05, 1000);
      this.dirty = true;
    }, {passive: false});
    this.resize();
  }

  resize() {
    const rect = this.canvas.getBoundingClientRect();
    this.width = rect.width; this.height = rect.height;
    this.renderer.setSize(Math.max(1, rect.width), Math.max(1, rect.height), false);
    this.viewCamera.aspect = Math.max(1, rect.width) / Math.max(1, rect.height);
    this.viewCamera.updateProjectionMatrix(); this.dirty = true;
  }

  async setBody(body) {
    if (body === this.body || sameDisplayBody(body, this.body)) {
      this.body = body;
      return this.bodyLoad;
    }
    this.body = body;
    this.followCamera.reset();
    const version = ++this.bodyVersion;
    this.bodyLoad = null;
    this.visual?.dispose(); this.visual = null;
    this.meshes.clear();
    for (const {object} of this.geomObjects.values()) {
      object.geometry.dispose(); object.material.dispose();
    }
    this.geomObjects.clear(); this.physics.clear(); this.environment.clear(); this.physics.visible = true;
    this.dirty = true;
    // UI body selection can be queued just after its first preview frame.
    // Rebuild that frame after clearing the previous body's geometry cache.
    if (this.frame) this.setFrame(this.frame);
    if (!body?.visual) return;
    const visual = body.visual;
    const pending = (async () => {
      const loaded = await loadVisualScene(visual.files.map(file => ({path: file.path, data: decodeAsset(file.data)})), {entry: visual.entry});
      if (version !== this.bodyVersion) { disposeScene(loaded.scene); return; }
      this.visual = new VisualBinding(loaded.scene, visual);
      this.scene.add(this.visual.group); this.physics.visible = false;
      if (this.frame) this.visual.setFrame(this.frame);
      this.dirty = true;
    })();
    this.bodyLoad = pending;
    try { await pending; }
    finally { if (this.bodyLoad === pending) this.bodyLoad = null; }
  }

  setMode(mode) { this.followCamera.setMode(mode); }

  setFrame(frame) {
    if (!frame?.geoms) return;
    this.frame = frame;
    for (const [key, mesh] of Object.entries(frame.meshes || {})) this.meshes.set(key, mesh);
    const current = new Set();
    frame.geoms.forEach((geom, index) => {
      const key = geom.id ?? index;
      current.add(key);
      const signature = `${geom.type}:${typeof geom.mesh === "string" ? geom.mesh : ""}:${geom.size?.join(",")}`;
      let cached = this.geomObjects.get(key);
      if (!cached || cached.signature !== signature) {
        if (cached) {
          cached.object.removeFromParent(); cached.object.geometry.dispose(); cached.object.material.dispose();
          this.geomObjects.delete(key);
        }
        const geometry = geometryFor(geom, this.meshes);
        if (!geometry) return;
        const object = new THREE.Mesh(geometry, new THREE.MeshStandardMaterial({roughness: .7, metalness: .05}));
        object.userData.geomType = geom.type;
        cached = {object, signature}; this.geomObjects.set(key, cached);
        (geom.type === "plane" ? this.environment : this.physics).add(object);
      }
      const {object} = cached, mat = geom.mat || [1,0,0,0,1,0,0,0,1];
      object.position.fromArray(geom.pos);
      object.quaternion.setFromRotationMatrix(new THREE.Matrix4().set(mat[0],mat[1],mat[2],0,mat[3],mat[4],mat[5],0,mat[6],mat[7],mat[8],0,0,0,0,1));
      const color = geom.rgba || [.45,.55,.61,1];
      object.material.color.setRGB(color[0], color[1], color[2]);
      object.material.opacity = color[3]; object.material.transparent = color[3] < 1;
      object.userData.baseOpacity = color[3];
    });
    for (const [key, {object}] of this.geomObjects) {
      if (current.has(key)) continue;
      object.removeFromParent(); object.geometry.dispose(); object.material.dispose(); this.geomObjects.delete(key);
    }
    this.ground.visible = !frame.geoms.some(geom => geom.type === "plane");
    const taskTarget = frame.target || (["target", "custom"].includes(frame.task?.preset) ? frame.task.target : null);
    this.targetMarker.visible = Boolean(taskTarget?.length === 3 && Array.from(taskTarget).every(Number.isFinite));
    if (this.targetMarker.visible) this.targetMarker.position.fromArray(taskTarget);
    this.visual?.setFrame(frame);
    this.followCamera.setFrame(frame, this.body);
    this.dirty = true;
  }

  reset(body = this.body) {
    let bounds = body?.visual?.bounds;
    if (!bounds && body?.kind === "mjcf" && this.physics.children.length) {
      this.physics.updateMatrixWorld(true);
      const box = new THREE.Box3().setFromObject(this.physics);
      if (!box.isEmpty()) bounds = {min: box.min.toArray(), max: box.max.toArray()};
    }
    let size;
    if (bounds) {
      size = vector(bounds.max).sub(vector(bounds.min));
      const center = vector(bounds.max).add(vector(bounds.min)).multiplyScalar(.5);
      this.camera.target = globalThis.RLsimCameraFollow.visualCenter(body, this.frame) || center.toArray();
    } else {
      size = new THREE.Vector3(body?.torso_length || .8, .5,
        (body?.upper_length || .42) + (body?.lower_length || .42));
      const root = this.frame?.torso_position || [this.frame?.position || 0, 0, .8];
      this.camera.target = [root[0], root[1], root[2] * .55];
    }
    this.camera.distance = Math.max(.1, size.length() * 2.4);
    this.followCamera.reset();
    this.setAngle("perspective");
  }

  setAngle(angle) {
    const angles = {perspective: [-1.15, .30], side: [-Math.PI / 2, .025], front: [0, .025]};
    [this.camera.azimuth, this.camera.elevation] = angles[angle] || angles.perspective;
    this.dirty = true;
  }

  draw(now) {
    if (this.followCamera.advance(now)) this.dirty = true;
    if (!this.dirty || !this.width || !this.height) return;
    const overlay = Boolean(this.visual && this.showCollision);
    this.physics.visible = !this.visual || overlay;
    for (const {object} of this.geomObjects.values()) {
      const wireframe = overlay && object.userData.geomType !== "plane";
      object.material.wireframe = wireframe;
      object.material.depthTest = !wireframe;
      object.material.opacity = wireframe ? .55 : object.userData.baseOpacity;
      object.material.transparent = wireframe || object.material.opacity < 1;
      object.renderOrder = wireframe ? 2 : 0;
    }
    const camera = this.camera, target = vector(camera.target);
    this.viewCamera.position.set(
      target.x + Math.cos(camera.azimuth) * Math.cos(camera.elevation) * camera.distance,
      target.y + Math.sin(camera.azimuth) * Math.cos(camera.elevation) * camera.distance,
      target.z + Math.sin(camera.elevation) * camera.distance);
    this.viewCamera.near = Math.max(.001, camera.distance / 1000);
    this.viewCamera.far = Math.max(200, camera.distance * 30);
    this.viewCamera.updateProjectionMatrix();
    this.viewCamera.lookAt(target);
    this.grid.position.x = Math.round(target.x); this.grid.position.y = Math.round(target.y);
    this.renderer.render(this.scene, this.viewCamera); this.dirty = false;
  }
}
