/** Native MuJoCo WASM import and dynamics for authored MJCF and URDF robots. */
import {XMLParser, XMLBuilder, XMLValidator} from "./vendor/xml.mjs";
import {validateEnvironment} from "./browser-physics.mjs";
import {validateTask, evaluateTask} from "./task-regimes.mjs";
import {validatePhysicalRig} from "./physical-rig.mjs";

const decoder = new TextDecoder(), encoder = new TextEncoder();
const parser = new XMLParser({ignoreAttributes: false, preserveOrder: true, trimValues: false});
const builder = new XMLBuilder({ignoreAttributes: false, preserveOrder: true});
const LIMIT_BYTES = 64 * 1024 * 1024;
let modelSequence = 0;
export function actuatorControlNames(model,module){
  const names=Array(model.nu);
  for(let i=0;i<(model.nactuator??model.nu);i++){
    const name=module.mj_id2name(model,module.mjtObj.mjOBJ_ACTUATOR.value,i)||`actuator_${i}`;
    const start=model.actuator_ctrladr?.[i]??i,count=model.actuator_ctrlnum?.[i]??1;
    for(let j=0;j<count;j++)names[start+j]=count===1?name:`${name}[${j}]`;
  }
  return names;
}
const clip = (x, a, b) => Math.min(b, Math.max(a, x));
const typeOf = node => Object.keys(node).find(key => key !== ":@" && key !== "#text");
const attrs = node => node[":@"] || (node[":@"] = {});
const children = node => node[typeOf(node)] || [];
const element = (type, attributes = {}, nested = []) => ({
  [type]: nested, ":@": Object.fromEntries(Object.entries(attributes).map(([key, value]) => [`@_${key}`, String(value)])),
});
function walk(nodes, visit) {
  for (const node of nodes) {
    const type = typeOf(node);
    if (!type || type.startsWith("?") || type.startsWith("!")) continue;
    visit(node, type);
    if (Array.isArray(node[type])) walk(node[type], visit);
  }
}
function parseXML(xml) {
  if (typeof xml !== "string" || xml.length > 4 * 1024 * 1024) throw new Error("XML must be text smaller than 4 MB.");
  if (/<!DOCTYPE|<!ENTITY/i.test(xml)) throw new Error("External XML entities and document types are unsupported. Include local asset files.");
  const valid = XMLValidator.validate(xml);
  if (valid !== true) throw new Error(`Invalid XML: ${valid.err.msg} (line ${valid.err.line}).`);
  const nodes = parser.parse(xml);
  const root = nodes.find(node => ["mujoco", "robot"].includes(typeOf(node)));
  if (!root) throw new Error("Select an MJCF <mujoco> XML or URDF <robot> file.");
  return {nodes, root, type: typeOf(root)};
}
function cleanPath(value) {
  if (typeof value !== "string" || !value || value.length > 1024 || /[\0\r\n]/.test(value)) throw new Error("Invalid asset path.");
  const path = value.replaceAll("\\", "/");
  if (path.startsWith("/") || /^[a-z][a-z0-9+.-]*:/i.test(path)) throw new Error(`Asset path must be local and relative: ${value}`);
  const parts = [];
  for (const part of path.split("/")) {
    if (!part || part === ".") continue;
    if (part === "..") {
      if (!parts.length) throw new Error(`Asset path leaves the imported folder: ${value}`);
      parts.pop();
    } else parts.push(part);
  }
  if (!parts.length) throw new Error("Asset path cannot be empty.");
  return parts.join("/");
}
function bytesToBase64(bytes) {
  let binary = "";
  for (let start = 0; start < bytes.length; start += 32768) binary += String.fromCharCode(...bytes.subarray(start, start + 32768));
  return btoa(binary);
}
function base64ToBytes(text) {
  if (typeof text !== "string" || text.length > LIMIT_BYTES * 1.4 || !/^[A-Za-z0-9+/]*={0,2}$/.test(text)) throw new Error("Invalid base64 asset.");
  try { return Uint8Array.from(atob(text), char => char.charCodeAt(0)); }
  catch { throw new Error("Invalid base64 asset."); }
}
function nameCheck(name) {
  if (typeof name !== "string" || !name.trim() || name.length > 120) throw new Error("Imported body name must contain 1–120 characters.");
  return name;
}
export function validateImportedBody(body) {
  if (!body || body.kind !== "mjcf") throw new Error("Expected an imported MJCF body.");
  nameCheck(body.name);
  const parsed = parseXML(body.xml);
  if (parsed.type !== "mujoco") throw new Error("Imported bodies must contain canonical MJCF. Import URDF using the file importer.");
  if (!Array.isArray(body.assets || []) || (body.assets || []).length > 512) throw new Error("A model can include at most 512 asset files.");
  const paths = new Set();
  let total = body.xml.length;
  const assets = (body.assets || []).map(asset => {
    const path = cleanPath(asset.path);
    if (paths.has(path)) throw new Error(`Duplicate asset path: ${path}`);
    paths.add(path);
    const bytes = base64ToBytes(asset.data);
    total += bytes.length;
    if (total > LIMIT_BYTES) throw new Error("Imported body assets exceed 64 MB.");
    return {path, data: asset.data};
  });
  walk(parsed.nodes, node => {
    for (const key of ["@_file", "@_filename"]) {
      const value = node[":@"]?.[key];
      if (value) cleanPath(value);
    }
  });
  if (body.root_body != null && (typeof body.root_body !== "string" || !body.root_body || body.root_body.length > 200)) throw new Error("Invalid root body name.");
  const result = {kind: "mjcf", name: body.name, xml: body.xml, assets};
  for (const key of ["root_body", "visual", "metadata"]) if (body[key] !== undefined) result[key] = structuredClone(body[key]);
  if (body.physical_rig !== undefined) {
    const rig = validatePhysicalRig(body.physical_rig);
    if (body.root_body !== rig.root) throw new Error("The physical rig root must match the body's tracked root.");
    result.physical_rig = rig;
  }
  return result;
}
function makeVFS(module, assets) {
  const vfs = new module.MjVFS();
  try {
    for (const asset of assets) vfs.addBuffer(asset.path, typeof asset.data === "string" ? base64ToBytes(asset.data) : asset.data);
    return vfs;
  } catch (error) { vfs.delete(); throw error; }
}
function limits(model) {
  if (model.nbody > 129) throw new Error("This model exceeds 128 bodies.");
  if (model.nu > 256) throw new Error("This model exceeds 256 actuators.");
  if (model.nq + model.nv + 10 + model.nu > 8192) throw new Error("This model exceeds 8,192 observation values.");
  if (model.nmeshvert > 500000 || model.nmeshface > 500000) throw new Error("Simplify model meshes to at most 500,000 vertices and faces.");
  if (model.nhfielddata > 250000) throw new Error("Simplify terrain to at most 250,000 height samples.");
}
function compile(module, xml, assets) {
  const vfs = makeVFS(module, assets);
  let model;
  try {
    model = module.MjModel.from_xml_string(xml, vfs);
    limits(model);
    return model;
  } catch (error) {
    model?.delete();
    throw new Error(`MuJoCo could not compile this model: ${error.message || String(error)}`);
  } finally { vfs.delete(); }
}
function savedXML(module, model) {
  const path = `/rlsim-export-${++modelSequence}.xml`;
  try {
    if (!module.mj_saveLastXML(path, model)) throw new Error("MuJoCo could not serialize the imported model.");
    return module.FS.readFile(path, {encoding: "utf8"});
  } finally { try { module.FS.unlink(path); } catch {} }
}
function resolveAsset(reference, context, files, prefix = "") {
  const packagePath = reference.replace(/^package:\/\//, "");
  if (/^[a-z][a-z0-9+.-]*:/i.test(packagePath) || packagePath.startsWith("/")) throw new Error(`Asset must be included locally: ${reference}`);
  const directory = context.includes("/") ? context.slice(0, context.lastIndexOf("/") + 1) : "";
  const attempts = [directory + prefix + packagePath, directory + packagePath, prefix + packagePath, packagePath];
  for (const attempt of attempts) {
    try { const clean = cleanPath(attempt); if (files.has(clean)) return clean; } catch {}
  }
  const suffixes = [packagePath, packagePath.split("/").slice(1).join("/"), packagePath.split("/").at(-1)].filter(Boolean);
  for (const suffix of suffixes) {
    const matches = [...files.keys()].filter(path => path === suffix || path.endsWith("/" + suffix));
    if (matches.length === 1) return matches[0];
    if (matches.length > 1) throw new Error(`Asset reference is ambiguous: ${reference}. Include its directory structure.`);
  }
  throw new Error(`Missing asset "${reference}" referenced by "${context}". Import the XML/URDF together with its mesh, texture and include files.`);
}

/**
 * Compile uploaded MJCF/URDF with its local assets and return portable MJCF.
 * files: [{path, data: Uint8Array}]; no URL or server asset lookup occurs.
 */
export async function preparePhysicsFiles(module, files, options = {}) {
  if (!Array.isArray(files) || !files.length || files.length > 512) throw new Error("Select the model and its local asset files (up to 512).");
  const bundle = new Map();
  let bytes = 0;
  for (const file of files) {
    const path = cleanPath(file.path), data = file.data;
    if (!(data instanceof Uint8Array)) throw new Error(`Asset ${path} must contain bytes.`);
    if (bundle.has(path)) throw new Error(`Duplicate file: ${path}`);
    bytes += data.length;
    if (bytes > LIMIT_BYTES) throw new Error("Imported files exceed 64 MB.");
    bundle.set(path, data);
  }
  const candidates = [...bundle.keys()].filter(path => /\.(xml|mjcf|urdf)$/i.test(path));
  let entry = options.entry ? cleanPath(options.entry) : undefined;
  if (!entry) {
    const roots = candidates.filter(path => {
      try { parseXML(decoder.decode(bundle.get(path))); return true; } catch { return false; }
    });
    if (roots.length !== 1) throw new Error("Choose the main MJCF or URDF entry file when importing multiple XML files.");
    entry = roots[0];
  }
  if (!bundle.has(entry)) throw new Error(`Entry file was not provided: ${entry}`);
  const entryDocument = parseXML(decoder.decode(bundle.get(entry)));
  const originalType = entryDocument.type;
  const rewritten = new Map(bundle);
  for (const path of candidates) {
    const text = decoder.decode(bundle.get(path));
    // Include fragments need not have a complete <mujoco> root.
    if (/<!DOCTYPE|<!ENTITY/i.test(text)) throw new Error("XML external entities are unsupported.");
    const valid = XMLValidator.validate(text);
    if (valid !== true) throw new Error(`Invalid XML asset ${path}: ${valid.err.msg}`);
    const tree = parser.parse(text);
    let meshdir = "", texturedir = "";
    walk(tree, (node, type) => {
      if (type !== "compiler") return;
      const a = attrs(node);
      meshdir = a["@_meshdir"] || a["@_assetdir"] || "";
      texturedir = a["@_texturedir"] || a["@_assetdir"] || "";
      delete a["@_meshdir"]; delete a["@_texturedir"]; delete a["@_assetdir"];
      if (originalType === "robot") { a["@_fusestatic"] = "false"; a["@_discardvisual"] = "false"; }
    });
    walk(tree, (node, type) => {
      const a = node[":@"] || {};
      for (const key of ["@_file", "@_filename"]) {
        if (!a[key]) continue;
        const prefix = type === "mesh" ? meshdir : type === "texture" ? texturedir : "";
        a[key] = resolveAsset(a[key], path, bundle, prefix ? prefix.replace(/\/?$/, "/") : "");
      }
    });
    if (path === entry && originalType === "robot") {
      const robot = tree.find(node => typeOf(node) === "robot");
      let extension = children(robot).find(node => typeOf(node) === "mujoco");
      if (!extension) { extension = element("mujoco"); children(robot).push(extension); }
      let compiler = children(extension).find(node => typeOf(node) === "compiler");
      if (!compiler) { compiler = element("compiler"); children(extension).push(compiler); }
      Object.assign(attrs(compiler), {"@_fusestatic": "false", "@_discardvisual": "false"});
    }
    rewritten.set(path, encoder.encode(builder.build(tree)));
  }
  let model, vfs = makeVFS(module, [...rewritten].map(([path, data]) => ({path, data})));
  let canonical, authoredHasPlane = false, authoredActuators = 0, jointTypes = [];
  try {
    // Loading by path lets the native URDF reader handle inertials and joints.
    model = module.MjModel.from_xml_path(entry, vfs);
    limits(model);
    authoredHasPlane = model.geom_type.some(type => type === module.mjtGeom.mjGEOM_PLANE.value);
    authoredActuators = model.nu;
    jointTypes = Array.from(model.jnt_type);
    canonical = parseXML(savedXML(module, model));
  } catch (error) {
    throw new Error(`MuJoCo import failed: ${error.message || String(error)}`);
  } finally { model?.delete(); vfs.delete(); }
  const root = canonical.root, contents = children(root);
  const world = contents.find(node => typeOf(node) === "worldbody");
  if (!world) throw new Error("The model does not contain a world body.");
  let jointNumber = 0, bodyNumber = 0, jointIndex = 0;
  const existingNames = new Set();
  walk(children(world), node => { if (node[":@"]?.["@_name"]) existingNames.add(node[":@"]["@_name"]); });
  const unusedName = stem => {
    let value = stem;
    while (existingNames.has(value)) value += "_";
    existingNames.add(value);
    return value;
  };
  walk(children(world), (node, type) => {
    if (type === "body" && !attrs(node)["@_name"]) attrs(node)["@_name"] = unusedName(`rlsim_body_${bodyNumber++}`);
    if (type === "joint") {
      if (!attrs(node)["@_name"]) attrs(node)["@_name"] = unusedName(`rlsim_joint_${jointNumber++}`);
      // Resolve default-class joint types before generating automatic motors.
      attrs(node)["@_type"] = ["free", "ball", "slide", "hinge"][jointTypes[jointIndex++]];
    } else if (type === "freejoint") jointIndex++;
  });
  if (options.floating_base) {
    const base = children(world).find(node => typeOf(node) === "body");
    if (!base) throw new Error("This model has no body to float.");
    if (children(base).some(node => ["joint", "freejoint"].includes(typeOf(node)))) throw new Error("The root already has joints. Author its floating joint directly in MJCF.");
    children(base).unshift(element("freejoint", {name: "rlsim_floating_base"}));
  }
  if (options.add_floor !== false && !authoredHasPlane) {
    children(world).unshift(element("geom", {
      name: "rlsim_floor", type: "plane", size: "200 200 .1", rgba: ".25 .27 .3 1", priority: 1, condim: 3,
    }));
  }
  let actuator = contents.find(node => typeOf(node) === "actuator");
  if (options.auto_actuators !== false && authoredActuators === 0) {
    const strength = options.motor_strength ?? 60;
    if (!Number.isFinite(strength) || strength <= 0 || strength > 10000) throw new Error("Motor strength must be positive and at most 10,000.");
    if (!actuator) { actuator = element("actuator"); contents.push(actuator); }
    walk(children(world), (node, type) => {
      if (type !== "joint") return;
      const a = attrs(node), jointType = a["@_type"] || "hinge";
      if (!["hinge", "slide", "ball"].includes(jointType)) return;
      const count = jointType === "ball" ? 3 : 1;
      for (let axis = 0; axis < count; axis++) {
        const gear = count === 1 ? String(strength) : [0, 1, 2, 3, 4, 5].map(i => i === axis ? strength : 0).join(" ");
        children(actuator).push(element("motor", {
          name: `rlsim_motor_${a["@_name"]}_${axis}`, joint: a["@_name"], gear,
          ctrllimited: "true", ctrlrange: "-1 1",
        }));
      }
    });
  }
  const xml = builder.build(canonical.nodes);
  const assets = [...rewritten].filter(([path]) => path !== entry).map(([path, data]) => ({path, data: bytesToBase64(data)}));
  const result = validateImportedBody({
    kind: "mjcf", name: options.name || entry.split("/").at(-1).replace(/\.(xml|mjcf|urdf)$/i, ""),
    xml, assets, ...(options.root_body ? {root_body: options.root_body} : {}),
    metadata: {source_format: originalType === "robot" ? "urdf" : "mjcf", entry},
  });
  const verified = compile(module, result.xml, result.assets);
  try {
    if (options.root_body && module.mj_name2id(verified, module.mjtObj.mjOBJ_BODY.value, options.root_body) < 1) throw new Error(`Tracked body "${options.root_body}" does not exist.`);
    result.metadata = {...result.metadata, bodies: verified.nbody - 1, joints: verified.njnt, actuators: verified.nu};
  } finally { verified.delete(); }
  return result;
}

function randomSource(seed) {
  if (!Number.isInteger(seed) || seed < 0 || seed > 0xffffffff) throw new Error("Seed must be a nonnegative 32-bit integer.");
  let value = seed >>> 0;
  const next = () => {
    value = (value + 0x6d2b79f5) >>> 0;
    let t = Math.imul(value ^ value >>> 15, value | 1);
    t ^= t + Math.imul(t ^ t >>> 7, t | 61);
    return ((t ^ t >>> 14) >>> 0) / 4294967296;
  };
  next.getState=()=>value;
  next.setState=s=>{if(!Number.isInteger(s)||s<0||s>0xffffffff)throw new Error("Invalid environment random state.");value=s>>>0;};
  return next;
}

export class ImportedWalker {
  constructor(module, body, environment = {}, options = {}) {
    this.mj = module; this.body = validateImportedBody(body);
    this.environment = validateEnvironment(environment);
    this.task = validateTask(options.task);
    if(this.task.preset==="walk")throw new Error("Coordinated walking currently supports template bodies only; choose Velocity for imported rigs.");
    this.target_speed = this.requested_target_speed = options.target_speed ?? 1;
    if (!Number.isFinite(this.target_speed) || this.target_speed < .1 || this.target_speed > 5) throw new Error("Target speed must be between 0.1 and 5.");
    this.randomize = Boolean(options.randomize); this.random = randomSource(options.seed ?? 7);
    this.observationSchema = "rlsim-general-v1";
    this._closed = false; this._meshPrefix = `model-${++modelSequence}`;
    try {
      this.model = compile(module, this.body.xml, this.body.assets);
      this.data = new module.MjData(this.model);
      this._velocityBuffer = new module.DoubleBuffer(6);
      const tracked = this.task.tracked_body || this.body.root_body;
      this._root = tracked ? module.mj_name2id(this.model, module.mjtObj.mjOBJ_BODY.value, tracked) : -1;
      if (tracked && this._root < 1) throw new Error(`Tracked body "${tracked}" does not exist.`);
      if (this._root < 1) {
        for (let i = 1; i < this.model.nbody; i++) if (this.model.body_dofnum[i] > 0) { this._root = i; break; }
        if (this._root < 1) this._root = this.model.nbody > 1 ? 1 : 0;
      }
      this.actionSize = this.model.nu;
      this.observationSize = this.model.nq + this.model.nv + 10 + this.actionSize;
      this.actionNames = actuatorControlNames(this.model,module);
      this.actionSchema="rlsim-actuator-controls-v1";
      walk(parseXML(this.body.xml).nodes,(node,type)=>{
        if(type==="text"&&node[":@"]?.["@_name"]==="rlsim_action_schema"){
          if(node[":@"]["@_data"]!=="rlsim-joint-target-v1")throw new Error("Unsupported joint control schema.");
          this.actionSchema=node[":@"]["@_data"];
        }
      });
      const positions = Array.from({length: this.model.nq}, (_, i) => `qpos[${i}]`);
      const velocities = Array.from({length: this.model.nv}, (_, i) => `qvel[${i}]`);
      for (let i = 0; i < this.model.njnt; i++) {
        const name = module.mj_id2name(this.model, module.mjtObj.mjOBJ_JOINT.value, i) || `joint_${i}`;
        const qstart = this.model.jnt_qposadr[i], vstart = this.model.jnt_dofadr[i];
        const qend = i + 1 < this.model.njnt ? this.model.jnt_qposadr[i + 1] : this.model.nq;
        const vend = i + 1 < this.model.njnt ? this.model.jnt_dofadr[i + 1] : this.model.nv;
        for (let j = qstart; j < qend; j++) positions[j] = `position:${name}[${j - qstart}]`;
        for (let j = vstart; j < vend; j++) velocities[j] = `velocity:${name}[${j - vstart}]`;
      }
      this.observationNames = [...positions, ...velocities, "target_speed",
        "target_delta.x", "target_delta.y", "target_delta.z", "body_up.x", "body_up.y", "body_up.z",
        "body_velocity.x", "body_velocity.y", "body_velocity.z", ...this.actionNames.map(name => `previous_action:${name}`)];
      this.bodyNames = Array.from({length: this.model.nbody}, (_, i) => module.mj_id2name(this.model, module.mjtObj.mjOBJ_BODY.value, i) || `body_${i}`);
      this.trackedBody = this.bodyNames[this._root];
      const option = this.model.opt;
      option.gravity.set([0, 0, -this.environment.gravity]);
      const friction = this.model.geom_friction;
      for (let i = 0; i < this.model.ngeom; i++) {
        if (this.model.geom_type[i] === module.mjtGeom.mjGEOM_PLANE.value) friction[i * 3] = this.environment.friction;
      }
      this._base = {
        body_mass: this.model.body_mass.slice(), body_inertia: this.model.body_inertia.slice(),
        actuator_gear: this.model.actuator_gear.slice(), geom_friction: friction.slice(),
        ...(this.actionSchema==="rlsim-joint-target-v1"?{
          actuator_gainprm:this.model.actuator_gainprm.slice(),actuator_biasprm:this.model.actuator_biasprm.slice(),
          actuator_forcerange:this.model.actuator_forcerange.slice(),
        }:{}),
      };
      this._previousAction = new Float64Array(this.actionSize);
      const timestep = option.timestep;
      option.delete();
      if (!Number.isFinite(timestep) || timestep <= 0 || timestep > .1) throw new Error("Physics timestep must be positive and at most 0.1 seconds.");
      this.frame_skip = Math.max(1, Math.round(.02 / timestep));
      this.dt = this.frame_skip * timestep;
      this.max_episode_steps = Math.max(1, Math.ceil((this.task.episode_seconds || 20) / this.dt));
      this._meshes = this._meshData();
      this.reset();
    } catch (error) { this.close(); throw error; }
  }
  _open() { if (this._closed) throw new Error("Imported physics environment is closed."); }
  setTrainingProgress(progress) {
    this._open();
    if (!Number.isFinite(progress)) throw new Error("Training progress must be finite.");
    this.target_speed = this.requested_target_speed * (.25 + .75 * clip(progress, 0, 1));
    return this._observation();
  }
  setLearningStage(stage) {
    this._requestedTask??=this.task;
    this.task={...this._requestedTask,preset:stage.preset};
    this.target_speed=this.requested_target_speed*stage.speed_fraction;
    this.randomization_strength=stage.randomization;
  }
  _position() { return Array.from(this.data.xpos.subarray(this._root * 3, this._root * 3 + 3)); }
  _up() {
    const matrix = this.data.xmat, start = this._root * 9;
    return [matrix[start + 2], matrix[start + 5], matrix[start + 8]];
  }
  _targetDelta() {
    const target = this.task.target || this.task.target_position || [2, 0, 1];
    return this._position().map((value, i) => Number(target[i] ?? 0) - value);
  }
  _observation() {
    const obs = new Float32Array(this.observationSize);
    let offset = 0;
    for (const values of [this.data.qpos, this.data.qvel, [this.target_speed], this._targetDelta(), this._up(), this._velocity, this._previousAction]) {
      obs.set(values, offset); offset += values.length;
    }
    return obs;
  }
  reset(seed) {
    this._open(); if (seed !== undefined) this.random = randomSource(seed);
    const scales = this.randomize ? [.2,.2,.4].map(range=>1+(2*this.random()-1)*range*(this.randomization_strength??1)) : [1, 1, 1];
    const servo=this.actionSchema==="rlsim-joint-target-v1";
    for (const [key, scale] of [["body_mass", scales[0]], ["body_inertia", scales[0]], ["actuator_gear", servo?1:scales[1]], ["geom_friction", scales[2]],
      ...(servo?[["actuator_gainprm",scales[1]],["actuator_biasprm",scales[1]],["actuator_forcerange",scales[1]]]:[])]) {
      const target = this.model[key], base = this._base[key];
      for (let i = 0; i < target.length; i++) target[i] = base[i] * scale;
    }
    this._scales = scales;
    this.mj.mj_setConst(this.model, this.data);
    if (this.model.nkey) this.mj.mj_resetDataKeyframe(this.model, this.data, 0);
    else this.mj.mj_resetData(this.model, this.data);
    const qpos = this.data.qpos, qvel = this.data.qvel;
    // Perturb scalar joints only; authored free/ball quaternions stay valid.
    for (let i = 0; i < this.model.njnt; i++) {
      const type = this.model.jnt_type[i], at = this.model.jnt_qposadr[i];
      if (type === this.mj.mjtJoint.mjJNT_HINGE.value || type === this.mj.mjtJoint.mjJNT_SLIDE.value) {
        qpos[at] += (this.random() - .5) * .01;
        if (this.model.jnt_limited[i]) qpos[at] = clip(qpos[at], this.model.jnt_range[2 * i], this.model.jnt_range[2 * i + 1]);
      }
    }
    for (let i = 0; i < qvel.length; i++) qvel[i] = (this.random() - .5) * .01;
    this.mj.mj_forward(this.model, this.data);
    this._previousAction.fill(0); this._steps = 0; this._fallen = false;
    this.mj.mj_objectVelocity(this.model, this.data, this.mj.mjtObj.mjOBJ_BODY.value, this._root, this._velocityBuffer, 0);
    this._velocity = Array.from(this._velocityBuffer.GetView().subarray(3));
    this._initialPosition = this._position();
    this._initialHeight = this._initialPosition[2];
    this._previousDistance = Math.hypot(...this._targetDelta()); this._geometryPending = true;
    this.last_info = this._info();
    return this._observation();
  }
  _info() {
    const position = this._position();
    return {
      position: position[0], position_3d: position, velocity: this._velocity[0], velocity_3d: [...this._velocity],
      height: position[2], upright: this._up()[2], target_speed: this.target_speed,
      fallen: this._fallen, step: this._steps, tracked_body: this.trackedBody,
      mass_scale: this._scales[0], motor_scale: this._scales[1], friction_scale: this._scales[2],
      observation_schema: this.observationSchema,
    };
  }
  step(action) {
    this._open();
    if (!action || action.length !== this.actionSize || !Array.from(action).every(Number.isFinite)) throw new Error(`Expected ${this.actionSize} finite actions.`);
    const normalized = Float64Array.from(action, value => clip(value, -1, 1));
    const ctrl = this.data.ctrl, ranges = this.model.actuator_ctrlrange;
    for (let i = 0; i < this.actionSize; i++) {
      const low=ranges[i*2],high=ranges[i*2+1];
      if(this.actionSchema==="rlsim-joint-target-v1"){
        // Zero holds the reference orientation (or nearest allowed hinge angle).
        // Positive/negative commands span their respective side of that origin.
        const center=clip(0,low,high);
        ctrl[i]=center+normalized[i]*(normalized[i]>=0?high-center:center-low);
      }else ctrl[i] = this.model.actuator_ctrllimited[i] ? low + (normalized[i] + 1) * .5 * (high-low) : normalized[i];
    }
    const before = this._position();
    for (let i = 0; i < this.frame_skip; i++) this.mj.mj_step(this.model, this.data);
    this.mj.mj_forward(this.model, this.data); this._steps++;
    const position = this._position();
    this._velocity = position.map((value, i) => (value - before[i]) / this.dt);
    let energy = 0, smoothness = 0;
    for (let i = 0; i < normalized.length; i++) {
      energy += normalized[i] ** 2; smoothness += (normalized[i] - this._previousAction[i]) ** 2;
    }
    energy /= Math.max(1, this.actionSize); smoothness /= Math.max(1, this.actionSize);
    const finite = this.data.qpos.every(Number.isFinite) && this.data.qvel.every(Number.isFinite);
    const result = evaluateTask(this.task, {
      velocity: this._velocity, position, upright: this._up()[2], height: position[2],
      initial_height: this._initialHeight, target_speed: this.target_speed, energy, smoothness,
      elapsed: this.data.time, previous_distance: this._previousDistance, dt: this.dt,
    });
    this._fallen = !finite || Boolean(result.fallen);
    this._previousDistance = Math.hypot(...this._targetDelta());
    this._previousAction.set(normalized);
    const reward = finite && Number.isFinite(result.reward) ? result.reward : -1;
    this.last_info = {...this._info(), ...result, fallen: this._fallen, reward, reward_components: result.components,
      energy_cost: energy, smoothness_cost: smoothness, speed_error: this._velocity[0] - this.target_speed};
    return {obs: this._observation(), reward, terminated: !finite || Boolean(result.terminated),
      truncated: this._steps >= this.max_episode_steps, info: {...this.last_info}};
  }
  _meshData() {
    const meshes = {};
    for (let i = 0; i < this.model.nmesh; i++) {
      const vertex = this.model.mesh_vertadr[i], count = this.model.mesh_vertnum[i];
      const face = this.model.mesh_faceadr[i], faces = this.model.mesh_facenum[i];
      meshes[`${this._meshPrefix}-${i}`] = {
        vertices: Array.from(this.model.mesh_vert.subarray(vertex * 3, (vertex + count) * 3)),
        faces: Array.from(this.model.mesh_face.subarray(face * 3, (face + faces) * 3)),
      };
    }
    for (let i = 0; i < this.model.nhfield; i++) {
      const rows = this.model.hfield_nrow[i], columns = this.model.hfield_ncol[i];
      const start = this.model.hfield_adr[i], size = this.model.hfield_size.subarray(i * 4, i * 4 + 4);
      const vertices = [], faces = [];
      for (let row = 0; row < rows; row++) for (let col = 0; col < columns; col++) {
        vertices.push((2 * col / (columns - 1) - 1) * size[0],
          (2 * row / (rows - 1) - 1) * size[1], this.model.hfield_data[start + row * columns + col] * size[2]);
        if (row < rows - 1 && col < columns - 1) {
          const at = row * columns + col;
          faces.push(at, at + 1, at + columns, at + 1, at + columns + 1, at + columns);
        }
      }
      meshes[`${this._meshPrefix}-hfield-${i}`] = {vertices, faces};
    }
    return meshes;
  }
  getFrame() {
    this._open();
    const geomNames = ["plane", "hfield", "sphere", "capsule", "ellipsoid", "cylinder", "box", "mesh", "sdf"];
    const geoms = [];
    for (let i = 0; i < this.model.ngeom; i++) {
      const type = geomNames[this.model.geom_type[i]];
      if (!type) continue;
      const mesh = this.model.geom_dataid[i];
      geoms.push({
        type, body: this.bodyNames[this.model.geom_bodyid[i]], group: this.model.geom_group[i],
        collision: Boolean(this.model.geom_contype[i] || this.model.geom_conaffinity[i]),
        pos: Array.from(this.data.geom_xpos.subarray(i * 3, i * 3 + 3)),
        mat: Array.from(this.data.geom_xmat.subarray(i * 9, i * 9 + 9)),
        size: Array.from(this.model.geom_size.subarray(i * 3, i * 3 + 3)),
        rgba: Array.from(this.model.geom_rgba.subarray(i * 4, i * 4 + 4)),
        ...(type === "mesh" ? {mesh: `${this._meshPrefix}-${mesh}`} : {}),
        ...(type === "hfield" ? {mesh: `${this._meshPrefix}-hfield-${mesh}`} : {}),
      });
    }
    const position = this._position();
    const frame = {
      time: this.data.time, position: position[0], velocity: this._velocity[0],
      torso_position: position, target_speed: this.target_speed, fallen: this._fallen,
      geoms, bodies: this.bodyNames.map((name, i) => ({
        name, pos: Array.from(this.data.xpos.subarray(i * 3, i * 3 + 3)),
        quat: Array.from(this.data.xquat.subarray(i * 4, i * 4 + 4)),
      })),
      task: this.task, observation_schema: this.observationSchema,
    };
    if (this._geometryPending) { frame.meshes = this._meshes; this._geometryPending = false; }
    return frame;
  }
  close() {
    if (this._closed) return;
    this._closed = true; this._velocityBuffer?.delete(); this.data?.delete(); this.model?.delete();
    this.data = null; this.model = null;
  }
}
