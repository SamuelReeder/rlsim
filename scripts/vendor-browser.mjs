import {copyFile, mkdir, readFile, writeFile} from "node:fs/promises";
import {createHash} from "node:crypto";
import {fileURLToPath} from "node:url";
import path from "node:path";
import {build} from "esbuild";

const root = fileURLToPath(new URL("../", import.meta.url));
const vendor = path.join(root, "morpholab/static/vendor");
const packageNames = ["@mujoco/mujoco", "@tensorflow/tfjs", "@tensorflow/tfjs-backend-webgpu",
  "three", "fflate", "fast-xml-parser", "esbuild", "onnxruntime-web"];
const assets = [
  ["@mujoco/mujoco/mujoco.js", "mujoco/mujoco.js"],
  ["@mujoco/mujoco/mujoco.wasm", "mujoco/mujoco.wasm"],
  ["@tensorflow/tfjs/dist/tf.min.js", "tf.min.js"],
  ["@tensorflow/tfjs-backend-webgpu/dist/tf-backend-webgpu.min.js", "tf-backend-webgpu.min.js"],
  ["onnxruntime-web/dist/ort.webgpu.min.js", "ort/ort.webgpu.min.js"],
  ["onnxruntime-web/dist/ort-wasm-simd-threaded.jsep.mjs", "ort/ort-wasm-simd-threaded.jsep.mjs"],
  ["onnxruntime-web/dist/ort-wasm-simd-threaded.jsep.wasm", "ort/ort-wasm-simd-threaded.jsep.wasm"],
  ["onnxruntime-web/dist/ort-wasm-simd-threaded.asyncify.mjs", "ort/ort-wasm-simd-threaded.asyncify.mjs"],
  ["onnxruntime-web/dist/ort-wasm-simd-threaded.asyncify.wasm", "ort/ort-wasm-simd-threaded.asyncify.wasm"],
];
const licenses = [
  ["mujoco/LICENSE", "https://raw.githubusercontent.com/google-deepmind/mujoco/3.16.0/LICENSE", "Apache License"],
  ["tensorflow-LICENSE", "https://raw.githubusercontent.com/tensorflow/tfjs/tfjs-v4.22.0/LICENSE", "Apache License"],
  ["onnxruntime-LICENSE", "https://raw.githubusercontent.com/microsoft/onnxruntime/v1.30.0/LICENSE", "MIT License"],
  ["onnxruntime-ThirdPartyNotices.txt", "https://raw.githubusercontent.com/microsoft/onnxruntime/v1.30.0/ThirdPartyNotices.txt", "Copyright"],
  ["nodable-entities-LICENSE", "https://raw.githubusercontent.com/nodable/val-parsers/main/LICENSE", "MIT License"],
];
const localLicenses = [
  ["three/LICENSE", "three-LICENSE"], ["fflate/LICENSE", "fflate-LICENSE"],
  ["fast-xml-parser/LICENSE", "fast-xml-parser-LICENSE"], ["esbuild/LICENSE.md", "esbuild-LICENSE"],
  ["fast-xml-builder/LICENSE", "fast-xml-builder-LICENSE"], ["is-unsafe/LICENSE", "is-unsafe-LICENSE"],
  ["path-expression-matcher/LICENSE", "path-expression-matcher-LICENSE"],
  ["strnum/LICENSE", "strnum-LICENSE"], ["xml-naming/LICENSE", "xml-naming-LICENSE"],
];

export async function vendorBrowser() {
  const project = JSON.parse(await readFile(path.join(root, "package.json"), "utf8"));
  const packages = Object.fromEntries(packageNames.map(name => [name, project.dependencies[name]]));
  for (const [name, version] of Object.entries(packages)) {
    if (!/^\d+\.\d+\.\d+$/.test(version || "")) throw new Error(`Pin ${name} to an exact version in package.json.`);
    let installed;
    try {
      installed = JSON.parse(await readFile(path.join(root, "node_modules", name, "package.json"), "utf8"));
    } catch {
      throw new Error(`Missing ${name}. Run npm ci first.`);
    }
    if (installed.version !== version) throw new Error(`${name}: expected ${version}, found ${installed.version}`);
  }
  await mkdir(path.join(vendor, "mujoco"), {recursive: true});
  await mkdir(path.join(vendor, "ort"), {recursive: true});
  const files = [];
  for (const [source, destination] of assets) {
    const output = path.join(vendor, destination);
    await copyFile(path.join(root, "node_modules", source), output);
    const contents = await readFile(output);
    files.push({file: destination, bytes: contents.length, sha256: createHash("sha256").update(contents).digest("hex")});
  }
  const common = {bundle: true, minify: true, platform: "browser", target: "es2022", legalComments: "inline",
    absWorkingDir: root, logLevel: "warning"};
  for (const [destination, configuration] of [
    ["scene.js", {entryPoints: ["morpholab/static/scene-runtime.mjs"], format: "iife", globalName: "RLsimScene"}],
    ["xml.mjs", {stdin: {contents: 'export {XMLParser, XMLBuilder, XMLValidator} from "fast-xml-parser";', resolveDir: root}, format: "esm"}],
    ["zip.mjs", {stdin: {contents: 'export * from "fflate";', resolveDir: root}, format: "esm"}],
  ]) {
    const output = path.join(vendor, destination);
    await build({...common, ...configuration, outfile: output});
    const contents = await readFile(output);
    files.push({file: destination, bytes: contents.length, sha256: createHash("sha256").update(contents).digest("hex")});
  }
  // npm distributions omit root LICENSE files. Cache each pinned release's
  // official Apache 2.0 text with the assets; subsequent builds work offline.
  for (const [destination, url, marker] of licenses) {
    const output = path.join(vendor, destination);
    let cached = "";
    try { cached = await readFile(output, "utf8"); } catch {}
    if (!cached.includes(marker)) {
      const response = await fetch(url, {signal: AbortSignal.timeout(30000)});
      if (!response.ok) throw new Error(`License download failed (${response.status}): ${url}`);
      const text = await response.text();
      if (!text.includes(marker)) {
        throw new Error(`Unexpected license content: ${url}`);
      }
      await writeFile(output, text);
    }
  }
  for (const [source, destination] of localLicenses) {
    await copyFile(path.join(root, "node_modules", source), path.join(vendor, destination));
  }
  await writeFile(path.join(vendor, "NOTICE.txt"), `Third-party runtime assets used by RLsim

MuJoCo ${packages["@mujoco/mujoco"]} WebAssembly bindings
Copyright Google DeepMind and the MuJoCo authors.
Source: https://github.com/google-deepmind/mujoco/tree/3.16.0/wasm
Package: https://www.npmjs.com/package/@mujoco/mujoco
License: Apache 2.0, included in mujoco/LICENSE.

TensorFlow.js ${packages["@tensorflow/tfjs"]} and WebGPU backend ${packages["@tensorflow/tfjs-backend-webgpu"]}
Copyright 2024 Google LLC. All Rights Reserved.
Source: https://github.com/tensorflow/tfjs/tree/tfjs-v4.22.0
Packages: @tensorflow/tfjs, @tensorflow/tfjs-backend-webgpu
License: Apache 2.0, included in tensorflow-LICENSE.

Three.js ${packages.three}, including the official glTF, FBX, OBJ, MTL, and STL loaders.
Source: https://github.com/mrdoob/three
License: MIT, included in three-LICENSE. FBX's bundled fflate notices are preserved.

fflate ${packages.fflate}; fast-xml-parser ${packages["fast-xml-parser"]}.
Sources: https://github.com/101arrowz/fflate and https://github.com/NaturalIntelligence/fast-xml-parser
Licenses: MIT, included in fflate-LICENSE and fast-xml-parser-LICENSE.
XML dependency license files: nodable-entities-LICENSE, fast-xml-builder-LICENSE,
is-unsafe-LICENSE, path-expression-matcher-LICENSE, strnum-LICENSE, xml-naming-LICENSE.

ONNX Runtime Web ${packages["onnxruntime-web"]}.
Source: https://github.com/microsoft/onnxruntime/tree/v${packages["onnxruntime-web"]}
License: MIT, included in onnxruntime-LICENSE and onnxruntime-ThirdPartyNotices.txt.

esbuild ${packages.esbuild} bundles the scene, XML, and ZIP modules for browsers.
Source: https://github.com/evanw/esbuild
License: MIT, included in esbuild-LICENSE.

MuJoCo, TensorFlow.js, and ONNX Runtime distributions are copied unchanged.
Scene, XML, and ZIP modules are bundled from pinned packages by esbuild.
Copyright and license notices embedded in bundles are preserved.
This notice identifies dependencies; it is not an endorsement by their authors.
`);
  const transitive = {};
  for (const name of ["@nodable/entities", "fast-xml-builder", "is-unsafe", "path-expression-matcher", "strnum", "xml-naming", "onnxruntime-common"]) {
    transitive[name] = JSON.parse(await readFile(path.join(root, "node_modules", name, "package.json"), "utf8")).version;
  }
  await writeFile(path.join(vendor, "manifest.json"), JSON.stringify({packages, transitive, files,
    licenses: [...licenses.map(([file, source]) => ({file, source})),
      ...localLicenses.map(([source, file]) => ({file, source: `node_modules/${source}`}))]}, null, 2) + "\n");
  console.log(`Prepared ${files.length} browser runtime assets in morpholab/static/vendor`);
}

if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  await vendorBrowser();
}
