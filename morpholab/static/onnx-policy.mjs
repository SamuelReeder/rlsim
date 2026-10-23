/** Standard ONNX controller inference on the visitor's device. */
const MAX_BYTES = 128 * 1024 * 1024;

function decodeBase64(text, label) {
  if (typeof text !== "string" || !text || text.length > MAX_BYTES * 1.4 || !/^[A-Za-z0-9+/]*={0,2}$/.test(text)) {
    throw new Error(`${label} must contain base64 data smaller than 128 MB.`);
  }
  try { return Uint8Array.from(atob(text), char => char.charCodeAt(0)); }
  catch { throw new Error(`${label} contains invalid base64 data.`); }
}
function dimension(value, max, label) {
  if (!Number.isInteger(value) || value < 1 || value > max) throw new Error(`Invalid ${label} dimension.`);
}
function externalFiles(files = []) {
  if (!Array.isArray(files) || files.length > 128) throw new Error("ONNX external_data must list at most 128 local files.");
  const names = new Set();
  return files.map(file => {
    if (typeof file.path !== "string" || !file.path || file.path.startsWith("/") ||
        /^[a-z][a-z0-9+.-]*:/i.test(file.path) || file.path.split(/[\\/]/).includes("..") ||
        /[\0\r\n]/.test(file.path) || names.has(file.path)) {
      throw new Error("ONNX external weights must have unique relative paths inside the imported package.");
    }
    names.add(file.path);
    return {path: file.path, data: decodeBase64(file.data, `ONNX weight ${file.path}`)};
  });
}
function inputShape(metadata, obsSize) {
  if (!metadata?.isTensor || metadata.type !== "float32") throw new Error("ONNX observation input must be a float32 tensor.");
  const shape = metadata.shape;
  if (!Array.isArray(shape) || (shape.length !== 1 && shape.length !== 2)) {
    throw new Error("ONNX observation input must have shape [observations] or [1, observations]. Recurrent/multiple-state inputs are unsupported.");
  }
  const feature = shape.at(-1);
  if (typeof feature === "number" && feature > 0 && feature !== obsSize) {
    throw new Error(`ONNX expects ${feature} observations; this body supplies ${obsSize}.`);
  }
  if (shape.length === 2 && typeof shape[0] === "number" && shape[0] > 0 && shape[0] !== 1) {
    throw new Error("ONNX inference requires a batch dimension of 1 (or a dynamic batch).");
  }
  return shape.length === 1 ? [obsSize] : [1, obsSize];
}

export class ONNXPolicy {
  constructor(session, ort, obsSize, actionSize, inputName, outputName, shape, provider, fallbackReason) {
    this.session = session; this.ort = ort; this.obsSize = obsSize; this.actionSize = actionSize;
    this.inputName = inputName; this.outputName = outputName; this.inputShape = shape;
    this.execution_provider = provider;
    this.inference_device = provider === "wasm" ? "cpu" : "webgpu";
    this.fallback_reason = fallbackReason || null;
    this.disposed = false;
  }

  static async restore(controller, obsSize, actionSize, hardware = {}) {
    if (!controller || controller.format !== "rlsim-onnx-v1") throw new Error("Select a supported ONNX controller package.");
    dimension(obsSize, 8192, "observation"); dimension(actionSize, 256, "action");
    const ort = globalThis.ort;
    if (!ort?.InferenceSession || !ort?.Tensor) throw new Error("ONNX Runtime is not loaded.");
    const data = decodeBase64(controller.data, "ONNX model");
    const externalData = externalFiles(controller.external_data);
    if (data.length + externalData.reduce((sum, item) => sum + item.data.length, 0) > MAX_BYTES) {
      throw new Error("ONNX model and external weights exceed 128 MB.");
    }
    ort.env.wasm.wasmPaths = new URL("./vendor/ort/", import.meta.url).href;
    ort.env.wasm.numThreads = 1;
    ort.env.wasm.proxy = false;
    const providers = hardware.training_device === "webgpu" ? ["webgpu", "wasm"] : ["wasm"];
    let fallbackReason = "";
    for (const provider of providers) {
      let session, policy;
      try {
        session = await ort.InferenceSession.create(data, {
          executionProviders: [provider], externalData,
          graphOptimizationLevel: "all", executionMode: "sequential",
        });
        if (session.inputNames.length !== 1) {
          throw new Error(`This ONNX controller has ${session.inputNames.length} inputs. Export one flat observation input; recurrent hidden-state inputs are unsupported.`);
        }
        const inputName = controller.input_name || session.inputNames[0];
        if (inputName !== session.inputNames[0]) throw new Error(`ONNX input "${inputName}" was not found.`);
        const outputName = controller.output_name || (session.outputNames.includes("actions") ? "actions"
          : session.outputNames.length === 1 ? session.outputNames[0] : null);
        if (!outputName) throw new Error("This model has multiple outputs. Set output_name to the action tensor in its controller manifest.");
        if (!session.outputNames.includes(outputName)) throw new Error(`ONNX output "${outputName}" was not found.`);
        const shape = inputShape(session.inputMetadata.find(item => item.name === inputName), obsSize);
        policy = new ONNXPolicy(session, ort, obsSize, actionSize, inputName, outputName, shape, provider, fallbackReason);
        await policy.act([new Array(obsSize).fill(0)], true);
        return policy;
      } catch (error) {
        if (policy) await policy.dispose();
        else if (session) await session.release();
        if (provider === "webgpu") {
          fallbackReason = `WebGPU ONNX inference failed: ${error.message || error}`;
        } else {
          throw new Error(`ONNX inference could not start: ${error.message || error}${fallbackReason ? ` (${fallbackReason})` : ""}`);
        }
      }
    }
    throw new Error("No ONNX inference device is available.");
  }

  async act(observations, deterministic = true) {
    if (this.disposed) throw new Error("This ONNX controller has been disposed.");
    if (!Array.isArray(observations) || !observations.length) throw new Error("Provide a batch of observations.");
    const actions = [];
    for (const observation of observations) {
      if (!observation || observation.length !== this.obsSize || !Array.from(observation).every(Number.isFinite)) {
        throw new Error(`ONNX input needs ${this.obsSize} finite observation values.`);
      }
      const input = new this.ort.Tensor("float32", Float32Array.from(observation), this.inputShape);
      let outputs;
      try {
        outputs = await this.session.run({[this.inputName]: input}, [this.outputName]);
        const output = outputs[this.outputName];
        if (!output || output.type !== "float32" || !(
          output.dims.length === 1 && output.dims[0] === this.actionSize ||
          output.dims.length === 2 && output.dims[0] === 1 && output.dims[1] === this.actionSize
        )) {
          throw new Error(`ONNX actions must be float32 with shape [${this.actionSize}] or [1, ${this.actionSize}].`);
        }
        const values = Array.from(await output.getData());
        if (values.length !== this.actionSize || !values.every(Number.isFinite)) throw new Error("ONNX controller produced nonfinite or incorrectly sized actions.");
        // The environment maps/clips these normalized actions to actuator ranges.
        actions.push(values);
      } finally {
        input.dispose();
        if (outputs) for (const tensor of Object.values(outputs)) tensor.dispose();
      }
    }
    return {actions, values: new Array(actions.length).fill(0)};
  }

  async dispose() {
    if (this.disposed) return;
    this.disposed = true;
    await this.session.release();
  }
}
