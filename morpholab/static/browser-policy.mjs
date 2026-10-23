// Browser PPO. TensorFlow.js and its WebGPU backend are loaded by the worker.
// Actions remain unbounded here: the simulator clips a copy, while PPO stores
// the original Gaussian sample so its log probability stays mathematically valid.

const LOG_2PI = Math.log(2 * Math.PI);
let policyCounter = 0;

function tensorflow() {
  if (!globalThis.tf) throw new Error("TensorFlow.js is not loaded.");
  return globalThis.tf;
}

function finiteVector(values, size, name) {
  if (!values || values.length !== size || !Array.from(values).every(Number.isFinite)) {
    throw new Error(`${name} must contain ${size} finite numbers.`);
  }
}

function integer(value, fallback, low, high) {
  const selected = value ?? fallback;
  if (!Number.isInteger(selected) || selected < low || selected > high) {
    throw new Error(`Expected an integer between ${low} and ${high}.`);
  }
  return selected;
}

function positive(value, fallback, name, allowZero = false) {
  const selected = value ?? fallback;
  if (!Number.isFinite(selected) || (allowZero ? selected < 0 : selected <= 0)) {
    throw new Error(`Invalid ${name}.`);
  }
  return selected;
}

/** Select a working training backend, including a real backward/Adam pass. */
export async function selectBackend() {
  const tf = tensorflow();
  let fallbackReason = "";
  let gpuName = "";
  try {
    if (!globalThis.navigator?.gpu) {
      throw new Error(globalThis.isSecureContext === false
        ? "WebGPU requires HTTPS or localhost."
        : "WebGPU is unavailable in this browser.");
    }
    const adapter = await navigator.gpu.requestAdapter({powerPreference: "high-performance"});
    if (!adapter) throw new Error("No WebGPU adapter is available.");
    const info = adapter.info || (adapter.requestAdapterInfo ? await adapter.requestAdapterInfo() : {});
    if (adapter.isFallbackAdapter || info.isFallbackAdapter) {
      throw new Error("Only a software WebGPU adapter is available.");
    }
    gpuName = info.description || [info.vendor, info.architecture].filter(Boolean).join(" ") || "WebGPU";
    if (/swiftshader|llvmpipe|software rasterizer/i.test(gpuName)) {
      throw new Error("Only a software WebGPU adapter is available.");
    }
    // Verify a complete update against the CPU reference before trusting GPU
    // kernels. Successful compilation or finite losses alone are insufficient.
    await tf.setBackend("cpu");
    await tf.ready();
    const reference = await probeTraining();
    // The default backend silently forwards small operations to CPU. Disable
    // that shortcut so this probe actually verifies GPU training kernels.
    tf.env().set("WEBGPU_CPU_FORWARD", false);
    tf.env().set("WEBGPU_USE_LOW_POWER_GPU", false);
    if (!await tf.setBackend("webgpu")) throw new Error("The WebGPU backend could not start.");
    await tf.ready();
    const candidate = await probeTraining();
    const maximumError = Math.max(...candidate.map((value, i) => Math.abs(value - reference[i])));
    if (!Number.isFinite(maximumError) || maximumError > 2e-5) {
      throw new Error("WebGPU training failed its CPU reference check.");
    }
    return {training_device: "webgpu", gpu_name: gpuName, execution_location: "browser"};
  } catch (error) {
    fallbackReason = error?.message || String(error);
  }
  if (!await tf.setBackend("cpu")) throw new Error("The CPU training backend could not start.");
  await tf.ready();
  await probeTraining();
  return {
    training_device: "cpu",
    gpu_name: null,
    fallback_reason: fallbackReason,
    execution_location: "browser",
  };
}

async function probeTraining() {
  const policy = new PPO(3, 2, {hidden_size: 64, epochs: 1, batch_size: 128, seed: 113});
  try {
    const observations = Array.from({length: 128}, (_, i) => [Math.sin(i), Math.cos(i), i / 128]);
    const before = await policy.act(observations, true);
    const sampled = await policy.act(observations);
    const metrics = await policy.update({
      observations,
      actions: sampled.actions,
      oldLogProbs: sampled.logProbs,
      advantages: observations.map(value => value[0]),
      returns: observations.map(value => value[1]),
    });
    const after = await policy.act(observations, true);
    if (!Number.isFinite(metrics.loss) || before.values.every((value, i) => value === after.values[i])) {
      throw new Error("The selected backend failed its training check.");
    }
    return Object.values((await policy.serialize()).weights).flatMap(weight => weight.values);
  } finally {
    policy.dispose();
  }
}

export class RunningNormalizer {
  constructor(size, clip = 10) {
    this.size = integer(size, undefined, 1, 8192);
    this.clip = positive(clip, 10, "normalization clipping");
    // A tiny unit-variance prior prevents division by zero on first observation.
    this.count = 1e-4;
    this.mean = new Array(size).fill(0);
    this.m2 = new Array(size).fill(1e-4);
  }

  update(observations) {
    const batch = typeof observations?.[0] === "number" ? [observations] : observations;
    if (!Array.isArray(batch)) throw new Error("Expected observations.");
    for (const observation of batch) {
      finiteVector(observation, this.size, "Observation");
      this.count++;
      for (let i = 0; i < this.size; i++) {
        const delta = observation[i] - this.mean[i];
        this.mean[i] += delta / this.count;
        this.m2[i] += delta * (observation[i] - this.mean[i]);
      }
    }
    return this;
  }

  normalize(observations) {
    const single = typeof observations?.[0] === "number";
    const batch = single ? [observations] : observations;
    if (!Array.isArray(batch)) throw new Error("Expected observations.");
    const result = batch.map(observation => {
      finiteVector(observation, this.size, "Observation");
      return Array.from(observation, (value, i) => Math.max(-this.clip, Math.min(this.clip,
        (value - this.mean[i]) / Math.sqrt(this.m2[i] / this.count + 1e-8))));
    });
    return single ? result[0] : result;
  }

  serialize() {
    return {size: this.size, clip: this.clip, count: this.count, mean: [...this.mean], m2: [...this.m2]};
  }

  static restore(data) {
    if (!data || !Number.isFinite(data.count) || data.count <= 0) {
      throw new Error("Invalid observation normalization.");
    }
    const result = new RunningNormalizer(data.size, data.clip);
    finiteVector(data.mean, data.size, "Normalization mean");
    finiteVector(data.m2, data.size, "Normalization variance");
    if (data.m2.some(value => value < 0)) throw new Error("Invalid normalization variance.");
    result.count = data.count;
    result.mean = [...data.mean];
    result.m2 = [...data.m2];
    return result;
  }
}

export class PPO {
  constructor(obsSize, actionSize, config = {}) {
    const tf = tensorflow();
    this.obsSize = integer(obsSize, undefined, 1, 8192);
    this.actionSize = integer(actionSize, undefined, 1, 256);
    this.config = {
      seed: integer(config.seed, 1, 0, 2147483647),
      hidden_size: integer(config.hidden_size, 64, 4, 256),
      hidden_layers: integer(config.hidden_layers, 2, 1, 4),
      epochs: integer(config.epochs, 4, 1, 32),
      batch_size: integer(config.batch_size, 128, 2, 4096),
      learning_rate: positive(config.learning_rate, 3e-4, "learning rate"),
      clip_range: positive(config.clip_range, 0.2, "PPO clipping"),
      entropy_coef: positive(config.entropy_coef, 0.005, "entropy coefficient", true),
      value_coef: positive(config.value_coef, 0.5, "value coefficient", true),
      max_grad_norm: positive(config.max_grad_norm, 0.5, "gradient clipping"),
      target_kl: positive(config.target_kl, 0.03, "target KL", true),
    };
    this.randomState = this.config.seed >>> 0;
    this.disposed = false;
    this.weights = {};
    this.name = `browser_ppo_${++policyCounter}`;
    let weightIndex = 0;
    const makeWeight = (name, shape, std = null, initial = 0) => {
      this.weights[name] = tf.tidy(() => tf.variable(
        std === null ? tf.fill(shape, initial) : tf.randomNormal(shape, 0, std, "float32", this.config.seed + ++weightIndex),
        true, `${this.name}_${name}`));
    };
    try {
      const width = this.config.hidden_size;
      for (const head of ["actor", "critic"]) {
        for (let layer = 1; layer <= this.config.hidden_layers; layer++) {
          const inputSize = layer === 1 ? obsSize : width;
          // Keep w1/b1 and w2/b2 names and initialization order unchanged so
          // existing two-layer browser checkpoints retain identical behavior.
          makeWeight(`${head}_w${layer}`, [inputSize, width], Math.sqrt(2 / inputSize));
          makeWeight(`${head}_b${layer}`, [width]);
        }
        makeWeight(`${head}_out_w`, [width, head === "actor" ? actionSize : 1],
          (head === "actor" ? 0.01 : 1) / Math.sqrt(width));
        makeWeight(`${head}_out_b`, [head === "actor" ? actionSize : 1]);
      }
      makeWeight("log_std", [actionSize], null, -0.7);
      this.variables = Object.values(this.weights);
      this.optimizer = tf.train.adam(this.config.learning_rate, 0.9, 0.999, 1e-5);
    } catch (error) {
      tf.dispose(Object.values(this.weights));
      throw error;
    }
  }

  random() {
    let t = this.randomState = (this.randomState + 0x6D2B79F5) >>> 0;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  }

  _checkLive() {
    if (this.disposed) throw new Error("This policy has been disposed.");
  }

  _head(observations, head) {
    const tf = tensorflow();
    const w = this.weights;
    let hidden = observations;
    for (let layer = 1; layer <= this.config.hidden_layers; layer++) {
      hidden = tf.tanh(tf.add(tf.matMul(hidden, w[`${head}_w${layer}`]), w[`${head}_b${layer}`]));
    }
    return tf.add(tf.matMul(hidden, w[`${head}_out_w`]), w[`${head}_out_b`]);
  }

  _forward(observations) {
    const tf = tensorflow();
    return {
      means: this._head(observations, "actor"),
      values: this._head(observations, "critic").reshape([-1]),
      logStd: tf.clipByValue(this.weights.log_std, -3, 1),
    };
  }

  async predict(batchObs) {
    // Playback needs only the deterministic actor. Avoid evaluating the critic
    // and two additional GPU readbacks on every physics control step.
    this._checkLive();
    if (!Array.isArray(batchObs) || !batchObs.length) throw new Error("Expected a nonempty observation batch.");
    batchObs.forEach(observation => finiteVector(observation, this.obsSize, "Observation"));
    const tf = tensorflow();
    const means = tf.tidy(() => this._head(tf.tensor2d(batchObs, [batchObs.length, this.obsSize]), "actor"));
    try {
      const actions = await means.array();
      if (!actions.flat().every(Number.isFinite)) throw new Error("Policy produced non-finite actions.");
      return actions;
    } finally {
      means.dispose();
    }
  }

  async act(batchObs, deterministic = false) {
    this._checkLive();
    if (!Array.isArray(batchObs) || !batchObs.length) throw new Error("Expected a nonempty observation batch.");
    batchObs.forEach(observation => finiteVector(observation, this.obsSize, "Observation"));
    const tf = tensorflow();
    const output = tf.tidy(() => this._forward(tf.tensor2d(batchObs, [batchObs.length, this.obsSize])));
    try {
      const [means, values, logStd] = await Promise.all([
        output.means.array(), output.values.array(), output.logStd.array(),
      ]);
      const logProbs = [];
      const actions = means.map(mean => {
        let logProbability = 0;
        const action = mean.map((mu, i) => {
          const noise = deterministic ? 0
            : Math.sqrt(-2 * Math.log(Math.max(1e-12, this.random()))) * Math.cos(2 * Math.PI * this.random());
          logProbability += -0.5 * (noise * noise + LOG_2PI) - logStd[i];
          return mu + Math.exp(logStd[i]) * noise;
        });
        logProbs.push(logProbability);
        return action;
      });
      if (![...actions.flat(), ...values, ...logProbs].every(Number.isFinite)) {
        throw new Error("Policy produced non-finite actions or values.");
      }
      return {actions, values, logProbs};
    } finally {
      tf.dispose(output);
    }
  }

  _loss(batch) {
    const tf = tensorflow();
    const {means, values, logStd} = this._forward(batch.observations);
    const standardized = batch.actions.sub(means).div(logStd.exp());
    const logProbs = standardized.square().add(LOG_2PI).mul(-0.5).sub(logStd).sum(1);
    const logRatio = tf.clipByValue(logProbs.sub(batch.oldLogProbs), -20, 20);
    const ratio = logRatio.exp();
    const unclipped = ratio.mul(batch.advantages);
    const clipped = tf.clipByValue(ratio, 1 - this.config.clip_range, 1 + this.config.clip_range).mul(batch.advantages);
    const policyLoss = tf.minimum(unclipped, clipped).mean().neg();
    const valueLoss = values.sub(batch.returns).square().mean();
    const entropy = logStd.add(0.5 * (1 + LOG_2PI)).sum();
    const loss = policyLoss.add(valueLoss.mul(this.config.value_coef)).sub(entropy.mul(this.config.entropy_coef));
    return {
      loss,
      policy_loss: policyLoss,
      value_loss: valueLoss,
      entropy,
      approx_kl: ratio.sub(1).sub(logRatio).mean(),
      clip_fraction: ratio.sub(1).abs().greater(this.config.clip_range).toFloat().mean(),
    };
  }

  async update({observations, actions, oldLogProbs, advantages, returns}) {
    this._checkLive();
    const count = observations?.length;
    if (!count || count < 2) throw new Error("PPO needs at least two rollout samples.");
    observations.forEach(observation => finiteVector(observation, this.obsSize, "Observation"));
    if (!Array.isArray(actions) || actions.length !== count) throw new Error("Invalid action batch.");
    actions.forEach(action => finiteVector(action, this.actionSize, "Action"));
    for (const [name, data] of Object.entries({oldLogProbs, advantages, returns})) finiteVector(data, count, name);
    const mean = advantages.reduce((sum, value) => sum + value, 0) / count;
    const scale = Math.sqrt(advantages.reduce((sum, value) => sum + (value - mean) ** 2, 0) / count + 1e-8);
    const tf = tensorflow();
    const batch = {
      observations: tf.tensor2d(observations, [count, this.obsSize]),
      actions: tf.tensor2d(actions, [count, this.actionSize]),
      oldLogProbs: tf.tensor1d(oldLogProbs),
      advantages: tf.tensor1d(Array.from(advantages, value => (value - mean) / scale)),
      returns: tf.tensor1d(returns),
    };
    try {
      const order = Array.from({length: count}, (_, i) => i);
      let updates = 0, epochs = 0, earlyStopped = false;
      epochLoop: for (let epoch = 0; epoch < this.config.epochs; epoch++) {
        for (let i = count - 1; i > 0; i--) {
          const j = Math.floor(this.random() * (i + 1));
          [order[i], order[j]] = [order[j], order[i]];
        }
        for (let start = 0; start < count; start += this.config.batch_size) {
          // Check before applying another gradient. PPO clipping alone does not
          // bound the total policy change over repeated minibatch updates.
          if (this.config.target_kl > 0) {
            const kl = tf.tidy(() => {
              const indices = tf.tensor1d(order.slice(start, start + this.config.batch_size), "int32");
              const miniBatch = Object.fromEntries(Object.entries(batch).map(([key, tensor]) => [key, tf.gather(tensor, indices)]));
              return this._loss(miniBatch).approx_kl;
            });
            let value;
            try { value = (await kl.data())[0]; } finally { kl.dispose(); }
            if (!Number.isFinite(value)) throw new Error("Non-finite PPO divergence; reduce the learning rate.");
            if (value > 1.5 * this.config.target_kl) { earlyStopped = true; break epochLoop; }
          }
          tf.tidy(() => {
            const indices = tf.tensor1d(order.slice(start, start + this.config.batch_size), "int32");
            const miniBatch = Object.fromEntries(Object.entries(batch).map(([key, tensor]) => [key, tf.gather(tensor, indices)]));
            const {grads} = tf.variableGrads(() => this._loss(miniBatch).loss, this.variables);
            // AddN would bind all network gradients at once, exceeding WebGPU's
            // per-stage storage-buffer limit on common adapters. Pairwise sums
            // keep each dispatch within that limit and preserve gradient clipping.
            const squaredNorms = Object.values(grads).map(gradient => gradient.square().sum());
            const norm = tf.sqrt(squaredNorms.reduce((sum, value) => sum.add(value)));
            const scale = tf.minimum(1, tf.scalar(this.config.max_grad_norm).div(norm.add(1e-8)));
            this.optimizer.applyGradients(Object.fromEntries(Object.entries(grads)
              .map(([name, gradient]) => [name, gradient.mul(scale)])));
          });
          updates++;
        }
        epochs++;
        // Let worker messages, including stop requests, run between epochs.
        await tf.nextFrame();
      }
      const metrics = tf.tidy(() => this._loss(batch));
      try {
        const result = Object.fromEntries(await Promise.all(Object.entries(metrics).map(async ([name, tensor]) => [name, (await tensor.data())[0]])));
        if (!Object.values(result).every(Number.isFinite)) throw new Error("Training diverged; reduce the learning rate.");
        return {...result, updates_completed: updates, epochs_completed: epochs, kl_early_stopped: Number(earlyStopped)};
      } finally {
        tf.dispose(metrics);
      }
    } finally {
      tf.dispose(batch);
    }
  }

  async serialize() {
    this._checkLive();
    const weights = Object.fromEntries(await Promise.all(Object.entries(this.weights).map(async ([name, tensor]) =>
      [name, {shape: [...tensor.shape], values: Array.from(await tensor.data())}])));
    return {
      format: "morpholab-browser-ppo-v1",
      obsSize: this.obsSize,
      actionSize: this.actionSize,
      config: {...this.config},
      randomState: this.randomState,
      weights,
    };
  }

  async trainingState() {
    const weights=await this.optimizer.getWeights();
    try {
      const entries=await Promise.all(weights.map(async ({tensor})=>({shape:[...tensor.shape],values:Array.from(await tensor.data())})));
      return {format:'rlsim-adam-v1',iterations:entries[0].values[0],slots:entries.slice(1)};
    } finally {weights[0].tensor.dispose();} // Adam owns the remaining slot variables.
  }

  static async continue(data, optimizer) {
    const policy=await PPO.restore(data),tf=tensorflow(),tensors=[];
    try {
      const n=policy.variables.length;
      if(optimizer?.format!=='rlsim-adam-v1' || !Number.isSafeInteger(optimizer.iterations) || optimizer.iterations<0 ||
        !Array.isArray(optimizer.slots) || optimizer.slots.length!==(optimizer.iterations?2*n:0))throw new Error('Invalid or missing Adam training state.');
      tensors.push({name:'iter',tensor:tf.scalar(optimizer.iterations,'int32')});
      for(let i=0;i<optimizer.slots.length;i++){
        const slot=optimizer.slots[i],variable=policy.variables[i%n];
        if(JSON.stringify(slot.shape)!==JSON.stringify(variable.shape))throw new Error('Optimizer slot shape does not match the policy.');
        finiteVector(slot.values,variable.size,'Optimizer state');
        if(i>=n&&slot.values.some(v=>v<0))throw new Error('Invalid Adam second moment.');
        tensors.push({name:`${variable.name}/${i<n?'m':'v'}`,tensor:tf.tensor(slot.values,slot.shape)});
      }
      await policy.optimizer.setWeights(tensors);
      return policy;
    } catch(error){policy.dispose();throw error;}
    finally{tf.dispose(tensors.map(item=>item.tensor));}
  }

  static async restore(data) {
    if (!data || data.format !== "morpholab-browser-ppo-v1" || !data.weights) {
      throw new Error("Not a supported browser PPO checkpoint.");
    }
    const tf = tensorflow();
    const policy = new PPO(data.obsSize, data.actionSize, data.config);
    try {
      for (const [name, variable] of Object.entries(policy.weights)) {
        const saved = data.weights[name];
        if (!saved || JSON.stringify(saved.shape) !== JSON.stringify(variable.shape)) {
          throw new Error(`Invalid checkpoint weight: ${name}.`);
        }
        finiteVector(saved.values, variable.size, `Checkpoint weight ${name}`);
        tf.tidy(() => variable.assign(tf.tensor(saved.values, variable.shape, "float32")));
      }
      if (Number.isInteger(data.randomState)) policy.randomState = data.randomState >>> 0;
      return policy;
    } catch (error) {
      policy.dispose();
      throw error;
    }
  }

  dispose() {
    if (this.disposed) return;
    this.disposed = true;
    this.optimizer.dispose();
    tensorflow().dispose(this.variables);
  }
}

/** Playback loads the actor only: no critic, exploration state or optimizer. */
export class InferenceActor {
  static async restore(data) {
    if (!data || data.format !== "morpholab-browser-ppo-v1" || !data.weights) throw new Error("Not a supported browser PPO checkpoint.");
    const actor = Object.create(InferenceActor.prototype);
    actor.obsSize = integer(data.obsSize, undefined, 1, 8192);
    actor.actionSize = integer(data.actionSize, undefined, 1, 256);
    const width = integer(data.config?.hidden_size, 64, 4, 256);
    actor.config = {hidden_layers: integer(data.config?.hidden_layers, 2, 1, 4)};
    actor.weights = {}; actor.disposed = false;
    const tf = tensorflow();
    const load = (name, shape) => {
      const saved = data.weights[name];
      if (!saved || JSON.stringify(saved.shape) !== JSON.stringify(shape)) throw new Error(`Invalid checkpoint weight: ${name}.`);
      finiteVector(saved.values, shape.reduce((a,b)=>a*b,1), `Checkpoint weight ${name}`);
      actor.weights[name] = tf.tensor(saved.values, shape, "float32");
    };
    try {
      for (let layer=1; layer<=actor.config.hidden_layers; layer++) {
        load(`actor_w${layer}`, [layer===1?actor.obsSize:width,width]);
        load(`actor_b${layer}`, [width]);
      }
      load("actor_out_w", [width,actor.actionSize]); load("actor_out_b", [actor.actionSize]);
      return actor;
    } catch (error) { actor.dispose(); throw error; }
  }
  _checkLive() { if (this.disposed) throw new Error("This policy has been disposed."); }
  _head(observations, head) { return PPO.prototype._head.call(this, observations, head); }
  predict(observations) { return PPO.prototype.predict.call(this, observations); }
  dispose() {
    if (this.disposed) return;
    this.disposed = true; tensorflow().dispose(Object.values(this.weights));
  }
}
