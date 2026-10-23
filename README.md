# RLsim

Edit a jointed body and its environment, train a PPO controller, and inspect
the resulting motion in a real MuJoCo physics simulation.

The execution selector chooses where the work happens:

| Mode | Training device | Physics | Saved models |
| --- | --- | --- | --- |
| **This device** | Visitor's WebGPU GPU, with CPU fallback | MuJoCo WebAssembly on the visitor's CPU | Browser storage (IndexedDB), JSON export/import |
| **Server** | Server's CUDA GPU, Apple MPS GPU, or CPU, in that order | Native MuJoCo on the server's CPU | Server `runs/` directory, checkpoint bundle download |

Both paths run a real forward/backward training operation before accepting a
device. The interface shows the selected device and explains any fallback.
**This device** is the default and works on a static deployment: visitors use
their own hardware. **Server** uses the machine running the Python service;
deploying that service does not give it access to a visitor's GPU.

## Static deployment

Install Node.js 20 or newer, then run:

```bash
npm ci
npm run build:static
```

Publish the contents of `dist/browser/` with any static web host. Serve over
**HTTPS** (or localhost during development), with `.wasm` served as
`application/wasm` and `.js`/`.mjs` as JavaScript. Relative asset paths support a
subdirectory such as `/rlsim/`. The deployed application needs no Node or
Python server, API keys, CDN, GPU rental, or account system for importing prepared
bodies, simulation, or training. Optional SkinTokens rig generation runs
separately in a configured local CUDA environment; its output can then be
imported into the static site.

The build includes pinned MuJoCo 3.16.0, TensorFlow.js 4.22.0, Three.js 0.186.1,
ONNX Runtime Web 1.30.0, and the XML/ZIP import helpers. It bundles official
Three.js loaders for conventional model files and copies inference runtimes
locally. Dependency license files, attribution notices, versions, and asset
hashes are under `vendor/`. The first build fetches missing license texts from
official upstream sources; subsequent builds use the cached copies.

WebGPU availability depends on the visitor's browser, graphics driver, and
hardware. If WebGPU is missing or its training operation fails, TensorFlow.js
uses the visitor's CPU. Physics always runs on the CPU in this version.
The single-threaded MuJoCo WASM build does not require cross-origin isolation.

Keep the tab open while training. Browsers can throttle background tabs or
suspend work when a device sleeps. Models are saved locally to the current
origin; clearing site data removes them. Export important models as JSON.
Browser checkpoints include policy weights, observation normalization, body,
environment, and training configuration. They are distinct from the native
Stable-Baselines3 checkpoint format; `rlsim export-browser` converts supported
native policies for browser inference.

## Local server

Once `.venv` is set up (see below for a fresh checkout), start the server:

```bash
./run.sh
```

Open [RLsim](http://localhost:8777). Choose **Server** to use native Python
training or access the existing local experiments.

For a fresh Linux/WSL or macOS checkout:

```bash
npm ci
npm run vendor:browser
./setup.sh
./run.sh
```

`setup.sh` installs the appropriate PyTorch build and verifies device
availability. It recognizes NVIDIA drivers on Linux/WSL; supported macOS
installations can use MPS. `./setup.sh cpu` requests a CPU installation and
`./setup.sh cuda` requests the pinned CUDA build. A working CPU is sufficient.
Stop the server with Ctrl+C.

Before building a Python wheel, run `npm run vendor:browser` so its package
includes the generated browser runtime files.

## Use it

1. Choose **This device** or **Server**.
2. Open **Body**, edit a 2-, 4-, or 6-legged template, or import a conventional
   character/robot model. Set gravity and floor friction.
3. Open **Train** and choose a step budget, target speed, seed, environment
   count, curriculum and randomization. Start training; the viewport samples
   actual simulator state.
4. **Stop and save** preserves the current model. An interrupted partial
   rollout does not receive a final PPO update.
5. Open **Model**, select a saved policy, set inference speed and seed, and run
   inference. **Use model settings** restores its body and environment.
6. Pause, seek, adjust playback speed, or switch camera views. Download the
   model and body XML; browser mode also imports JSON and ONNX controllers.

During training, **Fixed view** keeps the camera steady while the viewport shows
sampled poses from training episodes. Falls, exploration noise and episode resets
can still move the body abruptly. **Reset view** recentres the camera if a body
moves out of frame. During playback, **Follow** smoothly tracks horizontal motion
and holds the camera height steady so the floor does not bounce with the body.

Browser inference starts playback as simulation frames arrive. A small indicator
shows loading or buffering when generation has not caught up with playback.
Pause and seeking remain available for generated frames; completion preserves
your playback position and pause choice. Initial model loading and GPU setup can
still take time. Repeated inference of the same body reuses its loaded visual
assets, and deterministic PPO inference only evaluates the actor network.

A 4,096-step run checks the pipeline; it does not establish that a useful gait
has been learned. Locomotion may need hundreds of thousands or millions of
steps, and some bodies and seeds may fail. Ordinary runs train one body shape;
optional native proportion training samples a family with the same joint layout.
Changing the number of joints requires a new policy.

## Demo gallery, comparisons and continuation

Open **Explore → Demo gallery** and choose **Watch demo** in This device mode.
These are bundled pretrained controllers evaluated in live browser physics,
not prerecorded animations. A demo is added to your local model library and
can be replayed with another seed or target speed. The gallery identifies its
training steps and source. Planar templates are labelled explicitly. The early
quadruped example deliberately shows upright survival without useful walking;
its run is independent of the speed-only example, not an intermediate checkpoint
from the same training trajectory. Curated gallery examples are demonstrations,
not an unbiased estimate of training success.

**Explore → Experiment comparisons** presents a native fixed-budget curriculum
pilot: three body types, five training seeds, two conditions, 65,536 steps per
run. All planned runs are retained, including failures. Each final controller
is selected using training-time validation, then evaluated on three different
fixed test seeds. Tables show means and sample standard deviations across
training seeds, not an inflated sample count from individual test episodes.
A short budget can favor some methods and bodies; this is a pilot, not a claim
that one configuration is universally best. The JSON download contains the
protocol, full configurations, hardware, training times, checkpoint steps,
per-episode results and source hashes. Domain randomization is enabled during
training; final tests use nominal physics with seeded initial states.
The bundled development pilot spans three training-code revisions while
continuation was being implemented. It is an exploratory snapshot; reproduce
the suite at one frozen revision before using it as benchmark evidence.

Reproduce or extend the pilot (a completed directory is reused only with the
same protocol):

```bash
.venv/bin/python -m morpholab.experiments --output experiments/my-comparison \
  --bodies biped quadruped hexapod --seeds 7 17 27 37 47 --steps 65536
```

The gallery also includes a longer hexapod continuation run, separate from the
equal-budget pilot. Its card reports the selected checkpoint's step count;
the full run reached 327,680 steps. To repackage the included native sources
and recheck every demo in browser physics:

```bash
.venv/bin/python scripts/build-gallery.py
.venv/bin/python scripts/verify-gallery.py --url http://localhost:8777/
npm run build:static
```

The verifier records three live test episodes and simulator thumbnails. A
historical speed-only example must survive all three and average more than five
metres; this gate does not establish a coordinated gait. The newer coordinated
examples use the stricter checks documented below.

**Compare your models** evaluates 2–6 saved models in the selected execution
mode, using their original bodies, worlds, objectives and speeds. It selects
Best evaluated when available, otherwise Latest, and uses a shared suite of
three seeds distinct from those used for best-checkpoint selection. Differences
in bodies, tasks, budgets and related training seeds are flagged: descriptive
comparisons do not isolate the effect of a particular change. Download the
raw results or stop the comparison; incomplete runs remain labelled partial.
Repeatedly tuning against this suite makes it a validation suite, so use fresh
seeds for a final scientific claim. Speed error is meaningful for velocity
tracking; interpret other tasks using their objective and return.

**Model → Continue training** accepts an additional step budget and creates a
new run from the **latest** checkpoint. It preserves the original run, body,
objective, architecture and hyperparameters. It restores actor/critic weights,
Adam moments and iteration count, normalization statistics, policy and
simulation random-generator state, and staged-curriculum gate history. The
original gradual-speed schedule horizon is retained so continuation does not
reset the speed ramp. The progress display shows cumulative steps, and the
new checkpoint records its parent run.

Episodes restart on continuation: simulator poses, contact caches, incomplete
rollouts and partial-episode statistics are not restored. This continues
learning, but is not a bit-for-bit continuation of the same trajectories.
Changing device/runtime can also change numerical results. Use **Stop and save**
before closing the page; after an unexpected close, only the last durable
checkpoint is available (browser periodic saves occur every 16,384 steps).
Browser JSON exports include continuation state and can be reimported. Older
browser files, ONNX controllers and native-to-browser exports remain usable
for inference but cannot be continued in the browser without optimizer state.

Native checkpoints include `learning_state.json` and matching hashes for the
policy/normalizer pair. The native ZIP download includes these files. Continue
an existing native run with:

```bash
.venv/bin/rlsim resume runs/RUN_ID --additional-steps 65536
```

Older native runs without full continuation metadata remain inference-only.
The gallery's native-exported browser controllers are also inference-only.

## One policy, different proportions

**Explore → One policy, different proportions** replays a frozen quadruped
controller on editable torso, upper-leg and lower-leg lengths. Release a slider
to start a new episode, or disable automatic replay and use **Apply proportions
& replay**. Switching to **Trained on one body** runs the original matched baseline on the
same proportions and episode seed. No optimization or normalization updates
occur during these replays. Checkpoint identifiers remain fixed as shapes change.

The **Coordinated · wider range** controller adds a speed slider (0.5–1.5 m/s)
and samples proportions from 60–140%, while retaining the original withheld
upper/lower-leg corner. The interface labels the range of the selected model.
The broader study compares two equal-budget refinements of the same pretrained
policy, selects a condition on a separate validation audit, then evaluates it
on 24 fresh random shapes plus five nominal speed commands, with three episode
seeds. Native, browser CPU and browser WebGPU results are reported separately
in `static/demos/locomotion-study.json`. These 87 episodes per backend are not
87 independent training runs. Failures are retained; the sliders are a way to
explore both successful transfer and its limits. Earlier controllers remain
available in the selector.

The published broad controller has 2,490,368 cumulative policy steps. Its final
gait checks passed **84/87** episodes in native physics and **86/87** on both
browser CPU and NVIDIA WebGPU. The previous coordinated policy passed **65/87**
on the same native suite. On those 65 matched successful native episodes, the
median ratio of normalized motor work was 0.609 (about 39% lower). The paired
no-work-penalty refinement also passed all 58 validation episodes, but the
selected penalty condition used about half its normalized work on matched
successes. These are within-study comparisons from one warm-started seed.

The separate boundary/speed stress audit passed **40/43**. Very long upper legs
with very short lower legs, and the withheld shape at 1.5 m/s, can still fall or
underuse a leg. The demo reports failed gait checks, and all failures are retained
in the downloadable study. This improves the measured gait; it does not establish
global optimality, arbitrary-body transfer, or real-world energy efficiency.

The original experiment samples lengths independently from 80–120% of the nominal dimensions,
widening over 131,072 steps. The region with upper legs ≥110% and lower legs ≤90%
is excluded at every reset. The demo labels that withheld region and extrapolation
outside the training limits separately. MuJoCo rebuilds inertia, joint anchors
and initial standing height. Joint count, topology, segment thickness, nominal
total mass and nominal torque are held constant. Both conditions use the same
mass/torque/friction randomization during training.

The experiment pairs three training seeds with fixed-body and varied-body PPO,
using 1,048,576 steps per run and the same validation suite to select checkpoints.
Final evaluation covers twelve specified bodies and three separate episode
seeds per body. The original speed-and-survival check requires surviving 20 seconds and mean absolute
speed error ≤0.35 m/s at a 1 m/s target. Aggregate tables report means and sample
standard deviations across **training runs**, retaining failures. The two demo
controllers are selected by validation return, never by final test performance.
This is a small fixed-topology study, not evidence of arbitrary-body or real-robot
transfer. Physics and floating-point differences can affect browser playback.
The evidence panel also reports the actual bundled checkpoints evaluated in
CPU browser physics, including falls. It keeps those single-controller episode
counts separate from native averages across training runs.

Reproduce from the repository with the native CUDA trainer:

```bash
.venv/bin/python -m morphology_study --output experiments/my-proportions \
  --steps 1048576 --seeds 7 17 27 --workers 2
.venv/bin/python scripts/build-generalization.py --study experiments/my-proportions
.venv/bin/python scripts/verify-generalization.py --url http://localhost:8777/
npm run build:static
```

The study refuses to reuse an output directory with a different protocol or
training source hashes. One worker uses less memory than two. Browser playback
uses WebGPU when available and CPU otherwise; training across proportions is
currently native only. A native `TrainConfig` accepts
`morphology={"enabled": true, "width": 0.2, "ramp_steps": 131072}`. Omit it for
ordinary training. Full native continuation preserves sampler RNG and the
cumulative variation schedule. Portable browser exports are inference-only.

## Coordinated walking and gait quality

For a template quadruped, biped or hexapod, choose **Train → Task → Coordinated
walking (templates)**. Use Gradual speed if curriculum is enabled. Start a new
run: this objective changes the observation and action contracts. Existing
Velocity controllers and true continuation retain their original behavior.
Imported rigs currently require their existing Velocity/Balance objectives.

The policy learns joint-position targets, converted to torque with a PD loop at
each physics substep. It receives a phase clock, per-foot ground estimates and
previous actions. The reward adds alternating contact/clearance targets, body
height and level posture, and penalties for foot movement along the ground,
vertical bounce, pitch rate and abrupt actions. The phase specifies alternating
support groups (diagonal pairs for a quadruped); it does not supply joint motion
or replay an animation. **Simulate rest targets** sends zero target actions with
the PD servos still active, so it is not a zero-torque baseline. This is a gait prior, not unrestricted discovery of all
possible gaits. The template's planar root still supplies lateral stability.

The initial development runs use PPO entropy coefficient 0.01 instead of 0.005.
A separate refinement forks the varied controller, increases the ground-drift
penalty from 0.2 to 1.0 and lowers entropy to 0.005 for another 524,288 steps.
Weights, Adam state and normalization are retained, while episodes and validation
selection restart. This is explicit reward fine-tuning, not unchanged continuation.
The optional **Ground drift penalty** field controls that weight (0–5; new UI
runs start at 1). Excessive penalties can make learning harder. Entropy
encourages exploration during training; deterministic inference uses the learned
mean action. More random action variation by itself does not require every leg
to contribute. The reward and measured foot use address that missing objective.

Two further optional controls are available for coordinated walking:

- **Count knee or torso contact as a fall** treats an active, penetrating
  non-foot contact with the floor as a fall, with a penalty. It uses MuJoCo
  collision contacts at every physics substep. New UI walking runs enable it;
  older checkpoints keep their original setting. With **End episode on fall** disabled,
  it still marks and penalizes the fall but does not terminate the episode.
- **Motor work penalty** (0–2, default 0) discourages absolute mechanical motor
  work, normalized by body weight, commanded speed and control-step duration.
  This is optional reward shaping; a large value can discourage walking itself.
  It is not an established optimal setting.

Template inference reports mechanical work in joules, mean mechanical power,
RMS torque, actuator saturation, and work divided by body weight and forward
travel (cost of transport). Work integrates `abs(actuator_force *
actuator_velocity)` at each 4 ms physics substep. Positive work is also recorded;
negative work is not assumed to regenerate electricity. These are numerical
simulation measurements, not battery consumption. Cost of transport is omitted
below 0.1 m forward travel. Compare efficiency only between episodes that meet
comparable walking, survival and speed criteria; an early fall is not efficient
locomotion.

The optional native `ProportionTraining` settings also support a width up to
±40%, a distinct `withhold_width`, a per-episode `speed_range`, and a fraction
of samples on the range boundaries (`boundary_fraction`). The default sampler
and checkpoint metadata remain compatible with the original ±20% experiment.
`robust_validation: true` uses eight fixed development cases for the broad
±40%, 0.5–1.5 m/s experiment; it is not an adaptive validation generator.

The paired coverage/efficiency study uses the same pretrained policy, optimizer,
normalizer, seed and budget in two forks, with work weights 0 and 0.2. Both add
foot-only support and wider sampling. It records the source hashes and selects
checkpoints using separate validation episodes. This isolates the work-penalty
choice within one warm-started seed; it is not evidence across independent
training seeds. Run it and audit the completed folders with:

```bash
.venv/bin/python optimization_study.py --output experiments/my-optimization
.venv/bin/python locomotion_audit.py --source experiments/my-optimization/work-0 \
  --output experiments/my-optimization/work-0-corners.json
.venv/bin/python locomotion_audit.py --source experiments/my-optimization/work-0.2 \
  --output experiments/my-optimization/work-0.2-corners.json
```

The audit's `physical` work field is a separate 50 Hz endpoint estimate,
retained for comparisons with its initial baseline. Its `effort` field uses
physics-substep integration. It also checks actual foot contact surface velocity
and counts non-foot ground support. `--suite validation` and `--suite final`
use disjoint shape and episode seeds; reserve the final suite for the selected
candidate. None of these checks proves a globally optimal gait.

Inference reports steps per leg and foot movement on the ground. A counted step
needs at least 0.12 s airborne, clearance of 4% of total leg length, and horizontal
travel of 6% of leg length. Ground contact uses foot-sphere clearance within
12 mm of the flat floor. The displayed foot-slip metric measures foot-center
drift between consecutive grounded samples; it also includes rolling of the
spherical foot, so it is not a force-based measurement of contact-point slip.

The stricter gait check requires 20 s without falling, mean absolute speed error
≤0.35 m/s, at least six steps **on every leg**, each leg grounded 15–85% of the
samples, foot-center drift ≤0.35 m/s and torso pitch RMS ≤0.25 rad. This catches
stationary legs, dragging and excessive body tilt, but visual inspection still
matters. The older gallery/comparison results measured speed and survival only.

Reproduce the two new development runs (one fixed shape, one varied shape):

```bash
.venv/bin/python -m walking_study --output experiments/my-walking --device cuda
.venv/bin/python refine_walking.py --source experiments/my-walking/varied \
  --output experiments/my-refinement/varied --foot-drift-weight 1
.venv/bin/python scripts/evaluate-walking.py --study experiments/my-walking \
  --refinement experiments/my-refinement/varied
.venv/bin/python scripts/capture-walking.py
.venv/bin/python scripts/build-walking-gallery.py
npm run build:static
```

Both use 1,048,576 steps, 64×64 networks, eight environments and four PPO epochs.
The varied run uses the same withheld proportion region as the previous study.
Checkpoint choice uses separate validation episodes. This small development
comparison changes the reward, action representation, observations and entropy
together; it cannot attribute improvements to a single component or establish
reliability across training seeds.

Design references: [Legged Gym's control and reward implementation](https://github.com/leggedrobotics/legged_gym/blob/master/legged_gym/envs/base/legged_robot.py)
and [contact-conditioned multi-gait learning](https://arxiv.org/abs/2408.00776).
RLsim uses its own small PPO objective, not a reproduction of those systems.

## Recommend a starting setup

Open **Train → Recommend settings** after importing or editing a body. The tool
shows the current and suggested values, its reasons, and physical issues worth
checking. **Apply settings** fills the training controls; it does not start a
run. **Undo** restores the previous values until you edit the setup again.
Changing the body, task, environment or training settings discards the old
recommendation. All suggested values remain editable.

The adviser uses the compiled observation/action dimensions, articulation,
physical-rig structure, and the tested training device. Browser previews also
report nominal mass and conservative collision bounds for the tracked body's
tree, excluding the floor and unrelated scenery. Rendered polygon count does
not determine policy capacity.

| Signal | Suggested adjustment |
| --- | --- |
| Controls, observations, humanoid/free-body coordination | Two hidden layers with 64, 128 or 256 units; more demanding bodies get larger trial budgets. |
| Coupled, high-dimensional control | More conservative learning rate and PPO clipping, with fewer repeated passes over each rollout. |
| CPU/GPU and available browser resource hints | Environment count and minibatch size; minibatches divide the full rollout. Physics still runs sequentially on CPU. |
| Selected objective | Speed curriculum for Velocity; shorter initial budgets for Balance. Custom rewards receive a warning because their behavior cannot be inferred from body shape. |
| Reliable leg lengths and gravity | A modest starting Velocity target. Other bodies retain the current speed. |
| Actuation, scale, mass distribution, damping and floor friction | Explain issues that network changes cannot solve; no setup is offered for a body with zero actuators. |

The tool initially recommends fixed physics to establish behavior. Randomization
can be enabled in later trials to train robustness; disabling it is a tradeoff,
not a universal improvement. The body, reward, selected objective, seed and device
choice are preserved. The recommendation may include a **target speed** change,
shown explicitly before Apply. Selecting a saved model initializes its inference
speed to the trained value; you can edit that speed to test other commands.

These are deterministic engineering heuristics, **not an optimization search
or an experimentally established best configuration**. Larger networks, longer
runs or a smaller learning rate can also perform worse. Complex humanoids may
need more than the current 2,000,000-step limit. A Balance trial starts a separate
policy; it does not warm-start Velocity training. Test several seeds and compare
inference behavior, falls and task performance, not only reward.

The qualitative guidance follows [SB3's training/evaluation advice](https://stable-baselines3.readthedocs.io/en/master/guide/rl_tips.html)
and [PPO's rollout/minibatch definitions](https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html).
The [empirical on-policy RL study](https://arxiv.org/html/2006.05990v1) discusses
interactions among implementation and tuning choices. RLsim's capacity thresholds,
sample budgets and speed formula are its own starting rules, not results claimed
by those sources.

## Import conventional bodies and characters

Open **Body → Import model files**. Choose a model together with its linked
textures, materials, and binary buffers, or select its folder/ZIP archive.
When a bundle contains multiple models, choose its entry file. Imports use
**This device**, and the supplied assets stay in the browser.

| Source | Physics and rendering |
| --- | --- |
| MJCF / MuJoCo XML | Uses the model's bodies, joints, actuators, and supplied assets. |
| URDF | Converts the robot hierarchy and supported joints into a MuJoCo model. |
| Rigged GLB, glTF, or FBX | Preserves the source mesh, materials, and skin weights. Builds a separate physical rig from the reference pose, skin influences, and inferred body connections. |
| Unrigged GLB, glTF, FBX, OBJ + MTL, or STL | Keeps the original mesh and creates one rigid bounding-box collision body with six force/torque controls. |

**Auto** uses an available source skeleton. **Skeleton** requires one;
**Rigid body** imports a single physical object. The separate **Body structure**
chooses **Auto**, **Humanoid**, or **Generic skeleton**. Auto applies humanoid repair only
when it recognizes the required anatomy; generic conversion keeps other
articulated shapes usable without imposing a human skeleton. Read the import
report and inspect the collision overlay, especially for unfamiliar rigs.

An unrigged statue does not gain movable limbs from its visible shape during
ordinary import. Rig it in a modeling tool or use the optional SkinTokens
preparation workflow below, then import the rigged result.

Scale and mass affect the generated physical body. GLTF uses metres; FBX
units are converted using its unit metadata. OBJ and STL coordinates are
treated as metres and Y-up, so orient and scale those files when necessary.
Imported characters move freely in 3D. Their mesh follows the simulated body
and joint transforms; saved animation clips are not played to suggest learning.
**Show colliders** overlays the generated collision geometry for inspection.

The converter uses bind/reference transforms when available, so an asset's
exported animation pose does not define its physical rest pose. It distinguishes
animation helpers from deforming bones and preserves structural ancestors where
needed. Recognized humanoids get connected thigh–shin–foot chains, knee
hinges, fitted foot contact boxes, and a pelvis-based tracking root. Terminal
deforming bones can have joints; a bone does not lose its joint simply because
it has no children. Recognized humanoids are uniformly rotated to face the
simulation's +X travel direction.

Collision shapes are fitted to bone-associated geometry and skeleton segments.
Mass is estimated from fitted collider volume and the selected total mass,
with conservative humanoid bounds to limit distortion from hair and clothing.
These are approximations, not measurements of anatomy: a graphics file does
not specify density, joint limits, strength, friction, or internal structure.
Clothing and hair can also distort geometric estimates. Self-contact is disabled
and floor contact remains active. Use authored MJCF/URDF when these physical
properties are already known.

The import report describes repairs and assumptions. **Advanced physics**
lets you edit the generated rig and apply it only after validation and a MuJoCo
preview succeed. The full graph is available for advanced topology and collider
changes. Source assets and skin weights remain part of the body bundle; a
mapping from physical bodies to source bones drives the original mesh.
Export the body JSON to retain the editable graph, visual assets, and physics.
An MJCF export retains physics but not the Three.js character skin or editor
metadata.

### Lock fingers and prepare locomotion physics

For one joint, open **Body → Advanced physics**, select the body part, set
**Joint → Fixed**, and click **Apply physics**. Fixed means rigidly attached to
its parent in the rig's rest pose. Setting motor torque to zero instead leaves
a passive joint that can still move. Both keep the visible mesh.

Recognized humanoids also offer **Body → Locomotion setup**:

1. Choose **Total mass** for the body you intend to simulate. The input starts
   at the current mass. **60 kg is an example adult mass, not a measurement
   inferred from a character file.** Mass remains a physical design constraint;
   do not optimize it solely to increase reward.
2. Start with **Strength multiplier: 1** and **Lock finger joints** enabled.
   Click **Review physics setup** to inspect the control count, mass and
   individual joint changes.
3. **Apply setup** compiles the proposed rig before committing it. Finger
   joints become fixed, their motors disappear, and their mass and visible
   geometry remain. The helper sets separate starting torque and damping values
   for recognized major joints. It preserves joint limits, pivots and colliders.
4. **Undo setup** restores the previous body until another setup, body or
   environment edit invalidates that review. **Export body JSON** retains the
   complete prepared body. Existing saved controllers remain attached to their
   original physics.
5. Use **Train → Recommend settings** again for the reduced control dimensions,
   then train a new controller.

This helper uses chosen mass, gravity, leg length and joint role to scale motor
strength. It is restricted to recognized humanoids with plausible connected leg
chains. Unknown joints are flagged for manual review; generic robots can use
the individual joint editor. Mass changes preserve the existing part-mass
fractions, which may still need correction for an unusual mesh. It does not
measure muscle strength, fit physiological anatomy or search for optimal values.
Ball-joint torque is a per-axis limit: simultaneous commands on three axes can
produce a larger combined torque. Damping resists motion; it does not hold a
standing pose. See the [MuJoCo joint and actuator reference](https://mujoco.readthedocs.io/en/stable/XMLreference.html#body-joint).

To tune settings in practice, first inspect collisions and a short simulation
for disconnected parts, invalid states or explosive motion. An unpowered fall
is expected and does not prove that training will fail. Then compare strength
multipliers such as **0.75, 1 and 1.25** at the **same chosen mass**, with the
same task, target speed, network, PPO settings and sample budget. Use at least
three training seeds and several separate evaluation seeds. Compare falls,
upright duration, speed error and distance, not reward alone: changing the
number of motor controls also changes action-based effort penalties. Short
runs only screen for compatibility; a reliable choice requires longer trials.
After choosing a setup, test its robustness to modest physical variation.

Ordinary imported-rig runs retain their original physical configuration. Train and
evaluate a new run to use an edited rig; changing mass or torque changes its dynamics.
The proportion demo above explicitly supports selected length changes with frozen weights.
Saved runs keep their original body and are not silently migrated. To update a
character imported with an older converter, reimport its original model file.
An old checkpoint will still replay its original physics.

Successful import means the model compiles and can be inspected, not that it
has learned to stand or walk. Check joints and contacts first, try **Balance**
for a standing character, and then experiment with **Velocity**. Starting a
different training task currently starts a new policy; there is no automatic
balance-to-walking transfer. New browser checkpoints support continuation
with their original objective; they do not switch a Balance policy into Velocity.

Include every referenced file locally; imports do not fetch remote asset URLs.
The visual loaders support ordinary uncompressed meshes and standard image
textures, including TGA through its local decoder. Export Draco/Meshopt/KTX-compressed
assets without those codecs before importing. The WebGL renderer displays
original mesh geometry and skinning; if WebGL is unavailable, the basic canvas
view displays the physical proxies.

Body bundles may contain up to **512 files and 64 MB of expanded data**.
Compiled physical models support up to **128 bodies and 256 actuators**.
These are import limits, not a guarantee that a complex model will train
quickly on every visitor's hardware.

## Optional SkinTokens rig preparation

[TokenRig / SkinTokens](https://github.com/VAST-AI-Research/SkinTokens) generates
a skeleton and skin weights for a visual mesh. It broadens the set of assets
that can enter the physics converter; it does not generate measured masses,
contact properties, motor limits, or a locomotion policy.

Open **Body → Optional AI rigging** to generate a rig with a configured local
SkinTokens installation. Choose an unrigged GLB, check setup, and generate.
Then **Review body settings → Import body** runs physics checks automatically.
Prepared humanoid, giraffe and six-legged examples are available in the same panel,
including on static hosting. They contain no walking policies.

The guided workflow keeps original geometry, materials and textures, adds
predicted skinning, and rejects changed shape/alignment. It flags suspicious
skeleton asymmetry and checks collisions, mass distribution, motor responses and
short simulation stability. Generic unnamed skeletons still need joint-limit
review. Errors block new training; warnings explain remaining assumptions.
See [measured validation and limitations](docs/auto-rig-validation.md).

The optional `rlsim rig-prepare --preserve-mesh` command provides the same
preservation step outside the UI. Ordinary imports never download pretrained
models or invoke the generator.

The official inference implementation requires **an NVIDIA GPU with at least
14 GB VRAM**, CUDA, Python, Blender's Python module, and the pretrained weights.
This requirement belongs to rig generation; RLsim's browser training still
uses WebGPU with CPU fallback. A static site cannot directly execute the
visitor's installed CUDA runtime. Users without the preparation environment
can import a rigged GLB produced elsewhere or use an existing authored rig.

See [SkinTokens setup and commands](docs/skintokens.md) for isolated installation,
configuration, readiness checks, generation options, and troubleshooting.
The application also includes a browser-readable import/rigging guide.
AI-generated rigs remain editable estimates and should be checked before training.

## Define the task

Open **Train → Task** to choose **Velocity**, **Balance**, **Reach target**, or
**Custom reward**. Episodes can last **1–120 simulated seconds**; 20 seconds
is the default. Select the **Tracked body** whose position and orientation
define progress and falling. An empty selection uses the model's root/default
body. For example, track a robot's hand to reward reaching rather than its base.

Target coordinates are world-space metres with Z up. The target marker appears
for reaching and custom tasks. Fall detection uses the tracked body's height
relative to its starting height and its tilt threshold. Disable **End episode
on fall** for tasks that should continue after tumbling. Body, task, target,
episode duration, and fall settings are saved with the controller.

Custom rewards use a parsed mathematical expression; they do not execute
JavaScript or Python. Arithmetic, parentheses, comparisons, logical operators,
and `condition ? yes : no` are supported. Both `^` and `**` mean exponentiation.
Available functions are `abs`, `min`, `max`, `sqrt`, `exp`, `log`, `sin`, `cos`,
`tanh`, `pow`, and `clamp(value, low, high)`.

| Variables | Meaning |
| --- | --- |
| `speed_tracking`, `forward_progress` | Target-speed agreement and normalized signed forward progress |
| `alive`, `upright`, `height_tracking` | Survival, uprightness, and agreement with starting height |
| `energy`, `smoothness`, `fall` | Negative effort and action-change costs; `fall` is −1 when fallen |
| `target_progress`, `target_distance`, `target_success` | Approach rate, negative target distance, and 1 within 0.2 m of the target |
| `x`, `y`, `z`, `vx`, `vy`, `vz`, `speed` | Tracked-body position, velocity components, and velocity magnitude |
| `height`, `initial_height`, `target_speed`, `distance` | Current/starting height, commanded forward speed, and positive target distance |
| `time`, `dt`, `pi` | Elapsed episode seconds, seconds per action, and π |

For example, `target_progress + 2*target_success + 0.05*energy` rewards reaching
with a small effort penalty. Expressions are limited to 2,000 characters.
Unknown names and non-finite results, such as division by zero, produce an
error. Inspect the resulting motion as well as reward: a formula can reward
unintended behavior.

## Import a trained controller

Under **Model → Import model**, browser mode accepts exported RLsim PPO JSON,
a standard `.onnx` file, or a ZIP containing an ONNX model and an RLsim manifest.
These imports support **inference only**; they do not resume an external
optimizer or train an ONNX graph.

Select the body and task, then use **Export interface** to download the exact
observation names/order, action names/order, tensor dimensions, body/assets,
task configuration, and a manifest template. Matching the number of inputs is
insufficient: an external controller must use the same observation semantics
and preprocessing as its training environment.

ONNX controllers must accept one flat `float32` observation input shaped
`[observations]` or `[1, observations]` and emit a `float32` action vector
shaped `[actions]` or `[1, actions]`. Dynamic batch dimensions are allowed, but
inference uses one observation at a time. Recurrent state inputs, image inputs,
and multiple observation inputs are not supported. Actions are normalized
controls in `[-1, 1]`, mapped to the model's actuator ranges. A manifest can name
the action output when a graph has multiple outputs.

A bare ONNX file uses the currently selected body/task and raw observations.
For a self-contained controller, include a `rlsim-onnx-manifest-v1` JSON file
with its `model` filename, `config`, `observation_schema`, optional input/output
names, and normalization statistics. Include any ONNX external weight files
locally and list them in `external_data`. Embed preprocessing in the graph or
supply the training normalizer; do not apply both. The normalizer format is
`{size, clip, count, mean, m2}`, where `m2 = variance * count`; preprocessing is
`clip((obs - mean) / sqrt(variance + 1e-8), -clip, clip)`.

The editable PyTorch example builds an ONNX graph and its manifest ZIP for
any exported body interface:

```bash
.venv/bin/python -m pip install onnx
.venv/bin/python examples/export_onnx_controller.py \
  --interface rlsim-controller-interface.json --output controller.onnx
# Import controller.zip under Model.
```

The example weights are **untrained**. Replace its actor with your trained
network and provide the correct preprocessing before export. ONNX inference
uses WebGPU when supported and falls back to the visitor's CPU if the runtime
cannot execute the graph there.

For a native RLsim checkpoint, convert its policy and observation statistics
directly:

```bash
.venv/bin/rlsim export-browser runs/RUN_ID controller.json
```

This supports the project's flat PPO actor/critic networks with 1–4 equal-width
Tanh hidden layers. `examples/custom_controller.py` demonstrates the
`morpholab.portable.export_torch_policy` helper for compatible external PyTorch
actor/critic networks. Use ONNX for other supported feed-forward architectures;
arbitrary `.pt` and Stable-Baselines3 ZIP files are not browser model files.

## Network and training settings

Open **Train → Advanced** to change the policy architecture and PPO settings.
The actor and critic each use the selected number of hidden layers and neurons
per layer, with tanh activation. Settings are saved with each model and restored
by **Use model settings**. **Reset advanced settings** returns to the selected
execution mode's defaults.

| Setting | What it changes | Browser / server defaults |
| --- | --- | --- |
| Neurons per layer | Width of each hidden layer | 64 / 256 |
| Hidden layers | Depth of both networks | 2 / 2 |
| Learning rate | Size of optimizer updates | 0.0003 / 0.0003 |
| Batch size | Maximum samples in an optimizer minibatch; capped by rollout size | 128 / 512 |
| Update epochs | Passes over each collected rollout | 4 / 10 |
| Rollout steps | Steps collected per environment before updating | 128 / 256 |
| Discount factor | Weight assigned to future rewards | 0.99 / 0.99 |
| Entropy coefficient | Strength of the exploration bonus | 0.005 / 0.005 |
| PPO clip range | Limits changes to action probabilities during updates | 0.2 / 0.2 |

Start with the defaults, then vary one setting at a time while holding the body
and seed fixed. Larger networks use more memory and computation; they do not
necessarily learn locomotion faster. Changing these controls configures a new
training run. Existing model weights keep their original architecture.

### Validation checkpoints, PPO safeguards and inference caching

**Train → Advanced → Keep best evaluated checkpoint** is enabled for new runs.
Every 16,384 training transitions (at the next completed rollout), and at normal
completion, RLsim runs three deterministic validation episodes at the original
task and full requested speed, with nominal physics. These episodes do not update
the policy or observation normalizer and are additional to the training-step
budget. Change the interval/episode count, or disable selection, in Advanced.
Stopping interrupts browser validation; an incomplete evaluation never replaces
the best checkpoint. A run stopped before its first evaluation has latest weights
only.

“Best” means **highest mean raw task return on that fixed validation seed suite**,
not guaranteed best walking. The UI also reports survival and speed error. Use
fresh evaluation seeds and inspect the motion before drawing conclusions,
especially with custom rewards. Validation seeds are derived from the run seed
with a separate offset; use multiple training seeds for comparisons.

**Model → Checkpoint** selects **Best evaluated** or **Latest**. Both retain their
own matching observation-normalization statistics. Browser model downloads
include both; older files remain usable as latest-only models. Native downloads
retain `policy.zip`/`normalize.pkl` for latest and add
`best_policy.zip`/`best_normalize.pkl`/`best.json` when available. Native
`load_checkpoint(path, checkpoint="best")` selects the best pair; the helper's
default and `export-browser` retain their existing latest-checkpoint behavior.
Server inference defaults to best when available.

**Target KL**, default `0.03`, stops further optimization on a rollout when
approximate divergence exceeds `1.5 × target`. It does not stop the whole run.
Set `0` to disable it. Browser PPO checks before each minibatch update; native PPO
uses Stable-Baselines3's corresponding safeguard. Clipping and gradient limits
remain active. This reduces excessive updates but does not guarantee convergence.

Browser inference automatically retains **one** prepared controller and physics
model. Repeating inference resets episode state and the requested seed, while
reusing compiled physics and warmed actor weights. Changing checkpoint contents,
world settings or target speed replaces the cache; starting training releases it.
Errors also release it. PPO playback loads only actor tensors, without the critic
or optimizer. ONNX sessions use the same bounded cache. The first load still
requires preparation, and this change does not benchmark or switch inference
devices separately from the existing device selection.

**Stop inference**, beneath the viewer while loading or generating, cancels
further predictions in both This device and Server modes. **Pause** only pauses
playback. Stopping also pauses playback and retains generated frames as a partial
episode; Play and the timeline can review those frames without more inference.
For a body baseline, the same control is labelled **Stop simulation**. A model
load or prediction already executing may finish before cancellation is handled;
the control shows **Stopping…** until generation exits. Browser cancellation
releases the prepared inference cache, and saved models are unchanged.

### Optional balance-to-walking curriculum

With **Use curriculum** checked, choose **Balance → walking** for Velocity tasks.
It starts at zero-speed balance with nominal physics, then uses 25%, 60% and 100%
of the requested speed. Physics randomization grows from 0% to 25%, 60% and 100%
of its normal range when Randomize physics is enabled.

A stage advances only when the most recent 12 completed training episodes have
at least 80% survival and mean absolute speed error no greater than
`max(0.15 m/s, 0.35 × stage speed)`. The balance stage uses the same error gate
around zero speed. Stage changes occur between rollouts and reset training
episodes; observations keep the same schema. The UI displays the current stage.
A run can finish while still learning balance; full-speed validation always uses
the final requested task, so stage reward cannot masquerade as walking success.

**Gradual speed** preserves the original elapsed-budget speed ramp. Uncheck
Use curriculum for no curriculum. Balance, Reach target and Custom reward tasks
do not support the staged option. Both browser and native template training
support staged learning; imported bodies continue to train in the browser.
Native CLI options include `--curriculum-mode staged`, `--target-kl`,
`--evaluation-interval`, `--evaluation-episodes`, and `--no-best-checkpoint`.

### Optional joint-target control for imported rigs

For an imported character with an editable physical rig, open
**Body → Advanced physics → Control mode → Joint targets (PD)**, review excursion,
stiffness and damping, then **Apply physics**. Direct torque remains the default.
This changes what actions mean and requires a new controller; saved controllers
retain their original bodies and control mode.

Hinges use MuJoCo position servos. Ball joints use MuJoCo's geodesic orientation
servo, which computes quaternion orientation error, rather than independent
Euler-angle errors. They run at the physics timestep and produce bounded torque.
The joint torque field caps hinge torque magnitude and **total vector torque**
for ball servos (direct torque uses the field per axis).

Defaults are 30° target excursion, stiffness scale 1 and servo damping ratio 0.7.
Targets stay within the joint limits. Ball actions describe an exponential-map
vector with per-axis bounds `excursion / sqrt(3)`, so its norm stays within the
requested angle and 90% of the joint cone limit. Hinge targets use the overlap of
±excursion and the joint range. Normalized zero holds zero angle, or the nearest
allowed angle; positive and negative actions scale toward their respective bound.
Stiffness is estimated from torque divided by excursion, multiplied by the
stiffness scale. Servo damping is inertia-aware; existing passive joint damping
still applies. These defaults are starting values, not measured optimal gains.
Fixed/passive joints retain their behavior; actuated free roots are unsupported.
**Simulate rest targets** sends zero target actions while the servos remain
active; it is not a zero-torque baseline.

The mesh, skin weights and rest transforms remain unchanged. Exported body JSON
and MJCF XML retain the servos. XML includes the `rlsim-joint-target-v1` action
schema marker; reimporting it into RLsim preserves the normalized target mapping.
Outside RLsim, MuJoCo controls are target angles in radians, so external training
must apply the mapping described by **Export interface**. ONNX manifests now
include `action_schema`; mismatched declared control modes are rejected.
Use the bundled MuJoCo 3.16 runtime or a runtime supporting orientation servos.

## Physics and learning

MuJoCo integrates gravity, floor contact, friction, joint limits, and motor
torques. The built-in legged templates constrain the torso to x/z translation
and y-axis pitch; imported models use their own joint definitions and can move
freely in 3D. Template self-contact is disabled to allow a broad range of editor
dimensions. Each template leg has an independent hip and knee actuator; 60% of
body mass belongs to the torso and the rest to the legs.

For built-in templates, one action advances five 4 ms physics steps. Imported
models use their model's integration timestep. The default template task ends
after 1,000 actions (20 simulated seconds) or a fall; task controls can change
the duration and fall rules. Browser and native template environments use the
same model, observation layout, reward, and termination rules. Their
random-number generators differ, so equal seeds across backends do not imply
identical initial states. Conventional imported bodies run in browser mode.

The following values describe the defaults; **Train → Advanced** changes the
architecture and PPO settings for new runs.

| Default training setting | Browser | Native server |
| --- | --- | --- |
| Implementation | TensorFlow.js PPO | Stable-Baselines3 PPO |
| Actor and critic | Separate 64×64 networks | Separate 256×256 networks |
| Rollout length | Up to 128 steps/environment | 256 steps/environment |
| Minibatch size | Up to 128 | Up to 512 |
| Update epochs | 4 | 10 |
| Learning rate | 0.0003 | 0.0003 |
| Discount / GAE lambda | 0.99 / 0.95 | 0.99 / 0.95 |
| Normalization | Observations; rewards stay raw | Observations and rewards (`VecNormalize`) |

Browser execution uses a Web Worker to keep controls responsive. PPO samples
Gaussian actions, clips a copy for the simulator, and retains the original
samples for its likelihood calculation. New browser checkpoints include Adam state for continued training. Older
checkpoints and native-to-browser exports support inference only. Native step budgets
round up to complete rollouts.

WebGPU selection compares a complete PPO update with a CPU reference before
accepting the device. GPU acceleration does not guarantee higher throughput
for these small networks; physics and GPU readback still contribute to runtime.

The built-in template environments include normalized morphology inputs, an optional speed
curriculum, and per-episode physics randomization: mass/inertia ×0.8–1.2, motor
torque ×0.8–1.2, and friction ×0.6–1.4. The sampled factors are included in the
observation. The curriculum increases the commanded speed from 25% to 100%
over the first 60% of training. Imported bodies use their full joint positions
and velocities, task target, tracked-body orientation/velocity, and previous
actions instead of the template morphology vector. Use **Export interface**
for the exact observation and action order for a particular body.

The default velocity reward combines speed tracking, signed progress, upright survival,
squared control effort, action smoothness, and a fall penalty. Effort is a
normalized torque proxy, not joules. Check inference distance, speed and
survival alongside reward.

## Existing native experiment

Local run `20261020T132800-fc7a3017` trained the default quadruped for 524,288
steps with seed 7, curriculum and randomization. Training and final evaluation
took 259.8 seconds on an RTX 5080. The saved run is available through
**Server** on this development machine; static deployments do not include it.

| Three held-out episodes per condition | Mean distance | Mean duration | Full-episode survival |
| --- | ---: | ---: | ---: |
| Zero torque, nominal physics | −0.98 m | 1.8 s | 0/3 |
| Trained policy, nominal physics | 17.76 m | 20.0 s | 3/3 |
| Trained policy, randomized physics | 15.58 m | 20.0 s | 3/3 |

This result belongs to the native CUDA implementation and this configuration.
It is not a browser training benchmark or a guarantee for other bodies.

## Research context

PPO is an established algorithm. The morphology observations, randomization
and curriculum are implemented research-inspired components; this project
does not reproduce a recent cross-embodiment foundation model.

- [PolyMorph](https://arxiv.org/abs/2610.19615) studies morphology-aware
  locomotion across robot bodies. The current fixed-body MLP policy does not
  reproduce its graph-attention architecture.
- [McARL](https://arxiv.org/abs/2505.18418) explores morphology-conditioned
  locomotion. Training across varied bodies would be needed for transfer.
- [LocoFormer](https://arxiv.org/abs/2509.23745) uses robot diversity and memory
  for cross-body adaptation. A graph/transformer policy remains an extension.
- [MuJoCo Playground](https://github.com/google-deepmind/mujoco_playground)
  offers a path to GPU-parallel physics through MJX / MuJoCo Warp.
- [Stable-Baselines3 PPO](https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html)
  documents the native algorithm.
- [MuJoCo JavaScript bindings](https://github.com/google-deepmind/mujoco/tree/3.16.0/wasm)
  and [TensorFlow.js](https://github.com/tensorflow/tfjs/tree/tfjs-v4.22.0)
  provide the browser runtime.

Body evolution would add a separate search over body parameters, training and
evaluating controllers for candidate shapes. This version learns joint control
for the selected body.

## Commands and checks

```bash
# Inspect usable native devices, or explicitly require one.
.venv/bin/rlsim doctor
.venv/bin/rlsim doctor --device cpu

# Native auto chooses CUDA, then MPS, then CPU.
.venv/bin/rlsim train --preset quadruped --steps 262144 --device auto
.venv/bin/rlsim train --body-file my-body.json --steps 1000000 --device cpu
.venv/bin/rlsim evaluate runs/RUN_ID

# Physics/API tests and real MuJoCo WASM integration checks.
.venv/bin/pytest -q
npm test

# Original 3D assets, local textures, skinning, and WebGL geometry cache.
RLSIM_SCENE_TEST=1 .venv/bin/pytest -q tests/test_scene_visual.py

# Real browser training, CPU fallback, storage, import/export and inference
# on a static server (requires Playwright Chromium).
MORPHOLAB_BROWSER_TEST=1 .venv/bin/pytest -q \
  tests/test_browser_local.py tests/test_browser_imports.py tests/test_onnx_browser.py

# Editable physics, optional rigging help, and generated-rig train/save/replay.
MORPHOLAB_BROWSER_TEST=1 .venv/bin/pytest -q \
  tests/test_rig_ui.py tests/test_rig_training.py tests/test_playback_stability.py
# Recommendation review/apply/undo and a capped PPO compatibility check.
MORPHOLAB_BROWSER_TEST=1 .venv/bin/pytest -q tests/test_training_advisor.py
# Optional locomotion physics review/apply/undo and fixed-finger training/replay.
MORPHOLAB_BROWSER_TEST=1 .venv/bin/pytest -q tests/test_locomotion_setup.py
# Checkpoint selection, staged learning, joint-target control and cached replay.
MORPHOLAB_BROWSER_TEST=1 .venv/bin/pytest -q tests/test_training_optimizations.py
MORPHOLAB_NATIVE_TEST=1 .venv/bin/pytest -q tests/test_native_training_optimizations.py
# Set RLSIM_VALERY_ASSET=/absolute/path/to/compatible-valery.glb for the real body.
# Set RLSIM_FEMALE_ASSET=/absolute/path/character.glb to use a real humanoid
# in the generated-rig training check and tests/physical_rig_check.mjs.

# Existing UI checks against the local service.
MORPHOLAB_TEST_URL=http://localhost:8777 .venv/bin/pytest -q tests/test_ui.py

# Hardware WebGPU check using an isolated browser already exposing CDP.
MORPHOLAB_GPU_CDP=http://127.0.0.1:9339 MORPHOLAB_TEST_URL=http://localhost:8777 \
  .venv/bin/pytest -q -s tests/test_browser_gpu.py tests/test_onnx_gpu.py
```

Validated on October 20, 2026: native physics, device, API, task and portable
controller checks; real MuJoCo WASM imports and skinning; browser PPO learning;
WebGL and canvas fallback rendering; and ONNX numerical inference. CPU fallback
on static hosting and WebGPU on a native Windows Edge NVIDIA adapter completed
real 4,096-step training, saving, and inference, including an imported rigged
character. ONNX checks verify real GPU compute dispatches and CPU fallback.
Static browser checks cover custom rewards, body/controller ZIP and JSON imports,
reload, download, stopping, and URL subdirectories without server API requests.
These short runs verify the workflow; the locomotion results above belong to
the longer native experiment.

The editable-rig checks also cover helper-bone exclusion, humanoid foot
connections, bind-pose fidelity, legacy visuals, graph validation and export
roundtrips. The Casual Female GLB completed a 4,096-step browser CPU training,
save and fresh-worker replay check with its generated physics. The editor was
checked at 1440, 900 and 390 px widths. SkinTokens helper checks cover process
handling, readiness, output validation and preservation of existing files;
actual pretrained SkinTokens inference has not been run as part of these checks.

Playback stability checks cover fixed training cameras, smooth horizontal follow,
WebGL/canvas rendering, streamed-frame ordering, buffering and pause preservation.
The female controller produced identical frames and scores through streamed and
non-streamed browser CPU replay. Actor-only PPO predictions match the original
deterministic action path, including tensor cleanup; these changes preserve the
saved physics and policy. GPU-specific timing was not remeasured for this update.

Recommendation checks cover rule boundaries, compiled body measurements,
review/apply/undo, stale-setting invalidation, and preservation of custom rewards
and seeds. The Casual Female body completed a capped 4,096-step browser CPU run
with the recommended network and PPO settings. Desktop and mobile layouts were
checked. These validate compatibility and the controls, not a learned gait or
an improvement in training success; the heuristic has not been benchmarked
across bodies and seeds.

Locomotion physics checks cover actual fixed-joint removal, unchanged rest poses
and visuals, moving-hand/fixed-finger attachment, mass scaling, finite simulation,
and review/apply/undo. On the Valery character, locking 30 finger joints reduced
the action count from 146 to 56. The revised 60 kg body completed a capped
4,096-step browser CPU PPO run and saved-controller replay; desktop and mobile
controls were also checked. These checks establish runtime compatibility, not
successful walking or an empirically optimal mass, torque, or damping.

Training-option checks cover best/latest weight-normalizer pairing after a
deliberately regressing policy, KL rejection before another update, actor-only
prediction parity, cache eviction and deterministic reset, curriculum success
gates, and actual MuJoCo hinge/orientation feedback and torque limits. Valery
completed a capped 4,096-step browser CPU run with locked fingers, joint targets
and staged curriculum, including fresh-worker replay. An 8,192-step native CPU
run checked staged learning and best/latest checkpoints. Hardware WebGPU PPO and
4,096-step imported-character runs passed on an NVIDIA Blackwell adapter for
both direct torque and joint targets. These establish compatibility and correct
mechanics; they do not establish a better learned gait or a measured speedup.

Explicitly requesting an unavailable native device reports an error. Automatic
selection happens before a run; a failed training job is not silently restarted.
Native runs save their policy, normalization statistics, configuration,
evaluation, and replay under `runs/`. Keep those files together when restoring.
Independent native processes should use separate `--runs-dir` directories.
