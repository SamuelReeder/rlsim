# Optional SkinTokens rig preparation

RLsim can invoke a separate [SkinTokens / TokenRig installation](https://github.com/VAST-AI-Research/SkinTokens) to generate a skeleton and skin weights, then import its GLB result. This is an optional preparation step. Browser PPO and normal model imports do not require it.

The upstream tool requires Python 3.11+, CUDA 12.1+, and an NVIDIA GPU with at least 14 GB of memory. It uses Blender's `bpy` module and `flash-attn`; it does not offer the CPU/MPS fallback used by RLsim training. These requirements and the official CLI are documented in the [upstream README](https://github.com/VAST-AI-Research/SkinTokens#installation).

## Separate installation

Install into a separate checkout and environment, not RLsim's `.venv`. The helper targets upstream revision `273b691d35989d71cd17ff2895fdc735097b92d1`. Unit tests use a substitute executable to test failures and cancellation. Real pretrained-model inference was separately tested on an RTX 5080 with humanoid, giraffe and six-legged inputs; see [the validation report](auto-rig-validation.md).

For Linux/WSL, with Git, `uv` and the appropriate NVIDIA CUDA development toolkit installed:

```bash
mkdir -p ~/tools
git clone https://github.com/VAST-AI-Research/SkinTokens.git ~/tools/SkinTokens
git -C ~/tools/SkinTokens checkout 273b691d35989d71cd17ff2895fdc735097b92d1
cd ~/tools/SkinTokens
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 --index-url https://download.pytorch.org/whl/cu128
uv pip install --python .venv/bin/python -r requirements.txt \
  "transformers>=4.57,<5" "bpy==4.2.0" "numpy<2" "scipy<1.17" rtree scikit-image
uv pip install --python .venv/bin/python flash-attn --no-build-isolation
.venv/bin/python download.py --model
```

The CUDA wheel URL must match a supported driver/toolkit combination. Follow the upstream installation instructions if your system needs another build. Its downloader retrieves the TokenRig and SkinTokens checkpoints plus the Qwen model configuration. Nothing is downloaded automatically by RLsim.

Configure the two absolute paths in the shell where you run RLsim:

```bash
export RLSIM_SKINTOKENS_HOME="$HOME/tools/SkinTokens"
export RLSIM_SKINTOKENS_PYTHON="$HOME/tools/SkinTokens/.venv/bin/python"
```

For persistent configuration with `./run.sh`, create `rigging.local.json` in the RLsim project containing `{"checkout":"/absolute/path/to/SkinTokens","python":"/absolute/path/to/its/python"}`. This local file is ignored by Git; environment variables take precedence. Restart the server after configuration.

The same paths can be passed per invocation with `--checkout` and `--python`. The executable must belong to the separate SkinTokens environment; RLsim never guesses or installs one.

## Use the guided workflow

1. Open **Body → Optional AI rigging** in the local RLsim app.
2. Choose an unrigged GLB and click **Check local SkinTokens setup**, then **Generate rig**. You can also select a GLB under **Import model files → Generate a rig with SkinTokens…**.
3. Wait for preparation and preservation checks. **Cancel preparation** stops the helper. A failed job leaves the current body and original file intact.
4. Choose **Review body settings**, set the real-world scale, total mass and initial motor torque, then **Import body**. Physics checks run automatically. **Save rigged GLB** keeps a portable copy.
5. Inspect **Check results** and **Show colliders**. Use **Advanced physics** to correct inferred settings. Errors block new training for prepared bodies; review notes remain visible without blocking. Body or environment changes invalidate the report; run **Check physics** again.

Already-rigged models should use normal **Rig: Auto** import. The guided preservation workflow deliberately rejects them instead of replacing existing skinning. **Prepared examples** lets you inspect the three tested results on static hosting too, without waiting for generation. They have no walking controllers.

Preparation runs on the local native server's NVIDIA GPU. Browser training remains on the selected available device. Static hosting cannot run this CUDA generator in a visitor's browser; use a prepared GLB or run the local app. The job API only accepts local hostnames and same-origin browser requests; it is not a public GPU hosting service.

## Check and prepare

From the RLsim project, using its own executable:

```bash
.venv/bin/rlsim rig-prepare --doctor
.venv/bin/rlsim rig-prepare --doctor --check-runtime
.venv/bin/rlsim rig-prepare /absolute/path/character.glb \
  --output /absolute/path/character-rigged.glb --dry-run
.venv/bin/rlsim rig-prepare /absolute/path/character.glb \
  --output /absolute/path/character-rigged.glb --preserve-mesh
```

Quick doctor checks the checkout, required model files, interpreter and Blender service port without loading ML dependencies. A complete file installation reports `configured: true`, `ready: null`. `--check-runtime` additionally starts the configured interpreter to check modules, CUDA availability, GPU memory, and a small CUDA operation. It does not load the pretrained networks. `ready: true` means these checks passed; it does not guarantee a particular mesh will rig successfully.

Preparation always runs the runtime check first. `--dry-run` does not launch the external interpreter or create an output directory. It reports the equivalent official invocation; the real run uses temporary input/output paths.

The helper accepts one **self-contained GLB** with embedded resources, at most 64 MB. Output has the same size limit so it can be reimported into RLsim. Export OBJ, FBX, glTF or other assets to GLB in a 3D editor first. This narrower helper interface permits copying the complete input safely; upstream's standalone CLI supports additional formats.

By default the command forwards `--use_transfer`, which asks upstream to retain textures and scale. `--no-transfer` disables that behavior. To regenerate skin weights while retaining an existing skeleton:

```bash
.venv/bin/rlsim rig-prepare /absolute/path/already-rigged.glb \
  --output /absolute/path/reskinned.glb --use-skeleton
```

`--use-skeleton` requires an input skin/skeleton. It does not generate missing bones.

## Output, cancellation, and import

The original GLB is copied to a private staging directory. SkinTokens runs against that copy. A successful process must produce a valid GLB containing a skeleton and a mesh with joint indices and skin weights before the helper publishes it.

An existing output is protected unless `--overwrite` is explicit. Even with that flag, a failed run leaves the old output intact. The input itself, including a symlink to it, cannot be selected as the output. The helper keeps a uniquely named log beside the requested output and prints its location.

The default timeout is 1,800 seconds; change it with `--timeout SECONDS`. Ctrl+C or timeout stops the launched process and its Blender descendants. Existing processes are not stopped: preparation refuses to begin if the upstream Blender port is occupied. A checkout lock prevents two helper jobs from using the same service. After a hard system crash, confirm its process has exited before removing `.rlsim-rig-prepare.lock` from that checkout.

Stop competing GPU workloads if the runtime check reports insufficient free memory. The model's total VRAM requirement is not a guarantee that inference fits alongside another active training job.

In RLsim, open **Body → Import model files**, choose the resulting GLB, and select **Auto** or **Skeleton** rig conversion. Inspect the converted joints and colliders before training. Predicted bones and weights are a starting point; the physics conversion still approximates masses, collisions and motor constraints.

## Preservation and physics checks

The guided workflow always uses `--preserve-mesh`. After upstream texture/scale transfer, RLsim compares both rest vertex clouds, with a maximum distance of 0.01% of the source bounding-box diagonal (minimum one microunit). It attaches predicted skeleton nodes and four normalized weights per vertex to a copy of the original GLB. Original position, normal, index, UV, color, image and material data are retained, including their buffer bytes. Mesh-local transforms remain intact. A shape/alignment mismatch is rejected. This currently supports ordinary uncompressed GLBs without morph targets or sparse position accessors; these cases receive actionable errors.

Weights are transferred from the nearest coincident rest vertex. This preserves appearance at rest, but it does not establish correct anatomical deformation. A symmetry heuristic flags predicted joints without mirrored counterparts only when the source geometry is nearly symmetric. It can miss errors and cannot determine the intended number of limbs. A failed prediction can be regenerated with **Generate rig**; sampling is stochastic.

Physics checks use a separate MuJoCo environment in the browser worker. They validate graph/compilation, mass ratios, floor penetration, deep non-neighbor collider overlaps, small positive/negative motor commands in zero gravity, and two seconds of finite passive simulation without resets. Falling without a controller is informational. Collision-pair checking is capped for large rigs. The checks do not measure real anatomy, guarantee standing, or choose optimal controller settings. Generic unnamed skeletons retain generic joint assumptions and get an explicit review note. Self-contact remains disabled.

## Local status and jobs

The native server exposes `GET /api/rig-prepare/status` using the environment variables above. It performs the quick check only. `configured`, nullable `ready`, `runtime_checked`, `issues`, `warnings`, `requirements`, and the external paths are returned. Explicit **Generate rig** submits a bounded GLB to `POST /api/rig-prepare/jobs`. Poll `GET /api/rig-prepare/jobs/{id}`, cancel with `POST /api/rig-prepare/jobs/{id}/cancel`, and download a completed result from `GET /api/rig-prepare/jobs/{id}/model`. Only one preparation runs at a time; native training is blocked while it runs. At most eight jobs are retained in `.rig-jobs/`; older completed results are removed when new jobs are submitted. Download results you want to keep. Job handles reset when the server restarts.

SkinTokens is distributed under its [MIT license](https://github.com/VAST-AI-Research/SkinTokens/blob/main/LICENSE). Its code, runtime and model files remain outside the RLsim package.

## Tested installation notes

The verified Linux/WSL setup uses Python 3.11, PyTorch 2.7.0/cu128, Blender `bpy` 4.2, Transformers 4.57.x and FlashAttention 2.8.3. Blender 4.2 wheels do not support Python 3.12. The upstream requirements omit SciPy, needed during import; the command above adds it. An official FlashAttention wheel matching Python, PyTorch and C++ ABI avoids a CUDA toolkit mismatch when compiling from source. See `docs/skintokens-requirements-tested.txt` for the captured environment; it is specific to this platform.

For the local service, change the internal Blender RPC binding in `src/server/bpy_server.py` from `host='0.0.0.0'` to `host='127.0.0.1'`. This is the only source patch in the tested external checkout. The checkpoints and inference algorithm are unchanged.
