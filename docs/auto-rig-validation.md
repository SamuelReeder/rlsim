# Automatic body setup: measured validation

Tested on 2026-10-21 with the real pretrained SkinTokens/TokenRig models, an NVIDIA RTX 5080, and the pinned upstream revision `273b691d35989d71cd17ff2895fdc735097b92d1`. Physics checks ran in the browser's MuJoCo WASM worker. This is a body-setup test; no locomotion policies were trained for these examples.

| Input | Predicted joints | Motor controls | Generation time | Physics outcome |
| --- | ---: | ---: | ---: | --- |
| Giraffe | 47 | 138 | 75 seconds | Finite simulation; all motors respond; review generic joint assumptions and collider overlaps |
| Humanoid | 28 | 81 | 60 seconds | Finite simulation; all motors respond; review generic joint assumptions and collider overlaps |
| Six-legged crawler, selected second prediction | 14 | 39 | 37 seconds | Finite simulation; all motors respond; review generic joint assumptions |

Times include loading the inference model and writing the preserved output; they exclude installation/downloads and the initial runtime check. Other hardware, input complexity and sampling can change them. Test masses were illustrative: 60 kg for the humanoid and 10 kg for the other shapes. They are not measurements of those characters.

All three selected results retained the original geometry, materials and embedded assets. The predicted/rest vertex-cloud comparison had a measured maximum distance of zero for these inputs. Preservation copies the original buffer bytes and only appends skinning and skeleton data. An additional transformed-mesh test exercises the actual Three.js skinning/rest-pose path, not just GLB container validity.

## The failed first crawler prediction

The first six-legged input produced 12 joints in 46 seconds. It preserved the visual model but missed one rear leg branch. The source-symmetry check flags joints without corresponding branches on the opposite side. A second independent generation produced 14 joints and the six expected leg chains; that result is bundled as the prepared example. The first output and report remain in `experiments/auto-rig-v1/hexapod-attempt1-*` for comparison.

This is a documented retry, not a claim of perfect first-attempt rigging. The symmetry heuristic can miss mistakes, does not identify anatomical roles, and is not a general limb-count detector. Asymmetric characters need direct inspection.

## What was checked

- GLB input/output validity, normalized skin weights and rest-surface agreement.
- Original position/index/normal/UV/color/image/material retention; source file unchanged.
- Physical graph validity and MuJoCo compilation.
- Mass ratios, initial floor penetration and non-neighbor collider overlaps.
- Positive and negative small motor commands in zero gravity for every control.
- Two seconds of finite passive simulation without solver resets. Falling without a learned controller is reported separately.
- Real UI submission to the local GPU service, result import, physics checks and availability of training afterward.
- Desktop and 390-pixel layouts; static-hosted prepared example loading.
- Rejected shape changes and invalid skin weights; cancellation, malformed uploads, unavailable setup and job limits.

The physics reports intentionally say **Review**. SkinTokens names these bones generically; they therefore use generic physical joint assumptions. An animation skeleton does not establish a knee hinge, correct ankle range, realistic mass, motor strength or successful walking. Joint roles and collision approximations remain editable in **Advanced physics**. The humanoid and crawler fell during their passive tests; this was not presented as a numerical setup error. Self-contact stays disabled.

## Reproduce and inspect

Use **Body → Optional AI rigging → Prepared examples** to load a result, review scale/mass/torque, and import it. Physics checks run on the actual imported body. No external helper or training run is needed to inspect those samples.

For new input, follow [the setup and usage guide](skintokens.md). `scripts/make-rig-test-inputs.py` creates the original unrigged humanoid and crawler using trimesh in the separate preparation environment. `scripts/audit-auto-rigs.py` imports completed results into the actual browser app and saves diagnostic reports and screenshots. The complete local evidence is in `experiments/auto-rig-v1/`; `morpholab/static/demos/auto-rig-study.json` contains the portable results, source/output hashes, and first-attempt failure record.

The giraffe source is the example in the [MIT-licensed SkinTokens repository](https://github.com/VAST-AI-Research/SkinTokens/tree/273b691d35989d71cd17ff2895fdc735097b92d1/examples). Its license accompanies the bundled asset. The humanoid and crawler inputs were authored procedurally for RLsim. All predicted skeletons/weights came from real TokenRig inference, not hand-authored test skeletons.

The external checkout has one local-only source change: its Blender RPC listens on `127.0.0.1`. Model weights and generation logic are unchanged. Generation uses upstream defaults and stochastic sampling; output hashes identify the specific tested predictions.
