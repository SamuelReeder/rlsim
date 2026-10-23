"""Export an editable PyTorch actor as a conventional ONNX controller.

Install the optional exporter: .venv/bin/python -m pip install onnx
Run: .venv/bin/python examples/export_onnx_controller.py --output controller.onnx
For any imported body, first export its controller interface in RLsim, then:
    .venv/bin/python examples/export_onnx_controller.py \
        --interface rlsim-controller-interface.json --output controller.onnx

This example has random UNTRAINED weights. Load your trained actor weights
before export. The controller accepts one float32 observation tensor and emits
one float32 normalized action tensor. The simulator maps actions to motor limits.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import zipfile

import torch
from torch import nn

from morpholab.environment import WalkerEnv
from morpholab.training import TrainConfig


def main():
    parser = argparse.ArgumentParser(description="Export an untrained RLsim ONNX actor template")
    parser.add_argument("--output", type=Path, default=Path("controller.onnx"))
    parser.add_argument("--interface", type=Path, help="Controller interface exported by RLsim for any body/task")
    args = parser.parse_args()
    if args.interface:
        interface = json.loads(args.interface.read_text(encoding="utf-8"))
        if interface.get("format") != "rlsim-controller-interface-v1":
            parser.error("--interface must be an RLsim controller interface JSON file")
        config = interface["config"]
        observations = interface["model_info"]["observation_size"]
        actions = interface["model_info"]["action_size"]
        schema = interface["model_info"]["observation_schema"]
        action_schema = interface["model_info"].get("action_schema", "rlsim-actuator-controls-v1")
        # Imported MJCF/URDF/character bodies are deliberately kept as their
        # complete config, including assets, instead of validated as a leg preset.
        if not config.get("body"):
            parser.error("The controller interface has no body configuration")
    else:
        body_file = Path(__file__).with_name("custom-body.json")
        body = json.loads(body_file.read_text(encoding="utf-8"))
        native_config = TrainConfig(body=body["body"], environment=body["environment"])
        config = native_config.model_dump()
        environment = WalkerEnv(native_config.body, environment=native_config.environment)
        try:
            observations = environment.observation_space.shape[0]
            actions = environment.action_space.shape[0]
        finally:
            environment.close()
        schema = "rlsim-planar-v1"
        action_schema = "rlsim-actuator-controls-v1"
    if not isinstance(observations, int) or not 1 <= observations <= 8192:
        parser.error("Observation count must be between 1 and 8192")
    if not isinstance(actions, int) or not 1 <= actions <= 256:
        parser.error("Action count must be between 1 and 256; the body needs actuators")
    torch.manual_seed(7)
    # ONNX is not restricted to the portable PPO MLP architecture. Replace this
    # module with your trained feed-forward actor; keep the input/output contract.
    actor = nn.Sequential(
        nn.Linear(observations, 48), nn.ReLU(),
        nn.Linear(48, actions), nn.Tanh(),
    ).eval()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        actor, torch.zeros(1, observations), str(args.output),
        input_names=["observations"], output_names=["actions"],
        opset_version=17, dynamo=False,
    )
    manifest = {
        "format": "rlsim-onnx-manifest-v1",
        "model": args.output.name,
        "config": config,
        "name": "Custom ONNX controller (untrained)",
        "input_name": "observations", "output_name": "actions",
        "observation_schema": schema,
        "action_schema": action_schema,
        # The example actor consumes raw observations. Embed your preprocessing
        # in the ONNX graph or replace this with the training mean/variance.
        "normalizer": None, "steps": 0,
    }
    manifest_path = args.output.with_suffix(".manifest.json")
    manifest_path.write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8",
    )
    package = args.output.with_suffix(".zip")
    with zipfile.ZipFile(package, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.write(args.output, args.output.name)
        archive.write(manifest_path, manifest_path.name)
    print(f"Untrained ONNX actor written to {args.output.resolve()}")
    print(f"Import this controller package in RLsim: {package.resolve()}")


if __name__ == "__main__":
    main()
