"""Editable PyTorch controller template exported for RLsim browser inference.

Run from the repository:
    .venv/bin/python examples/custom_controller.py --output custom-controller.json

This creates random, UNTRAINED networks. Replace the networks with your trained
weights and pass the observation statistics used in your training pipeline.
The browser importer accepts the resulting JSON under Model > Import model.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch import nn

from morpholab.environment import WalkerEnv
from morpholab.portable import export_torch_policy
from morpholab.training import TrainConfig


def main():
    parser = argparse.ArgumentParser(description="Export an untrained RLsim controller template")
    parser.add_argument("--output", type=Path, default=Path("custom-controller.json"))
    args = parser.parse_args()
    body_file = Path(__file__).with_name("custom-body.json")
    body = json.loads(body_file.read_text(encoding="utf-8"))
    config = TrainConfig(
        body=body["body"], environment=body["environment"],
        hidden_size=32, hidden_layers=2, total_steps=4096,
    )
    environment = WalkerEnv(config.body, environment=config.environment)
    try:
        observations = environment.observation_space.shape[0]
        actions = environment.action_space.shape[0]
    finally:
        environment.close()
    torch.manual_seed(7)
    actor = nn.Sequential(
        nn.Linear(observations, 32), nn.Tanh(),
        nn.Linear(32, 32), nn.Tanh(), nn.Linear(32, actions),
    )
    critic = nn.Sequential(
        nn.Linear(observations, 32), nn.Tanh(),
        nn.Linear(32, 32), nn.Tanh(), nn.Linear(32, 1),
    )
    export_torch_policy(
        actor, critic, torch.full((actions,), -0.7), config, args.output,
        steps=0, name="Custom controller (untrained)",
        # When importing trained weights, provide observation_mean and
        # observation_variance from that training pipeline here.
    )
    print(f"Untrained controller template written to {args.output.resolve()}")


if __name__ == "__main__":
    main()
