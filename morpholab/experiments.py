"""Reproducible, bounded native curriculum comparison; all trials are retained."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
import uuid
import numpy as np
from .config import PRESETS
from .training import TrainConfig, atomic_json, read_json, load_checkpoint, evaluate_policy

EVALUATION_SEEDS = [500000007, 500000021, 500000039]


def summarize(episodes):
    return {"survival_rate": float(np.mean([not e["fell"] for e in episodes])),
            **{key: float(np.mean([e[key] for e in episodes])) for key in
               ["speed_error", "distance", "seconds", "return"]}}


def run_comparison(output, bodies, seeds, steps, device="auto"):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    protocol = {"bodies": bodies, "training_seeds": seeds, "evaluation_seeds": EVALUATION_SEEDS,
                "steps": steps, "conditions": ["No curriculum", "Gradual speed"],
                "checkpoint": "best", "device": device, "randomize": True,
                "note": "Fixed-budget pilot; final evaluation seeds are separate from checkpoint selection. Failures are retained. Compare within each body. Native physics, not a browser throughput benchmark."}
    report_path = output / "report.json"
    if report_path.exists():
        report = read_json(report_path)
        if report["protocol"] != protocol:
            raise ValueError("Existing report uses a different protocol. Choose a new output directory.")
    else:
        report = {"format": "rlsim-experiment-report-v1", "title": "Curriculum pilot · 65k steps" if steps == 65536 else "Curriculum comparison",
                  "created_at": datetime.now(timezone.utc).isoformat(), "protocol": protocol, "trials": []}
    for body in bodies:
        for seed in seeds:
            for condition in protocol["conditions"]:
                key = f"{body}-{seed}-{condition}"
                previous = next((t for t in report["trials"] if t["id"] == key), None)
                if previous and previous["status"] in ["completed", "failed"]:
                    continue
                config = TrainConfig(body=PRESETS[body], seed=seed, total_steps=steps, device=device,
                                     hidden_size=64, epochs=4, rollout_steps=128, batch_size=256, n_envs=8,
                                     curriculum=condition == "Gradual speed", randomize=True,
                                     evaluation_interval=32768, evaluation_episodes=3)
                run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + "-" + uuid.uuid4().hex[:8]
                folder = output / "runs" / run_id
                folder.mkdir(parents=True)
                atomic_json(folder / "config.json", config.model_dump())
                atomic_json(folder / "status.json", {"status": "starting", "run_id": run_id, "history": [], "steps": 0,
                            "created_at": datetime.now(timezone.utc).isoformat(), "config": config.model_dump()})
                trial = {"id": key, "body": body, "seed": seed, "condition": condition, "status": "training",
                         "run_id": run_id, "config": config.model_dump(), "budget": config.actual_steps,
                         "source_hash": hashlib.sha256(Path(__file__).with_name("training.py").read_bytes()).hexdigest()}
                if previous:
                    report["trials"].remove(previous)
                report["trials"].append(trial)
                atomic_json(report_path, report)
                start = time.monotonic()
                print(f"Training {key}", flush=True)
                with (folder / "worker.log").open("w") as log:
                    process = subprocess.Popen([sys.executable, "-m", "morpholab.cli", "worker", str(folder)], stdout=log, stderr=subprocess.STDOUT)
                    try:
                        code = process.wait()
                    except KeyboardInterrupt:
                        (folder / "STOP").touch()
                        process.wait(timeout=60)
                        raise
                state = read_json(folder / "status.json")
                trial.update(status=state["status"], steps=state.get("steps", 0), training_seconds=state.get("elapsed", time.monotonic()-start), hardware=state.get("hardware"))
                if code or state["status"] != "completed":
                    trial.update(status="failed", error=state.get("error", f"Worker exit {code}"))
                else:
                    normalizer = None
                    try:
                        config, model, normalizer = load_checkpoint(folder, "best")
                        episodes = [evaluate_policy(config, model, normalizer, seed=s)["summary"] for s in EVALUATION_SEEDS]
                        trial.update(episodes=episodes, metrics=summarize(episodes), checkpoint_steps=int(model.num_timesteps))
                    except Exception as exc:
                        trial.update(status="failed", error=f"Final evaluation: {type(exc).__name__}: {exc}")
                    finally:
                        if normalizer is not None:
                            normalizer.close()
                atomic_json(report_path, report)
                print(json.dumps({k:trial.get(k) for k in ["id", "status", "metrics", "training_seconds"]}), flush=True)
    report["completed_at"] = datetime.now(timezone.utc).isoformat()
    atomic_json(report_path, report)
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--bodies", nargs="+", choices=list(PRESETS), default=list(PRESETS))
    p.add_argument("--seeds", nargs="+", type=int, default=[7,17,27,37,47])
    p.add_argument("--steps", type=int, default=65536)
    p.add_argument("--device", choices=["auto","cpu","cuda","mps"], default="auto")
    args = p.parse_args()
    if len(set(args.seeds)) != len(args.seeds) or len(set(args.bodies)) != len(args.bodies):
        p.error("Seeds and bodies must be unique.")
    TrainConfig(total_steps=args.steps, seed=min(args.seeds))
    for seed in args.seeds:
        TrainConfig(seed=seed)
    run_comparison(args.output, args.bodies, args.seeds, args.steps, args.device)


if __name__ == "__main__":
    main()
