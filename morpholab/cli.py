from __future__ import annotations

import argparse
import json
from pathlib import Path
import time


def main():
    parser = argparse.ArgumentParser(description="RLsim · MuJoCo locomotion with PPO")
    sub = parser.add_subparsers(dest="command")
    serve = sub.add_parser("serve", help="Open the local body editor and training dashboard")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8777)
    serve.add_argument("--runs-dir", type=Path, default=Path("runs"))
    doctor = sub.add_parser("doctor", help="Verify available GPU/CPU training devices")
    doctor.add_argument("--device", choices=["auto", "cuda", "mps", "cpu"], default="auto")
    worker = sub.add_parser("worker", help=argparse.SUPPRESS)
    worker.add_argument("run_dir", type=Path)
    evaluate = sub.add_parser("evaluate", help="Evaluate a saved run on held-out seeds")
    evaluate.add_argument("run_dir", type=Path)
    export_browser = sub.add_parser(
        "export-browser", help="Convert a native checkpoint to portable browser JSON",
    )
    export_browser.add_argument("run_dir", type=Path)
    export_browser.add_argument("output", type=Path)
    export_browser.add_argument("--checkpoint", choices=["latest", "best"], default="latest")
    resume = sub.add_parser("resume", help="Continue a native checkpoint into a new run")
    resume.add_argument("run_dir", type=Path)
    resume.add_argument("--additional-steps", type=int, default=65536)
    rig = sub.add_parser("rig-prepare", help="Prepare a GLB rig with an optional external SkinTokens installation")
    rig.add_argument("input", type=Path, nargs="?", help="Self-contained input GLB (original is preserved)")
    rig.add_argument("--output", type=Path, help="New rigged GLB to import into RLsim")
    rig.add_argument("--checkout", type=Path, help="SkinTokens checkout; or RLSIM_SKINTOKENS_HOME")
    rig.add_argument("--python", type=Path, help="Separate SkinTokens Python; or RLSIM_SKINTOKENS_PYTHON")
    rig.add_argument("--doctor", action="store_true", help="Report installation files without loading ML dependencies")
    rig.add_argument("--check-runtime", action="store_true", help="With --doctor, probe modules and CUDA in the external interpreter")
    rig.add_argument("--dry-run", action="store_true", help="Validate files and print the invocation without running inference")
    rig.add_argument("--use-skeleton", action="store_true", help="Keep the existing skeleton and generate skin weights only")
    rig.add_argument("--no-transfer", action="store_true", help="Disable upstream texture and scale transfer")
    rig.add_argument("--preserve-mesh", action="store_true", help="Keep original geometry/materials and attach predicted weights after a surface check (unrigged GLB only)")
    rig.add_argument("--overwrite", action="store_true", help="Replace an existing output after successful validation")
    rig.add_argument("--timeout", type=float, default=1800, help="Inference timeout in seconds (default: 1800)")
    train = sub.add_parser("train", help="Train headlessly on an available GPU or CPU")
    train.add_argument("--device", choices=["auto", "cuda", "mps", "cpu"], default="auto")
    train.add_argument("--preset", choices=["biped", "quadruped", "hexapod"], default="quadruped")
    train.add_argument("--body-file", type=Path, help="BodyConfig JSON; overrides preset")
    train.add_argument("--steps", type=int, default=262144)
    train.add_argument("--seed", type=int, default=7)
    train.add_argument("--target-speed", type=float, default=1.0)
    train.add_argument("--num-envs", type=int, default=8)
    train.add_argument("--hidden-size", type=int, default=256, help="Units per hidden layer (16–256)")
    train.add_argument("--hidden-layers", type=int, default=2, help="Hidden layers per actor/critic (1–4)")
    train.add_argument("--learning-rate", type=float, default=0.0003)
    train.add_argument("--batch-size", type=int, default=512, help="Maximum PPO minibatch size")
    train.add_argument("--epochs", type=int, default=10, help="PPO update epochs per rollout")
    train.add_argument("--rollout-steps", type=int, default=256, help="Steps per environment per rollout")
    train.add_argument("--gamma", type=float, default=0.99, help="Reward discount")
    train.add_argument("--entropy-coef", type=float, default=0.005)
    train.add_argument("--clip-range", type=float, default=0.2)
    train.add_argument("--target-kl", type=float, default=0.03, help="PPO divergence safeguard; 0 disables")
    train.add_argument("--no-best-checkpoint", action="store_true", help="Skip periodic validation/best selection")
    train.add_argument("--evaluation-interval", type=int, default=16384)
    train.add_argument("--evaluation-episodes", type=int, default=3)
    train.add_argument("--curriculum-mode", choices=["speed", "staged"], default="speed")
    train.add_argument("--no-randomize", action="store_true")
    train.add_argument("--no-curriculum", action="store_true")
    train.add_argument("--runs-dir", type=Path, default=Path("runs"))
    args = parser.parse_args()
    if args.command == "worker":
        from .training import train_worker
        train_worker(args.run_dir)
    elif args.command == "doctor":
        from .training import hardware_info
        print(json.dumps(hardware_info(args.device), indent=2))
    elif args.command == "evaluate":
        from .training import load_checkpoint, evaluate_suite
        config, model, normalizer = load_checkpoint(args.run_dir)
        try:
            print(json.dumps(evaluate_suite(config, model, normalizer), indent=2))
        finally:
            normalizer.close()
    elif args.command == "export-browser":
        from .portable import export_checkpoint
        export_checkpoint(args.run_dir, args.output, checkpoint=args.checkpoint)
        print(f"Browser model saved to {args.output.resolve()}", flush=True)
    elif args.command == "rig-prepare":
        from .rig_prepare import RigPreparationError, installation_status, prepare_rig
        try:
            if args.doctor:
                result = installation_status(args.checkout, args.python, check_runtime=args.check_runtime)
                print(json.dumps(result, indent=2))
                if result["issues"]:
                    raise SystemExit(1)
            else:
                if args.input is None or args.output is None:
                    parser.error("rig-prepare requires INPUT and --output, or --doctor")
                if args.check_runtime:
                    parser.error("--check-runtime is a doctor option; normal preparation already checks the runtime")
                result = prepare_rig(
                    args.input, args.output, checkout=args.checkout, python=args.python,
                    use_skeleton=args.use_skeleton, transfer=not args.no_transfer,
                    preserve_mesh=args.preserve_mesh,
                    overwrite=args.overwrite, timeout=args.timeout, dry_run=args.dry_run,
                    on_started=lambda info: print(
                        f"SkinTokens is running. Log: {info['log']}"
                        + ("\n" + "\n".join(info["warnings"]) if info["warnings"] else ""), flush=True,
                    ),
                )
                print(json.dumps(result, indent=2))
        except RigPreparationError as exc:
            parser.exit(1, f"Rig preparation: {exc}\n")
        except KeyboardInterrupt:
            parser.exit(130, "Rig preparation cancelled.\n")
    elif args.command == "resume":
        from .server import RunManager
        path = args.run_dir.resolve()
        manager = RunManager(path.parent)
        state = manager.resume(path.name, args.additional_steps)
        print(f"Continued run: {manager.resolve(state['run_id'])}", flush=True)
        try:
            while state['status'] in {'starting', 'training', 'stopping'}:
                time.sleep(2)
                state = manager.status()
                print(json.dumps({k: state.get(k) for k in ['status', 'steps', 'total_steps', 'error']}), flush=True)
        except KeyboardInterrupt:
            manager.stop()
        finally:
            manager.close()
        if state['status'] == 'failed':
            raise SystemExit(1)
    elif args.command == "train":
        from .config import BodyConfig, PRESETS
        from .server import RunManager
        from .training import TrainConfig
        body = (BodyConfig.model_validate_json(args.body_file.read_text())
                if args.body_file else PRESETS[args.preset])
        manager = RunManager(args.runs_dir.resolve())
        state = manager.start(TrainConfig(
            body=body, total_steps=args.steps, seed=args.seed,
            target_speed=args.target_speed, n_envs=args.num_envs,
            randomize=not args.no_randomize, curriculum=not args.no_curriculum,
            device=args.device,
            hidden_size=args.hidden_size, hidden_layers=args.hidden_layers,
            learning_rate=args.learning_rate, batch_size=args.batch_size,
            epochs=args.epochs, rollout_steps=args.rollout_steps,
            gamma=args.gamma, entropy_coef=args.entropy_coef, clip_range=args.clip_range,
            target_kl=args.target_kl, best_checkpoint=not args.no_best_checkpoint,
            evaluation_interval=args.evaluation_interval,evaluation_episodes=args.evaluation_episodes,
            curriculum_mode=args.curriculum_mode,
        ))
        print(f"Run saved to {manager.resolve(state['run_id'])}", flush=True)
        try:
            while True:
                state = manager.status()
                print(json.dumps({k: state.get(k) for k in
                                  ["status", "steps", "total_steps", "elapsed", "metrics", "error"]}),
                      flush=True)
                if state["status"] not in {"starting", "training", "stopping"}:
                    break
                time.sleep(5)
        except KeyboardInterrupt:
            manager.stop()
            print("Stopping and saving checkpoint…", flush=True)
        finally:
            manager.close()
        if state["status"] == "failed":
            raise SystemExit(1)
    else:
        import uvicorn
        from .server import create_app
        uvicorn.run(create_app(getattr(args, "runs_dir", Path("runs"))),
                    host=getattr(args, "host", "127.0.0.1"), port=getattr(args, "port", 8777))


if __name__ == "__main__":
    main()
