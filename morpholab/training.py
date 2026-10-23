"""Isolated PPO worker, automatic device selection and reproducible checkpoints."""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
import time
import traceback
from typing import Any, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator

from .config import BodyConfig, EnvironmentConfig
from .tasks import TaskConfig
from .morphology import ProportionTraining


class TrainConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    body: BodyConfig = Field(default_factory=BodyConfig)
    morphology: ProportionTraining | None = None
    environment: EnvironmentConfig = Field(default_factory=EnvironmentConfig)
    task: TaskConfig | None = None
    total_steps: int = Field(default=65536, ge=4096, le=2_000_000)
    target_speed: float = Field(default=1.0, ge=0.1, le=3.0)
    randomize: bool = True
    curriculum: bool = True
    curriculum_mode: Literal["speed", "staged"] = "speed"
    seed: int = Field(default=7, ge=0, le=2**31 - 1)
    n_envs: int = Field(default=8, ge=1, le=32)
    device: Literal["auto", "cuda", "mps", "cpu"] = "auto"
    hidden_size: int = Field(default=256, ge=16, le=256)
    hidden_layers: int = Field(default=2, ge=1, le=4)
    learning_rate: float = Field(default=0.0003, ge=0.00001, le=0.01)
    batch_size: int = Field(default=512, ge=16, le=2048)
    epochs: int = Field(default=10, ge=1, le=20)
    rollout_steps: int = Field(default=256, ge=16, le=1024)
    gamma: float = Field(default=0.99, ge=0.8, le=0.9999)
    entropy_coef: float = Field(default=0.005, ge=0, le=0.1)
    clip_range: float = Field(default=0.2, ge=0.05, le=0.4)
    target_kl: float = Field(default=0.03, ge=0, le=1)
    best_checkpoint: bool = True
    evaluation_interval: int = Field(default=16384, ge=4096, le=2_000_000)
    evaluation_episodes: int = Field(default=3, ge=1, le=8)

    @model_validator(mode="after")
    def validate_curriculum(self):
        if self.morphology:
            from .morphology import body_at
            # Reject ranges that would leave the editor's physical bounds.
            body_at(self.body, [1-self.morphology.width]*3)
            body_at(self.body, [1+self.morphology.width]*3)
        if self.curriculum_mode == "staged":
            if self.task is None:
                self.task = TaskConfig()
            if self.curriculum and self.task.preset != "velocity":
                raise ValueError("Staged locomotion curriculum requires the Velocity objective.")
        return self

    @property
    def actual_steps(self) -> int:
        rollout = self.n_envs * self.rollout_steps
        return math.ceil(self.total_steps / rollout) * rollout


def atomic_json(path: Path, value: Any) -> None:
    temp = path.with_suffix(f".{os.getpid()}.tmp")
    temp.write_text(json.dumps(value, allow_nan=False), encoding="utf-8")
    temp.replace(path)


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _probe_device(device: str) -> None:
    """Verify the forward/backward operations needed by training."""
    import torch

    # Device enumeration alone does not establish that kernels work with the
    # installed PyTorch build (for example, a new NVIDIA architecture).
    probe = torch.ones((32, 32), device=device, requires_grad=True)
    result = probe @ probe
    result.sum().backward()
    if device == "cuda":
        torch.cuda.synchronize()
    elif device == "mps":
        torch.mps.synchronize()
    if float(result[0, 0].detach().cpu()) != 32:
        raise RuntimeError(f"{device.upper()} kernel verification failed.")
    if probe.grad is None or not bool(torch.isfinite(probe.grad).all().cpu()):
        raise RuntimeError(f"{device.upper()} gradient verification failed.")


def hardware_info(requested_device: str = "auto") -> dict:
    """Choose a usable device before training; never restart a failed run."""
    import torch

    if requested_device not in {"auto", "cuda", "mps", "cpu"}:
        raise ValueError("Device must be auto, cuda, mps, or cpu.")
    errors = []
    available = {}
    for device in ("cuda", "mps"):
        try:
            backend = torch.cuda if device == "cuda" else getattr(torch.backends, "mps", None)
            available[device] = bool(backend and backend.is_available())
        except Exception as exc:
            available[device] = False
            errors.append(f"{device.upper()} discovery failed: {exc}")
    hardware = {
        "cuda_available": available["cuda"],
        "mps_available": available["mps"],
        "gpu_name": None,
        "torch_version": str(torch.__version__),
        "cuda_version": torch.version.cuda,
        "physics_device": "cpu",
        "execution_location": "server",
        "requested_device": requested_device,
        "fallback_reason": None,
    }
    candidates = ("cuda", "mps", "cpu") if requested_device == "auto" else (requested_device,)
    for device in candidates:
        try:
            if device != "cpu" and not available[device]:
                raise RuntimeError(f"{device.upper()} is unavailable")
            _probe_device(device)
            gpu_name = (torch.cuda.get_device_name(0) if device == "cuda" else
                        "Apple GPU" if device == "mps" else None)
        except Exception as exc:
            reason = f"{device.upper()}: {exc}"
            if requested_device != "auto" or device == "cpu":
                advice = {
                    "cuda": "Install CUDA-enabled PyTorch and a compatible NVIDIA driver",
                    "mps": "Use a supported Apple device with an MPS-enabled PyTorch build",
                    "cpu": "Check the PyTorch installation",
                }[device]
                raise RuntimeError(
                    f"Cannot train on {device.upper()}. {advice}, or choose device='auto'. {exc}"
                ) from exc
            errors.append(reason)
            continue
        hardware.update(
            training_device=device, gpu_name=gpu_name,
            fallback_reason="; ".join(errors) if requested_device == "auto" and errors else None,
        )
        return hardware
    raise RuntimeError("No usable training device found.")


def require_cuda() -> dict:
    """Compatibility wrapper for callers explicitly requesting CUDA."""
    return hardware_info("cuda")


def replay_context(config: TrainConfig, seed: int = 10007) -> dict:
    """Describe the actual body and world used for an inference episode."""
    return {
        "body": config.body.model_dump(),
        "environment": config.environment.model_dump(),
        "target_speed": config.target_speed,
        "task": config.task.model_dump(exclude_none=True) if config.task else None,
        "seed": seed,
        "config": config.model_dump(),
    }


def evaluate_policy(
    config: TrainConfig, model=None, normalizer=None, *, seed: int = 10007,
    capture: bool = False, randomized: bool = False, cancel=None,
) -> dict:
    """Evaluate a full episode; zero actions mean rest targets for position control."""
    from .environment import WalkerEnv

    if cancel and cancel():
        return {"cancelled": True, "frames": []}
    env = WalkerEnv(
        config.body, target_speed=config.target_speed, randomize=randomized,
        environment=config.environment,
        task=config.task,
    )
    try:
        env.set_training_progress(1.0)
        obs, _ = env.reset(seed=seed)
        start_x = float(env.data.qpos[0])
        frames = [env.get_frame()] if capture else []
        total_reward, speeds, errors, efforts = 0.0, [], [], []
        terminated = truncated = False
        stopped = False
        fell_any = False
        for index in range(env.max_episode_steps):
            if cancel and cancel():
                stopped = True
                break
            if model is None:
                action = np.zeros(env.action_space.shape, dtype=np.float32)
            else:
                policy_obs = normalizer.normalize_obs(obs.copy()) if normalizer else obs
                action, _ = model.predict(policy_obs, deterministic=True)
            if cancel and cancel():
                stopped = True
                break
            obs, reward, terminated, truncated, info = env.step(action)
            fell_any |= bool(info.get("fallen", terminated))
            speed = float(env.data.qvel[0])
            total_reward += float(reward)
            speeds.append(speed)
            errors.append(abs(speed - config.target_speed))
            efforts.append(float(np.mean(np.square(action))))
            if capture and (index % 2 == 1 or terminated or truncated):
                frames.append(env.get_frame())
            if terminated or truncated:
                break
        if stopped and capture and frames[-1]["time"] != float(env.data.time):
            frames.append(env.get_frame())
        summary = {
            "distance": float(env.data.qpos[0]) - start_x,
            "mean_speed": float(np.mean(speeds)) if speeds else 0.0,
            "speed_error": float(np.mean(errors)) if errors else 0.0,
            "mean_squared_action": float(np.mean(efforts)) if efforts else 0.0,
            "return": total_reward,
            "steps": len(speeds),
            "seconds": float(env.data.time),
            "fell": fell_any,
            "seed": seed,
            "randomized": randomized,
            "gait": env.motion.summary(),
            "effort": env.effort.summary(float(env.data.qpos[0])-start_x, float(env.model.body_mass.sum()), config.environment.gravity),
        }
        model_info = env.model_info()
        return {
            "frames": frames, "summary": summary, "frame_dt": 0.04,
            **({"cancelled": True} if stopped else {}),
            **replay_context(config, seed),
            "model_info": model_info,
        }
    finally:
        env.close()


def evaluate_suite(config: TrainConfig, model=None, normalizer=None) -> dict:
    # Identical held-out seeds before and after. Never selects a lucky replay.
    nominal = [
        evaluate_policy(config, model, normalizer, seed=10007 + i)["summary"]
        for i in range(3)
    ]
    perturbed = [
        evaluate_policy(config, model, normalizer, seed=20007 + i, randomized=True)["summary"]
        for i in range(3)
    ]

    def aggregate(episodes):
        keys = ["distance", "mean_speed", "speed_error", "return", "seconds", "mean_squared_action"]
        return {**{k: float(np.mean([e[k] for e in episodes])) for k in keys},
                "survival_rate": float(np.mean([not e["fell"] for e in episodes])),
                "episodes": episodes}

    return {"nominal": aggregate(nominal), "randomized": aggregate(perturbed)}


def load_checkpoint(run_dir: Path, checkpoint: str = "latest"):
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
    from .environment import WalkerEnv

    if checkpoint not in {"best", "latest"}:
        raise ValueError("Checkpoint must be best or latest.")
    prefix = "best_" if checkpoint == "best" and (run_dir / "best_policy.zip").exists() else ""
    config = TrainConfig.model_validate(read_json(run_dir / "config.json"))
    env = DummyVecEnv([lambda: WalkerEnv(
        config.body, target_speed=config.target_speed, environment=config.environment, task=config.task,
    )])
    normalized = VecNormalize.load(str(run_dir / f"{prefix}normalize.pkl"), env)
    normalized.training = False
    normalized.norm_reward = False
    # Replay doesn't update network weights; CPU inference avoids a second GPU job.
    model = PPO.load(str(run_dir / f"{prefix}policy.zip"), device="cpu")
    return config, model, normalized


def replay_run(
    run_dir: Path, *, environment: EnvironmentConfig | dict | None = None,
    target_speed: float | None = None, seed: int | None = None,
    checkpoint: str = "best", cancel=None,
) -> dict:
    if cancel and cancel():
        return {"cancelled": True, "frames": []}
    config, model, normalizer = load_checkpoint(run_dir, checkpoint)
    try:
        # Keep the checkpoint's body and normalization shape. World settings
        # can change without adding observations or altering policy dimensions.
        values = config.model_dump()
        if environment is not None:
            values["environment"] = (
                environment.model_dump()
                if isinstance(environment, EnvironmentConfig) else environment
            )
        if target_speed is not None:
            values["target_speed"] = target_speed
        if seed is not None:
            values["seed"] = seed
        config = TrainConfig.model_validate(values)
        best = read_json(run_dir / "best.json") if checkpoint == "best" and (run_dir / "best.json").exists() else None
        return {
            **evaluate_policy(
                config, model, normalizer, capture=True,
                seed=10007 if seed is None else seed,
                **({"cancel": cancel} if cancel else {}),
            ),
            "source": "trained",
            "checkpoint": "best" if best else "latest",
            "checkpoint_steps": best["steps"] if best else int(model.num_timesteps),
        }
    finally:
        normalizer.close()


def train_worker(run_dir: Path) -> None:
    """Run in a separate process so the UI remains responsive during updates."""
    import torch
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
    from .environment import WalkerEnv, export_mjcf

    config = TrainConfig.model_validate(read_json(run_dir / "config.json"))
    state = read_json(run_dir / "status.json")
    started = time.monotonic()
    vec = None
    resume = read_json(run_dir / "resume.json") if (run_dir / "resume.json").exists() else None
    from .continuation import capture, read_state, restore_random
    try:
        saved = read_state(run_dir) if resume else None
        initial_steps = saved["steps"] if saved else 0
        schedule_steps = saved["schedule_steps"] if saved else config.actual_steps
        torch.set_num_threads(2)
        hardware = hardware_info(config.device)
        device = hardware["training_device"]
        state.update(hardware=hardware)
        atomic_json(run_dir / "status.json", state)
        from .morphology import ProportionWalker, body_at, validation_scales, validation_cases
        env_type = ProportionWalker if config.morphology and config.morphology.enabled else WalkerEnv
        envs = [
            env_type(
                config.body, target_speed=config.target_speed, randomize=config.randomize,
                **({"proportions": config.morphology} if env_type is ProportionWalker else {}),
                environment=config.environment,
                task=config.task,
            )
            for _ in range(config.n_envs)
        ]
        from .curriculum import LearningCurriculum
        curriculum = LearningCurriculum(config.target_speed) if config.curriculum and config.curriculum_mode == "staged" else None
        if curriculum and saved:
            curriculum.index = saved["curriculum"]["index"]
            curriculum.episodes = saved["curriculum"]["episodes"]
        for env in envs:
            if isinstance(env, ProportionWalker):
                env.set_variation_steps(initial_steps)
            if curriculum:
                env.set_learning_stage(curriculum.stage)
            else:
                env.set_training_progress(min(1.0, initial_steps / (schedule_steps * .6)) if config.curriculum else 1.0)
        if curriculum:
            state["curriculum"] = curriculum.snapshot()
        if saved:
            raw_vec = DummyVecEnv([lambda env=env: Monitor(env) for env in envs])
            vec = VecNormalize.load(str(run_dir / "normalize.pkl"), raw_vec)
            vec.training = True
            vec.norm_reward = True
            vec.returns = np.zeros(config.n_envs)
            model = PPO.load(str(run_dir / "policy.zip"), env=vec, device=device, force_reset=True)
            if model.num_timesteps != initial_steps:
                raise ValueError("Saved optimizer and training step counts differ.")
            raw_vec._seeds = [None] * config.n_envs
        else:
            vec = VecNormalize(
                DummyVecEnv([lambda env=env: Monitor(env) for env in envs]),
                norm_obs=True, norm_reward=True, clip_obs=10.0, gamma=config.gamma,
            )
            network = [config.hidden_size] * config.hidden_layers
            model = PPO(
                "MlpPolicy", vec, device=device, seed=config.seed,
                n_steps=config.rollout_steps,
                batch_size=min(config.batch_size, config.n_envs * config.rollout_steps),
                n_epochs=config.epochs,
                learning_rate=config.learning_rate, gamma=config.gamma,
                gae_lambda=0.95, clip_range=config.clip_range,
                ent_coef=config.entropy_coef, target_kl=config.target_kl or None,
                policy_kwargs={"net_arch": {"pi": network, "vf": network.copy()},
                               "log_std_init": -0.7},
                verbose=0,
            )
        if next(model.policy.parameters()).device.type != device:
            raise RuntimeError(f"Policy was not placed on the selected {device.upper()} device.")
        (run_dir / "body.xml").write_text(
            export_mjcf(config.body, config.environment), encoding="utf-8"
        )
        state.update(status="training", hardware=hardware, policy_device=str(model.device),
                     total_steps=initial_steps + config.actual_steps, frame=envs[0].get_frame(),
                     model_info=envs[0].model_info())
        atomic_json(run_dir / "status.json", state)
        baseline = evaluate_suite(config)
        atomic_json(run_dir / "baseline.json", baseline)
        if saved:
            restore_random(saved, envs)

        class Progress(BaseCallback):
            last_write = 0.0
            last_record = 0
            last_save = 0
            last_evaluation = 0
            episode_stats = [{"steps": 0, "error": 0., "fell": False} for _ in envs]

            def validate(self):
                state["phase"] = "evaluating"
                atomic_json(run_dir / "status.json", state)
                episodes = [
                    evaluate_policy(validation_cases(config)[i % len(validation_cases(config))], model, vec,
                                    seed=(config.seed + 1000003 + i * 997) % (2**31 - 1))["summary"]
                    for i in range(config.evaluation_episodes)
                ]
                evaluation = {
                    "mean_return": float(np.mean([e["return"] for e in episodes])),
                    **{key: float(np.mean([e[key] for e in episodes]))
                       for key in ("seconds", "speed_error", "mean_squared_action", "distance")},
                    "survival_rate": float(np.mean([not e["fell"] for e in episodes])),
                    "episodes": episodes,
                }
                if not state.get("best_validation") or evaluation["mean_return"] > state["best_validation"]["mean_return"]:
                    model.save(str(run_dir / "best_policy.zip"))
                    vec.save(str(run_dir / "best_normalize.pkl"))
                    atomic_json(run_dir / "best.json", {"steps": model.num_timesteps, "validation": evaluation})
                    state.update(best_validation=evaluation, best_checkpoint_steps=model.num_timesteps)
                state.update(validation=evaluation, phase="training")
                self.last_evaluation = model.num_timesteps
                atomic_json(run_dir / "status.json", state)

            def publish(self, record=False):
                now = time.monotonic()
                episodes = list(self.model.ep_info_buffer)
                reward = float(np.mean([e["r"] for e in episodes])) if episodes else None
                length = float(np.mean([e["l"] for e in episodes])) if episodes else None
                speed = float(np.mean([e.data.qvel[0] for e in envs]))
                state.update(
                    steps=self.num_timesteps, elapsed=now - started,
                    metrics={"fps": (self.num_timesteps - initial_steps) / max(now - started, .001),
                             "mean_reward": reward, "mean_length": length, "speed": speed,
                             "target_speed": envs[0].target_speed,
                             "gpu_memory_mb": (
                                 torch.cuda.memory_allocated() / 1024**2 if device == "cuda" else 0
                             )},
                    frame=envs[0].get_frame(),
                )
                if record:
                    state["history"].append({
                        "step": self.num_timesteps, "reward": reward, "speed": speed,
                        "length": length,
                    })
                    # Bound network traffic for very long runs.
                    if len(state["history"]) > 1000:
                        state["history"] = state["history"][::2]
                atomic_json(run_dir / "status.json", state)
                self.last_write = now

            def _on_step(self) -> bool:
                if (run_dir / "STOP").exists():
                    return False
                if config.morphology and config.morphology.enabled:
                    for env in envs:
                        env.set_variation_steps(self.num_timesteps)
                if curriculum:
                    for i, (info, done) in enumerate(zip(self.locals["infos"], self.locals["dones"])):
                        stats = self.episode_stats[i]
                        stats["steps"] += 1
                        stats["error"] += abs(info["velocity"] - config.target_speed * curriculum.stage["speed_fraction"])
                        stats["fell"] |= bool(info.get("fallen", False))
                        if done:
                            curriculum.record(fell=stats["fell"], speed_error=stats["error"] / stats["steps"])
                            self.episode_stats[i] = {"steps": 0, "error": 0., "fell": False}
                elif config.curriculum:
                    progress = min(1.0, self.num_timesteps / (schedule_steps * .6))
                    for env in envs:
                        env.set_training_progress(progress)
                if time.monotonic() - self.last_write > .5:
                    self.publish()
                return True

            def _on_rollout_start(self) -> None:
                # At rollout start, the preceding PPO optimization is complete.
                if curriculum:
                    if curriculum.advance():
                        for env in envs:
                            env.set_learning_stage(curriculum.stage)
                        self.episode_stats = [{"steps": 0, "error": 0., "fell": False} for _ in envs]
                        self.model._last_obs = vec.reset()
                        self.model._last_original_obs = vec.get_original_obs()
                        self.model._last_episode_starts = np.ones(config.n_envs, dtype=bool)
                    state["curriculum"] = curriculum.snapshot()
                if self.num_timesteps:
                    self.publish(record=True)
                if config.best_checkpoint and self.num_timesteps - self.last_evaluation >= config.evaluation_interval:
                    self.validate()
                if self.num_timesteps - self.last_save >= 32768:
                    save_checkpoint()
                    self.last_save = self.num_timesteps

        def save_checkpoint():
            model.save(str(run_dir / ".policy.tmp.zip"))
            vec.save(str(run_dir / ".normalize.tmp.pkl"))
            os.replace(run_dir / ".policy.tmp.zip", run_dir / "policy.zip")
            os.replace(run_dir / ".normalize.tmp.pkl", run_dir / "normalize.pkl")
            atomic_json(run_dir / "learning_state.json", capture(run_dir, model, envs, curriculum, schedule_steps))
            state["resumable"] = True
            state["checkpoint_available"] = True
            state["checkpoint_steps"] = model.num_timesteps

        progress = Progress()
        progress.last_save = progress.last_evaluation = initial_steps
        model.learn(total_timesteps=config.actual_steps, callback=progress, reset_num_timesteps=not bool(saved))
        if config.best_checkpoint and not (run_dir / "STOP").exists() and progress.last_evaluation != model.num_timesteps:
            progress.validate()
        save_checkpoint()
        progress.publish(record=True)
        vec.training = False
        vec.norm_reward = False
        result = evaluate_suite(config, model, vec)
        atomic_json(run_dir / "evaluation.json", {
            "baseline": baseline, "trained": result,
            "note": "Zero-torque baseline and deterministic policy, 3 identical held-out seeds per condition.",
        })
        state.update(
            status="stopped" if (run_dir / "STOP").exists() else "completed",
            steps=model.num_timesteps, elapsed=time.monotonic() - started,
            evaluation={"baseline": baseline["nominal"], "trained": result["nominal"],
                        "randomized": result["randomized"]},
        )
        # Persist a true trained replay as the final snapshot.
        final_replay = evaluate_policy(config, model, vec, capture=True)
        state["frame"] = final_replay["frames"][0]
        atomic_json(run_dir / "replay.json", {**final_replay, "source": "trained",
                                            "body": config.body.model_dump()})
        atomic_json(run_dir / "status.json", state)
    except Exception as exc:
        state.update(status="failed", error=f"{type(exc).__name__}: {exc}",
                     elapsed=time.monotonic() - started)
        atomic_json(run_dir / "status.json", state)
        traceback.print_exc()
    finally:
        if vec is not None:
            vec.close()
