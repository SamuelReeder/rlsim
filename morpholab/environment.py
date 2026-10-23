"""MuJoCo locomotion with editable morphology and a Gymnasium interface.

This is a planar control problem in a 3D rigid-body simulator. The torso is
restricted to x/z translation and y-axis rotation. Contacts, inertia, motors,
and gravity are integrated by MuJoCo; the browser only draws simulator state.
"""

from __future__ import annotations

import math
from typing import Any
from xml.sax.saxutils import escape

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces

from .config import BodyConfig, EnvironmentConfig
from .tasks import TaskConfig, evaluate_task
from .effort import EffortMeter
from .motion import MotionTracker, phase_targets, position_controls, walking_components


PHYSICS_TIMESTEP = 0.004
FRAME_SKIP = 5
EPISODE_STEPS = 1000


def _as_body(body: BodyConfig | dict | None) -> BodyConfig:
    if body is None:
        return BodyConfig()
    if isinstance(body, BodyConfig):
        return body.model_copy(deep=True)
    return BodyConfig.model_validate(body)


def _as_environment(
    environment: EnvironmentConfig | dict | None,
) -> EnvironmentConfig:
    if environment is None:
        return EnvironmentConfig()
    if isinstance(environment, EnvironmentConfig):
        return environment.model_copy(deep=True)
    return EnvironmentConfig.model_validate(environment)


def export_mjcf(
    body: BodyConfig | dict | None = None,
    environment: EnvironmentConfig | dict | None = None,
) -> str:
    """Build a complete MuJoCo XML model with independent hip/knee motors.

    Sixty percent of total mass is in the torso; the rest is divided equally
    among legs (55% upper segment, 40% lower segment, 5% foot). Body geoms
    collide with the floor but not each other. This permits a broad range of
    editor dimensions without self-contact trapping intersecting legs.
    """
    body = _as_body(body)
    environment = _as_environment(environment)
    foot_radius = body.leg_radius * 1.3
    standing_height = (
        (body.upper_length + body.lower_length) * math.cos(0.1)
        + foot_radius
        + 0.025
    )
    pair_count = body.legs // 2
    hip_x_positions = (
        [0.0]
        if pair_count == 1
        else np.linspace(-0.38 * body.torso_length, 0.38 * body.torso_length, pair_count)
    )
    leg_mass = body.body_mass * 0.4 / body.legs
    leg_bodies: list[str] = []
    motors: list[str] = []
    initial_joints: list[float] = []
    for pair_index, hip_x in enumerate(hip_x_positions):
        for side_index, side in enumerate((-1, 1)):
            leg_index = pair_index * 2 + side_index
            hip_y = side * (body.torso_radius + body.leg_radius * 0.5)
            color = "0.31 0.83 0.78 1" if side > 0 else "0.23 0.54 0.62 1"
            leg_bodies.append(
                f"""
                <body name="upper_{leg_index}" pos="{hip_x:.8f} {hip_y:.8f} 0">
                  <joint name="hip_{leg_index}" type="hinge" axis="0 1 0"
                         range="-1.3 1.3" damping="{body.joint_damping}" />
                  <geom name="upper_geom_{leg_index}" type="capsule"
                        fromto="0 0 0 0 0 {-body.upper_length}"
                        size="{body.leg_radius}" mass="{leg_mass * 0.55}"
                        rgba="{color}" />
                  <body name="lower_{leg_index}" pos="0 0 {-body.upper_length}">
                    <joint name="knee_{leg_index}" type="hinge" axis="0 1 0"
                           range="-2.1 0.45" damping="{body.joint_damping}" />
                    <geom name="lower_geom_{leg_index}" type="capsule"
                          fromto="0 0 0 0 0 {-body.lower_length}"
                          size="{body.leg_radius * 0.85}" mass="{leg_mass * 0.4}"
                          rgba="{color}" />
                    <geom name="foot_{leg_index}" type="sphere"
                          pos="0 0 {-body.lower_length}" size="{foot_radius}"
                          mass="{leg_mass * 0.05}" rgba="0.85 0.95 0.90 1" />
                  </body>
                </body>"""
            )
            for joint_name in (f"hip_{leg_index}", f"knee_{leg_index}"):
                motors.append(
                    f'<motor name="{joint_name}_motor" joint="{joint_name}" '
                    f'gear="{body.motor_strength}" ctrllimited="true" ctrlrange="-1 1" />'
                )
            initial_joints.extend((0.1, -0.2))
    qpos = " ".join(str(q) for q in (0.0, 0.0, 0.0, *initial_joints))
    model_name = escape(body.name, {'"': "&quot;"})
    return f"""<mujoco model="{model_name}">
      <compiler angle="radian" autolimits="true" />
      <option timestep="{PHYSICS_TIMESTEP}" gravity="0 0 {-environment.gravity}"
              integrator="implicitfast" iterations="50" />
      <default>
        <joint armature="0.025" limited="true" />
        <geom contype="1" conaffinity="0" friction="1.0 0.005 0.0001"
              condim="3" solref="0.012 1" solimp="0.9 0.95 0.001" />
      </default>
      <worldbody>
        <light pos="0 -3 5" dir="0 0 -1" />
        <geom name="floor" type="plane" size="200 5 0.1"
              contype="0" conaffinity="1" priority="1"
              friction="{environment.friction} 0.005 0.0001"
              rgba="0.12 0.15 0.18 1" />
        <body name="torso" pos="0 0 {standing_height}">
          <joint name="root_x" type="slide" axis="1 0 0" limited="false"
                 damping="0" armature="0" />
          <joint name="root_z" type="slide" axis="0 0 1" limited="false"
                 damping="0" armature="0" />
          <joint name="root_pitch" type="hinge" axis="0 1 0" limited="false"
                 damping="0" armature="0" />
          <geom name="torso_geom" type="capsule"
                fromto="{-body.torso_length / 2} 0 0 {body.torso_length / 2} 0 0"
                size="{body.torso_radius}" mass="{body.body_mass * 0.6}"
                rgba="0.88 0.93 0.83 1" />
          {''.join(leg_bodies)}
        </body>
      </worldbody>
      <actuator>{''.join(motors)}</actuator>
      <keyframe><key name="standing" qpos="{qpos}" /></keyframe>
    </mujoco>"""


class WalkerEnv(gym.Env):
    """Learn a requested forward speed using normalized torque actions.

    A control step advances 5 physics steps of 0.004 seconds. The 1,000-step
    time limit is therefore 20 simulated seconds. Episodes terminate early
    when the torso falls low or pitches more than about 63 degrees.

    Domain randomization, when enabled, draws independent episode-wide
    multipliers for all body masses/inertias (0.8–1.2), motor torque (0.8–1.2)
    and floor friction (0.6–1.4). The observation contains these multipliers.
    ``set_training_progress(p)`` sets the target speed to
    ``requested_speed * (0.25 + 0.75 * clip(p, 0, 1))``. The default progress
    is 1, so no speed curriculum runs unless training calls that method.
    """

    metadata = {"render_modes": [], "render_fps": 50}

    def __init__(
        self,
        body: BodyConfig | dict | None = None,
        target_speed: float = 1.0,
        randomize: bool = False,
        seed: int | None = None,
        *,
        environment: EnvironmentConfig | dict | None = None,
        task: TaskConfig | dict | None = None,
    ):
        super().__init__()
        if not math.isfinite(target_speed) or not 0.1 <= target_speed <= 5.0:
            raise ValueError("target_speed must be finite and between 0.1 and 5.0 m/s")
        self.body = _as_body(body)
        self.environment = _as_environment(environment)
        self.task = (task.model_copy(deep=True) if isinstance(task, TaskConfig)
                     else TaskConfig.model_validate(task) if task is not None else None)
        self.model = mujoco.MjModel.from_xml_string(
            export_mjcf(self.body, self.environment)
        )
        self.data = mujoco.MjData(self.model)
        self.frame_skip = FRAME_SKIP
        self.dt = PHYSICS_TIMESTEP * FRAME_SKIP
        self.max_episode_steps = (
            max(1, math.floor(self.task.episode_seconds / self.dt + .5))
            if self.task is not None else EPISODE_STEPS
        )
        self.randomize = bool(randomize)
        self.requested_target_speed = float(target_speed)
        self.target_speed = float(target_speed)
        self.training_progress = 1.0
        self._initial_seed = seed
        self._torso_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "torso")
        self._tracked_id = (
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, self.task.tracked_body)
            if self.task is not None and self.task.tracked_body else self._torso_id
        )
        if self._tracked_id < 1:
            raise ValueError(f"Tracked body {self.task.tracked_body!r} was not found.")
        self._base_mass = self.model.body_mass.copy()
        self._base_inertia = self.model.body_inertia.copy()
        self._base_friction = self.model.geom_friction.copy()
        self._base_gear = self.model.actuator_gear.copy()
        self._domain_scales = np.ones(3, dtype=np.float64)
        self._morphology = np.asarray(self.body.morphology_vector(), dtype=np.float64)
        self.action_space = spaces.Box(-1.0, 1.0, shape=(self.model.nu,), dtype=np.float32)
        observation_size = self.model.nq - 1 + self.model.nv + 1 + len(self._morphology) + 3
        self.observation_schema = "rlsim-planar-task-v1" if self.task is not None else "rlsim-planar-v1"
        if self.task is not None:
            observation_size += 9
        self.observation_space = spaces.Box(
            -np.inf, np.inf, shape=(observation_size,), dtype=np.float32
        )
        self.coordinated = self.task is not None and self.task.preset == "walk"
        self.motion = MotionTracker(self.body)
        self.effort = EffortMeter()
        self._feet_ids = np.array([mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, f"foot_{i}") for i in range(self.body.legs)])
        self._foot_id_set = set(self._feet_ids.tolist())
        self.phase_offset = 0.
        if self.coordinated:
            self.observation_schema = "rlsim-planar-gait-v1"
            self.observation_space = spaces.Box(-np.inf, np.inf, shape=(observation_size + 2 + self.body.legs + self.model.nu,), dtype=np.float32)
        self._previous_action = np.zeros(self.model.nu, dtype=np.float64)
        self._steps = 0
        self._fallen = False
        self.last_info: dict[str, Any] = {}
        self.reset(seed=seed)

    def set_training_progress(self, progress: float) -> None:
        """Set a speed curriculum explicitly; 1.0 restores the full target."""
        if not math.isfinite(progress):
            raise ValueError("training progress must be finite")
        self.training_progress = float(np.clip(progress, 0.0, 1.0))
        self.target_speed = self.requested_target_speed * (
            0.25 + 0.75 * self.training_progress
        )

    def _observation(self) -> np.ndarray:
        parts = [
                self.data.qpos[1:],
                self.data.qvel,
                [self.target_speed],
                self._morphology,
                (self._domain_scales - 1.0) / np.array([0.2, 0.2, 0.4]),
        ]
        if self.coordinated:
            phase, _, _ = phase_targets(self.data.time, self.phase_offset, self.body.legs, self.motion.length)
            parts.extend(([math.sin(2*math.pi*phase), math.cos(2*math.pi*phase)], self.motion.contacts, self._previous_action))
        if self.task is not None:
            parts.extend((self._task_position, np.asarray(self.task.target) - self._task_position,
                          self._task_velocity))
        return np.concatenate(parts).astype(np.float32)

    def set_learning_stage(self, stage) -> None:
        if self.task is None:
            raise ValueError("Staged curriculum requires an explicit velocity task.")
        if not hasattr(self, "_requested_task"):
            self._requested_task = self.task
        self.task = self._requested_task.model_copy(update={"preset": stage["preset"]})
        self.target_speed = self.requested_target_speed * stage["speed_fraction"]
        self.randomization_strength = stage["randomization"]

    def _randomize_model(self) -> None:
        self._domain_scales[:] = 1.0
        if self.randomize:
            self._domain_scales[:] = (
                self.np_random.uniform(0.8, 1.2),
                self.np_random.uniform(0.8, 1.2),
                self.np_random.uniform(0.6, 1.4),
            )
            self._domain_scales[:] = 1 + (self._domain_scales - 1) * getattr(self, "randomization_strength", 1)
        mass_scale, motor_scale, friction_scale = self._domain_scales
        self.model.body_mass[:] = self._base_mass * mass_scale
        self.model.body_inertia[:] = self._base_inertia * mass_scale
        self.model.actuator_gear[:] = self._base_gear * motor_scale
        self.model.geom_friction[:] = self._base_friction * friction_scale
        mujoco.mj_setConst(self.model, self.data)

    def reset(
        self, *, seed: int | None = None, options: dict | None = None
    ) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        if seed is not None:
            self.action_space.seed(seed)
        self._randomize_model()
        mujoco.mj_resetDataKeyframe(self.model, self.data, 0)
        # Small seeded pose/velocity perturbations prevent a single exact
        # initial state from being the only state seen during training.
        self.data.qpos[2:] += self.np_random.uniform(
            -0.015, 0.015, size=self.model.nq - 2
        )
        self.data.qvel[:] = self.np_random.uniform(
            -0.015, 0.015, size=self.model.nv
        )
        mujoco.mj_forward(self.model, self.data)
        self.phase_offset = float(self.np_random.uniform()) if self.coordinated else 0.
        self.motion.reset(self._feet())
        self.effort.reset()
        self._previous_action[:] = 0.0
        self._steps = 0
        self._fallen = False
        if self.task is not None:
            self._task_position = self.data.xpos[self._tracked_id].copy()
            self._task_velocity = np.zeros(3)
            self._task_initial_height = float(self._task_position[2])
            self._task_initial_up = self.data.xmat[self._tracked_id].reshape(3, 3)[2].copy()
            self._task_distance = math.dist(self.task.target, self._task_position)
        self.last_info = self._state_info(float(self.data.qvel[0]))
        return self._observation(), dict(self.last_info)

    def _feet(self):
        feet = self.data.geom_xpos[self._feet_ids].copy()
        feet[:, 2] -= self.body.leg_radius * 1.3
        return feet

    def _state_info(self, velocity: float) -> dict[str, Any]:
        return {
            "position": float(self.data.qpos[0]),
            "velocity": float(velocity),
            "target_speed": float(self.target_speed),
            "height": float(self.data.xpos[self._torso_id, 2]),
            "pitch": float(self.data.qpos[2]),
            "fallen": bool(self._fallen),
            "step": self._steps,
            "mass_scale": float(self._domain_scales[0]),
            "motor_scale": float(self._domain_scales[1]),
            "friction_scale": float(self._domain_scales[2]),
        }

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        action = np.asarray(action, dtype=np.float64)
        if action.shape != self.action_space.shape:
            raise ValueError(f"expected action shape {self.action_space.shape}, got {action.shape}")
        if not np.isfinite(action).all():
            raise ValueError("actions must contain only finite values")
        action = np.clip(action, -1.0, 1.0)
        x_before = float(self.data.qpos[0])
        step_work = 0.
        nonfoot_contact = False
        for _ in range(self.frame_skip):
            self.data.ctrl[:] = position_controls(action, self.data.qpos, self.data.qvel, self.body) if self.coordinated else action
            mujoco.mj_step(self.model, self.data)
            step_work += self.effort.observe(self.data.actuator_force, self.data.actuator_velocity,
                self.data.qfrc_actuator[3:], self.data.ctrl, PHYSICS_TIMESTEP)
            if self.task is not None and self.task.foot_only_support:
                nonfoot_contact |= any(c.efc_address >= 0 and c.dist <= 0 and 0 in c.geom
                    and int(c.geom[1] if c.geom[0] == 0 else c.geom[0]) not in self._foot_id_set
                    for c in self.data.contact)
        # mj_step leaves position-dependent derived fields from its last
        # integration point; forward gives the renderer the current pose.
        mujoco.mj_forward(self.model, self.data)
        self._steps += 1
        velocity = (float(self.data.qpos[0]) - x_before) / self.dt
        height = float(self.data.xpos[self._torso_id, 2])
        upright = math.cos(float(self.data.qpos[2]))
        finite = bool(np.isfinite(self.data.qpos).all() and np.isfinite(self.data.qvel).all())
        self._fallen = (
            not finite
            or height < max(self.body.torso_radius * 1.4,
                            (self.body.upper_length + self.body.lower_length) * 0.35)
            or upright < 0.45
        )
        # A smooth, bounded speed objective discourages explosive launches.
        # Signed progress supplies a gradient even far from the target speed.
        speed_error = (velocity - self.target_speed) / max(self.target_speed, 0.5)
        speed_tracking = math.exp(-2.0 * speed_error**2) if finite else 0.0
        forward_progress = float(np.clip(velocity / max(self.target_speed, 0.5), -1.0, 1.0))
        energy = float(np.mean(np.square(action)))
        smoothness = float(np.mean(np.square(action - self._previous_action)))
        feet = self._feet()
        slip = self.motion.update(feet, self.dt, float(self.data.qpos[2]), smoothness)
        reward_components = {
            "speed_tracking": 1.5 * speed_tracking,
            "forward_progress": 0.3 * forward_progress,
            "alive": 0.15 if not self._fallen else 0.0,
            "upright": 0.15 * max(upright, 0.0) if not self._fallen else 0.0,
            "energy": -0.04 * energy,
            "smoothness": -0.02 * smoothness,
            "fall": -1.0 if self._fallen else 0.0,
        }
        reward = float(sum(reward_components.values()))
        self._previous_action[:] = action
        self.last_info = self._state_info(velocity)
        self.last_info.update({
            "reward_components": reward_components,
            "speed_error": float(velocity - self.target_speed),
            "energy_cost": energy,
            "smoothness_cost": smoothness,
            "reward": reward,
        })
        if self.task is not None:
            previous = self._task_position
            self._task_position = self.data.xpos[self._tracked_id].copy()
            self._task_velocity = (self._task_position - previous) / self.dt
            task_upright = float(np.dot(
                self._task_initial_up, self.data.xmat[self._tracked_id].reshape(3, 3)[2],
            ))
            outcome = evaluate_task(self.task, {
                "position": self._task_position, "velocity": self._task_velocity,
                "height": float(self._task_position[2]), "initial_height": self._task_initial_height,
                "upright": task_upright, "target_speed": self.target_speed,
                "energy": energy, "smoothness": smoothness, "elapsed": self._steps * self.dt,
                "previous_distance": self._task_distance, "dt": self.dt,
            })
            self._task_distance = outcome["distance"]
            self._fallen = bool(outcome["fallen"])
            reward = outcome["reward"]
            self.last_info.update({
                "velocity": float(self._task_velocity[0]), "height": float(self._task_position[2]),
                "reward": reward, "reward_components": outcome["components"],
                "task_distance": self._task_distance, "fallen": self._fallen,
            })
        if nonfoot_contact:
            self._fallen = True
            outcome["terminated"] = self.task.terminate_on_fall
            reward -= 2.
            self.last_info["reward_components"]["nonfoot_contact"] = -2.
            self.last_info["fallen"] = True
        self.last_info["nonfoot_contact"] = bool(nonfoot_contact)
        self.last_info["motor_work_j"] = step_work
        if self.coordinated:
            if self.task.mechanical_work_weight is not None:
                work_cost = -self.task.mechanical_work_weight * step_work / (float(self.model.body_mass.sum()) * self.environment.gravity * max(self.target_speed, .1) * self.dt)
                reward += work_cost
                self.last_info["reward_components"]["mechanical_work"] = work_cost
            components = walking_components(self.body, feet, self.motion.contacts, self.data.time, self.phase_offset,
                velocity, self.target_speed, height, float(self.data.qpos[2]), float(self.data.qvel[1]), float(self.data.qvel[2]), slip, smoothness, .2 if self.task.foot_drift_weight is None else self.task.foot_drift_weight)
            if not self._fallen:
                reward += sum(components.values())
                self.last_info["reward_components"].update(components)
            self.last_info["reward"] = reward
        terminated = bool(self._fallen)
        if self.task is not None:
            terminated = bool(outcome["terminated"])
        truncated = self._steps >= self.max_episode_steps
        return self._observation(), reward, terminated, truncated, dict(self.last_info)

    def get_frame(self) -> dict[str, Any]:
        """Return actual world-space MuJoCo geometry for a browser renderer."""
        geom_types = {
            int(mujoco.mjtGeom.mjGEOM_CAPSULE): "capsule",
            int(mujoco.mjtGeom.mjGEOM_SPHERE): "sphere",
            int(mujoco.mjtGeom.mjGEOM_BOX): "box",
            int(mujoco.mjtGeom.mjGEOM_PLANE): "plane",
        }
        geoms = []
        for index in range(self.model.ngeom):
            kind = geom_types.get(int(self.model.geom_type[index]))
            if kind is None:
                continue
            geoms.append({
                "type": kind,
                "pos": self.data.geom_xpos[index].tolist(),
                "mat": self.data.geom_xmat[index].tolist(),
                "size": self.model.geom_size[index].tolist(),
                "rgba": self.model.geom_rgba[index].tolist(),
            })
        return {
            "time": float(self.data.time),
            "position": float(self.data.qpos[0]),
            "velocity": float(self.data.qvel[0]),
            "torso_position": self.data.xpos[self._torso_id].tolist(),
            "target_speed": float(self.target_speed),
            "fallen": bool(self._fallen),
            "geoms": geoms,
            **({"target": self.task.target} if self.task is not None and self.task.preset in {"target", "custom"} else {}),
        }

    def model_info(self) -> dict:
        """Describe the simulator interface for body inspection and task selection."""
        observations = [
            *(f"qpos[{index}]" for index in range(1, self.model.nq)),
            *(f"qvel[{index}]" for index in range(self.model.nv)),
            "target_speed",
            *(f"normalized_morphology:{key}" for key in (
                "legs", "torso_length", "torso_radius", "upper_length", "lower_length",
                "leg_radius", "body_mass", "motor_strength", "joint_damping",
            )),
            "normalized_mass_scale", "normalized_motor_scale", "normalized_friction_scale",
        ]
        if self.coordinated:
            observations.extend(["gait_phase_sin", "gait_phase_cos", *[f"foot_contact[{i}]" for i in range(self.body.legs)], *[f"previous_action[{i}]" for i in range(self.model.nu)]])
        if self.task is not None:
            observations.extend(
                f"{group}.{axis}"
                for group in ("tracked_position", "target_delta", "body_velocity")
                for axis in ("x", "y", "z")
            )
        actions = [mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, index)
                   for index in range(self.model.nu)]
        return {
            "bodies": [{"name": mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, index)}
                       for index in range(1, self.model.nbody)],
            "joints": [{"name": mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, index),
                        "type": ("free", "ball", "slide", "hinge")[self.model.jnt_type[index]]}
                       for index in range(self.model.njnt)],
            "actuators": [{"name": name, "normalized_range": [-1, 1],
                           "control_range": (self.model.actuator_ctrlrange[index].tolist()
                                             if self.model.actuator_ctrllimited[index] else [-1, 1])}
                          for index, name in enumerate(actions)],
            "observation_size": self.observation_space.shape[0],
            "action_size": self.action_space.shape[0], "observation_schema": self.observation_schema,
            "observation_names": observations, "action_names": actions,
            "action_schema": "rlsim-joint-targets-v1" if self.coordinated else "rlsim-actuator-controls-v1",
            "timestep": self.dt,
            "tracked_body": mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, self._tracked_id),
        }

    def close(self) -> None:
        """No render context or external process is owned by this environment."""
