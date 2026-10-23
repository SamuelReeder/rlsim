"""Fixed-topology body variation. Test regions are excluded from training."""
from __future__ import annotations

import gymnasium as gym
import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator

from .config import BodyConfig
from .environment import WalkerEnv

DIMENSIONS = ("torso_length", "upper_length", "lower_length")


class ProportionTraining(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    enabled: bool = True
    width: float = Field(default=.2, ge=.05, le=.4)
    ramp_steps: int = Field(default=131072, ge=0, le=2_000_000)
    withhold_width: float | None = Field(default=None, ge=.05, le=.4, exclude_if=lambda v: v is None)
    speed_range: tuple[float, float] | None = Field(default=None, exclude_if=lambda v: v is None)
    boundary_fraction: float | None = Field(default=None, ge=0, le=.5, exclude_if=lambda v: v is None)
    robust_validation: bool | None = Field(default=None, exclude_if=lambda v: v is None)

    @model_validator(mode="after")
    def validate_ranges(self):
        if self.speed_range is not None and not .1 <= self.speed_range[0] <= self.speed_range[1] <= 3:
            raise ValueError("Speed range must be ordered and inside 0.1–3 m/s.")
        if self.withhold_width is not None and self.withhold_width > self.width:
            raise ValueError("Withheld corner must lie within the sampling range.")
        return self


def body_at(base: BodyConfig, scales) -> BodyConfig:
    values = np.asarray(scales, dtype=float)
    if values.shape != (3,) or not np.isfinite(values).all() or np.any(values < .6) or np.any(values > 1.4):
        raise ValueError("Provide three proportion multipliers between 0.6 and 1.4.")
    return BodyConfig.model_validate({**base.model_dump(),
        **{key: getattr(base, key) * float(scale) for key, scale in zip(DIMENSIONS, values)}})


def region(scales, width=.2):
    if any(abs(float(s)-1) > width+1e-9 for s in scales):
        return "outside training range"
    if scales[1] >= 1+width/2-1e-9 and scales[2] <= 1-width/2+1e-9:
        return "withheld region"
    return "training range"


def validation_scales(width=.2):
    return [(1., 1., 1.), (1-width*.5, 1+width*.25, 1+width*.5),
            (1+width*.6, 1-width*.5, 1+width*.15)]


class ProportionWalker(gym.Wrapper):
    """Recompile at episode boundaries; preserve the environment's RNG stream.

    MuJoCo derives inertia, joint anchors and initial standing height from the
    changed geometry. Total mass, thickness and nominal torque stay fixed.
    """
    def __init__(self, body, proportions, **kwargs):
        self.base_body = body.model_copy(deep=True)
        self.proportions = proportions
        self.options = kwargs
        self.variation_progress = 0. if proportions.ramp_steps else 1.
        self.speed_progress = 0.
        self.learning_stage = None
        super().__init__(WalkerEnv(body, **kwargs))

    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        return getattr(self.env, name)

    def set_variation_steps(self, steps):
        self.variation_progress = min(1., steps/max(1, self.proportions.ramp_steps))

    def set_training_progress(self, progress):
        self.speed_progress = progress
        self.env.set_training_progress(progress)

    def set_learning_stage(self, stage):
        self.learning_stage = stage
        self.env.set_learning_stage(stage)

    def reset(self, *, seed=None, options=None):
        rng = np.random.default_rng(seed) if seed is not None else self.env.np_random
        width = self.proportions.width
        radius = width * self.variation_progress
        while True:
            scales = rng.uniform(1-radius, 1+radius, 3)
            if self.proportions.boundary_fraction and rng.random() < self.proportions.boundary_fraction:
                scales = 1 + radius * rng.choice([-1., 1.], size=3)
            withheld = self.proportions.withhold_width or width
            if not (scales[1] >= 1+withheld/2-1e-9 and scales[2] <= 1-withheld/2+1e-9):
                break
        options_copy = dict(self.options)
        if self.proportions.speed_range is not None:
            options_copy["target_speed"] = float(rng.uniform(*self.proportions.speed_range))
        replacement = WalkerEnv(body_at(self.base_body, scales), **options_copy)
        replacement.np_random = rng
        if self.learning_stage:
            replacement.set_learning_stage(self.learning_stage)
        else:
            replacement.set_training_progress(self.speed_progress)
        self.env.close()
        self.env = replacement
        observation, info = self.env.reset(options=options)
        return observation, {**info, "proportions": scales.tolist()}


def validation_cases(config):
    """Validation stays outside the deliberately withheld upper/lower corner."""
    m=config.morphology
    if m and m.robust_validation:
        raw=[([1,1,1],.5),([1,1,1],1.5),([.7,.65,.7],1.),([1.3,.7,1.25],1.),
             ([.75,1.3,1.25],1.2),([1.2,1.05,.65],.75),([1.25,.75,.75],1.),([1,1,1],1.)]
        return [config.model_copy(update={'body':body_at(config.body,s),'target_speed':v}) for s,v in raw]
    return [config.model_copy(update={'body':body_at(config.body,s)}) for s in validation_scales(m.width)] if m else [config]
