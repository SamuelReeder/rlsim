"""Validated, serializable body descriptions shared by the editor and simulator."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class EnvironmentConfig(BaseModel):
    """World gravity in m/s² and the floor's sliding friction coefficient."""

    model_config = ConfigDict(
        extra="forbid", validate_assignment=True, allow_inf_nan=False
    )

    gravity: float = Field(default=9.81, ge=0.5, le=20.0)
    friction: float = Field(default=1.0, ge=0.05, le=3.0)


class BodyConfig(BaseModel):
    """A symmetric body with two actuated joints per leg.

    Lengths are metres, ``body_mass`` is the total mass in kilograms, and
    ``motor_strength`` is the maximum torque in newton-metres per actuator.
    The root can translate in x/z and pitch around y; each leg moves in its
    own sagittal plane. Changing the number of legs changes the policy shape.
    """

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    name: str = Field(default="Quadruped", min_length=1, max_length=60)
    legs: Literal[2, 4, 6] = 4
    torso_length: float = Field(default=0.8, ge=0.3, le=1.8)
    torso_radius: float = Field(default=0.14, ge=0.07, le=0.3)
    upper_length: float = Field(default=0.42, ge=0.18, le=0.9)
    lower_length: float = Field(default=0.42, ge=0.18, le=0.9)
    leg_radius: float = Field(default=0.055, ge=0.025, le=0.1)
    body_mass: float = Field(default=4.0, ge=1.0, le=20.0)
    motor_strength: float = Field(default=60.0, ge=5.0, le=150.0)
    joint_damping: float = Field(default=1.5, ge=0.1, le=5.0)

    def morphology_vector(self) -> list[float]:
        """Map the nine numeric morphology settings into [-1, 1]."""
        limits = {
            "legs": (2, 6),
            "torso_length": (0.3, 1.8),
            "torso_radius": (0.07, 0.3),
            "upper_length": (0.18, 0.9),
            "lower_length": (0.18, 0.9),
            "leg_radius": (0.025, 0.1),
            "body_mass": (1.0, 20.0),
            "motor_strength": (5.0, 150.0),
            "joint_damping": (0.1, 5.0),
        }
        return [
            2.0 * (float(getattr(self, key)) - low) / (high - low) - 1.0
            for key, (low, high) in limits.items()
        ]


PRESETS: dict[str, BodyConfig] = {
    "quadruped": BodyConfig(),
    "biped": BodyConfig(
        name="Biped",
        legs=2,
        torso_length=0.45,
        torso_radius=0.16,
        upper_length=0.46,
        lower_length=0.46,
        body_mass=3.5,
        motor_strength=45.0,
    ),
    "hexapod": BodyConfig(
        name="Hexapod",
        legs=6,
        torso_length=1.15,
        torso_radius=0.16,
        upper_length=0.34,
        lower_length=0.36,
        body_mass=6.0,
        motor_strength=45.0,
    ),
}


def get_presets() -> dict[str, dict]:
    """Return fresh JSON-compatible configurations for the browser editor."""
    return {key: body.model_dump() for key, body in PRESETS.items()}
