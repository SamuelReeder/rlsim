"""Portable task settings and a bounded arithmetic reward language.

The parser and reward terms match static/task-regimes.mjs. Expressions are
interpreted as arithmetic trees; they cannot execute Python or access objects.
"""
from __future__ import annotations

from functools import lru_cache
import math
import re
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, StrictBool, model_validator


DEFAULT_EXPRESSION = (
    "1.5*speed_tracking + 0.3*forward_progress + 0.15*alive + 0.15*upright + "
    "0.04*energy + 0.02*smoothness + fall"
)
REWARD_VARIABLES = (
    "speed_tracking", "forward_progress", "alive", "upright", "energy",
    "smoothness", "fall", "target_progress", "target_distance", "target_success",
    "height_tracking", "x", "y", "z", "vx", "vy", "vz", "speed", "height",
    "initial_height", "target_speed", "distance", "time", "dt", "pi",
)
_PRECEDENCE = {
    "||": 1, "&&": 2, "==": 3, "!=": 3, "<": 4, ">": 4, "<=": 4, ">=": 4,
    "+": 5, "-": 5, "*": 6, "/": 6, "%": 6, "^": 7, "**": 7,
}
_NUMBER = re.compile(r"(?:\d+\.?\d*|\.\d+)(?:e[+-]?\d+)?", re.I)
_NAME = re.compile(r"[A-Za-z_][A-Za-z_0-9]*")
_OPERATOR = re.compile(r"\*\*|<=|>=|==|!=|&&|\|\||[+\-*/%^(),?:!<>]")
_FUNCTIONS = {"abs", "min", "max", "sqrt", "exp", "log", "sin", "cos", "tanh", "pow", "clamp"}


def _clip(value, low, high):
    return max(low, min(high, value))


@lru_cache(maxsize=64)
def compile_reward(source: str):
    """Compile the same limited expression grammar used by browser policies."""
    if not isinstance(source, str) or not source.strip() or len(source) > 2000:
        raise ValueError("Reward expression must contain 1–2,000 characters.")
    tokens = []
    cursor = 0
    while cursor < len(source):
        if source[cursor].isspace():
            cursor += 1
            continue
        found = None
        for kind, pattern in (("number", _NUMBER), ("name", _NAME), ("operator", _OPERATOR)):
            match = pattern.match(source, cursor)
            if match:
                found = (kind, match.group())
                cursor = match.end()
                break
        if found is None:
            raise ValueError(f"Invalid reward syntax near {source[cursor:cursor + 16]!r}.")
        tokens.append(found)
        if len(tokens) > 512:
            raise ValueError("Reward expression is too complex.")
    index = 0
    depth = 0

    def take(value):
        nonlocal index
        if index < len(tokens) and tokens[index][1] == value:
            index += 1
            return True
        return False

    def need(value):
        if not take(value):
            raise ValueError(f"Expected {value!r} in reward expression.")

    def parse(minimum=0):
        nonlocal index, depth
        depth += 1
        if depth > 64:
            raise ValueError("Reward expression is nested too deeply.")
        if index >= len(tokens):
            raise ValueError("Incomplete reward expression.")
        kind, token = tokens[index]
        index += 1
        if kind == "number":
            value = float(token)
            if not math.isfinite(value):
                raise ValueError("Reward constants must be finite.")
            node = ("number", value)
        elif token == "(":
            node = parse()
            need(")")
        elif token in {"+", "-", "!"}:
            node = ("unary", token, parse(8))
        elif kind == "name":
            if take("("):
                if token not in _FUNCTIONS:
                    raise ValueError(f"Unknown reward function: {token}.")
                args = []
                if not take(")"):
                    args.append(parse())
                    while take(","):
                        args.append(parse())
                    need(")")
                if not 1 <= len(args) <= 8:
                    raise ValueError("Reward functions require 1–8 arguments.")
                node = ("call", token, args)
            else:
                if token not in REWARD_VARIABLES:
                    raise ValueError(f"Unknown reward variable: {token}.")
                node = ("variable", token)
        else:
            raise ValueError(f"Unexpected token {token!r}.")
        while index < len(tokens):
            operator = tokens[index][1]
            priority = _PRECEDENCE.get(operator, -1)
            if priority < minimum:
                break
            index += 1
            node = ("binary", operator, node, parse(priority + (0 if operator in {"^", "**"} else 1)))
        if minimum == 0 and take("?"):
            yes = parse()
            need(":")
            node = ("conditional", node, yes, parse())
        depth -= 1
        return node

    tree = parse()
    if index != len(tokens):
        raise ValueError(f"Unexpected token {tokens[index][1]!r}.")

    def evaluate(node, variables):
        kind = node[0]
        if kind == "number":
            return node[1]
        if kind == "variable":
            return variables[node[1]]
        if kind == "conditional":
            return evaluate(node[2] if evaluate(node[1], variables) else node[3], variables)
        if kind == "call":
            name = node[1]
            args = [evaluate(arg, variables) for arg in node[2]]
            if name == "min":
                return min(args)
            if name == "max":
                return max(args)
            if name == "pow":
                return math.pow(args[0], args[1])
            if name == "clamp":
                return _clip(args[0], args[1], args[2])
            function = abs if name == "abs" else getattr(math, name)
            return function(args[0])
        if kind == "unary":
            value = evaluate(node[2], variables)
            return -value if node[1] == "-" else float(not value) if node[1] == "!" else value
        operator = node[1]
        left = evaluate(node[2], variables)
        if operator == "&&":
            return float(bool(evaluate(node[3], variables))) if left else 0.0
        if operator == "||":
            return 1.0 if left else float(bool(evaluate(node[3], variables)))
        right = evaluate(node[3], variables)
        if operator == "+": return left + right
        if operator == "-": return left - right
        if operator == "*": return left * right
        if operator == "/": return left / right
        if operator == "%": return math.fmod(left, right)
        if operator in {"^", "**"}: return math.pow(left, right)
        if operator == "<": return float(left < right)
        if operator == ">": return float(left > right)
        if operator == "<=": return float(left <= right)
        if operator == ">=": return float(left >= right)
        if operator == "==": return float(left == right)
        if operator == "!=": return float(left != right)
        raise ValueError("Unknown reward operation.")

    def reward(variables):
        try:
            value = float(evaluate(tree, variables))
            if not math.isfinite(value):
                raise ValueError("non-finite result")
            return value
        except (ArithmeticError, ValueError, TypeError, IndexError, KeyError) as exc:
            raise ValueError(
                "Custom reward produced an invalid value. Check divisions, variables and function domains."
            ) from exc
    return reward


Coordinate = Annotated[float, Field(strict=True, ge=-1000, le=1000)]


class TaskConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    preset: Literal["velocity", "walk", "balance", "target", "custom"] = "velocity"
    episode_seconds: float = Field(default=20, ge=1, le=120, strict=True)
    target: list[Coordinate] = Field(default_factory=lambda: [2, 0, .8], min_length=3, max_length=3)
    tracked_body: str = Field(default="", max_length=200, strict=True)
    terminate_on_fall: StrictBool = True
    min_height_ratio: float = Field(default=.35, ge=0, le=1, strict=True)
    max_tilt_degrees: float = Field(default=64, ge=0, le=180, strict=True)
    mechanical_work_weight: float | None = Field(default=None, ge=0, le=2, exclude_if=lambda v: v is None)
    foot_only_support: StrictBool | None = Field(default=None, exclude_if=lambda v: v is None)
    foot_drift_weight: float | None = Field(default=None, ge=0, le=5, exclude_if=lambda v: v is None)
    expression: str | None = Field(default=None, max_length=2000)
    weights: dict[str, Annotated[float, Field(strict=True, ge=-100, le=100)]] | None = None

    @model_validator(mode="after")
    def check_reward(self):
        if (self.mechanical_work_weight is not None or self.foot_only_support is not None) and self.preset != "walk":
            raise ValueError("Mechanical work and foot support settings require Coordinated walking.")
        if self.foot_drift_weight is not None and self.preset != "walk":
            raise ValueError("Foot drift weight requires the Coordinated walking objective.")
        if self.preset == "walk" and (self.tracked_body not in {"", "torso"} or self.weights is not None):
            raise ValueError("Coordinated walking uses the template torso and its built-in reward weights.")
        if self.preset == "custom":
            self.expression = self.expression or DEFAULT_EXPRESSION
            compile_reward(self.expression)
        else:
            self.expression = None
        if self.weights is not None:
            for key in self.weights:
                if key not in REWARD_VARIABLES[:11]:
                    raise ValueError(f"Invalid reward weight: {key}.")
        return self


def evaluate_task(task: TaskConfig, state: dict) -> dict:
    """Evaluate a task from simulator measurements, matching browser semantics."""
    position = state["position"]
    velocity = state["velocity"]
    if isinstance(velocity, (int, float)):
        velocity = [velocity, 0, 0]
    x, y, z = position
    vx, vy, vz = velocity
    distance = math.dist(task.target, position)
    fallen = (
        state["height"] < max(.01, state["initial_height"] * task.min_height_ratio)
        or state["upright"] < math.cos(math.radians(task.max_tilt_degrees))
    )
    target = state["target_speed"] or 1
    denominator = max(abs(target), .5)
    terms = {
        "speed_tracking": math.exp(-2 * ((vx - target) / denominator) ** 2),
        "forward_progress": _clip(vx / denominator, -1, 1),
        "alive": 0.0 if fallen else 1.0,
        "upright": 0.0 if fallen else max(state["upright"], 0),
        "energy": -state["energy"], "smoothness": -state["smoothness"],
        "fall": -1.0 if fallen else 0.0,
        "target_progress": _clip((state.get("previous_distance", distance) - distance) / state["dt"], -10, 10),
        "target_distance": -distance, "target_success": float(distance < .2),
        "height_tracking": math.exp(-2 * ((state["height"] - state["initial_height"]) / max(abs(state["initial_height"]), .1)) ** 2),
    }
    defaults = {
        "balance": {"alive": .2, "upright": 1, "height_tracking": .5, "energy": .03, "smoothness": .01, "fall": 1},
        "target": {"alive": .05, "upright": .1, "target_progress": 1, "target_distance": .05, "target_success": 1, "energy": .02, "smoothness": .01, "fall": 1},
    }
    weights = task.weights if task.weights is not None else defaults.get(task.preset, {
        "speed_tracking": 1.5, "forward_progress": .3, "alive": .15, "upright": .15,
        "energy": .04, "smoothness": .02, "fall": 1,
    })
    components = {key: value * terms[key] for key, value in weights.items()}
    if task.preset == "custom":
        reward = compile_reward(task.expression)({
            **terms, "x": x, "y": y, "z": z, "vx": vx, "vy": vy, "vz": vz,
            "speed": math.hypot(*velocity), "height": state["height"],
            "initial_height": state["initial_height"], "target_speed": target,
            "distance": distance, "time": state["elapsed"], "dt": state["dt"], "pi": math.pi,
        })
    else:
        reward = sum(components.values())
    if not math.isfinite(reward):
        raise ValueError("The task produced a non-finite reward.")
    return {"reward": float(reward), "terminated": task.terminate_on_fall and fallen,
            "components": components, "terms": terms, "distance": distance, "fallen": fallen}
