"""Flat-floor template gait objective and measurements, mirrored in motion.mjs.
Ground contact is estimated from foot-sphere clearance with a 12 mm tolerance.
"""
import math
import numpy as np


def phase_targets(time, offset, legs, length):
    phase = (time / (.7 * math.sqrt(length / .84)) + offset) % 1
    phases = (phase + np.array([(i // 2 + i % 2) % 2 * .5 for i in range(legs)])) % 1
    swing = phases > .6
    height = np.where(swing, .09 * length * np.sin(np.pi * (phases - .6) / .4), 0)
    return phase, ~swing, height


def position_controls(action, qpos, qvel, body):
    # Learned targets, not a scripted joint trajectory. Recompute each substep.
    targets = np.tile([.45, -.9], body.legs) + .8 * action
    return np.clip((25 * (targets - qpos[3:]) - .7 * qvel[3:]) *
                   (body.body_mass / 4) / body.motor_strength, -1, 1)


class MotionTracker:
    def __init__(self, body):
        self.body = body
        self.length = body.upper_length + body.lower_length
        self.reset(np.zeros((body.legs, 3)))

    def reset(self, feet):
        n = self.body.legs
        self.previous = feet.copy()
        self.contacts = feet[:, 2] <= .012
        self.air = np.zeros(n)
        self.peak = np.zeros(n)
        self.lift_x = feet[:, 0].copy()
        self.strides = np.zeros(n, dtype=int)
        self.contact_samples = np.zeros(n)
        self.slip = self.slip_samples = self.samples = self.pitch2 = self.action_rate = 0.

    def update(self, feet, dt, pitch, action_rate):
        contacts = feet[:, 2] <= .012
        speed = (feet[:, 0] - self.previous[:, 0]) / dt
        lifting = self.contacts & ~contacts
        self.lift_x[lifting] = self.previous[lifting, 0]
        self.air[~contacts] += dt
        self.peak = np.maximum(self.peak, feet[:, 2])
        landing = ~self.contacts & contacts
        valid = landing & (self.air >= .12) & (self.peak >= .04 * self.length)
        valid &= np.abs(feet[:, 0] - self.lift_x) >= .06 * self.length
        self.strides += valid
        self.air[contacts] = 0
        self.peak[contacts] = 0
        # Ignore impact transitions when measuring persistent sliding.
        planted = contacts & self.contacts
        self.slip += float(np.sum(np.abs(speed[planted])))
        self.slip_samples += int(np.sum(planted))
        self.contact_samples += contacts
        self.samples += 1
        self.pitch2 += pitch * pitch
        self.action_rate += action_rate
        self.contacts = contacts
        self.previous = feet.copy()
        return float(np.mean(np.minimum(speed[planted] ** 2, 4))) if np.any(planted) else 0.

    def summary(self):
        return {"steps_per_leg": self.strides.tolist(),
                "contact_fraction": (self.contact_samples / max(1, self.samples)).tolist(),
                "foot_slip_m_s": self.slip / max(1, self.slip_samples),
                "pitch_rms": math.sqrt(self.pitch2 / max(1, self.samples)),
                "action_change": self.action_rate / max(1, self.samples)}


def walking_components(body, feet, contacts, time, offset, velocity, target, height, pitch, vz, pitch_rate, slip, smoothness, foot_drift_weight=.2):
    _, desired_contact, desired_height = phase_targets(time, offset, body.legs, body.upper_length + body.lower_length)
    length = body.upper_length + body.lower_length
    moving = math.exp(-2 * ((velocity - target) / max(target, .5)) ** 2)
    return {"gait_contacts": 1.2 * float(np.mean(contacts == desired_contact)) * (.25 + .75 * moving),
            "foot_clearance": .8 * float(np.mean(np.exp(-((feet[:, 2] - desired_height) / (.04 * length)) ** 2))) * (.25 + .75 * moving),
            "body_height": .3 * math.exp(-((height - (.88 * length + body.leg_radius * 1.3)) / (.15 * length)) ** 2),
            "level_body": .4 * math.exp(-8 * pitch * pitch),
            "foot_slip": -foot_drift_weight * slip, "body_bounce": -.08 * min(vz * vz, 10),
            "pitch_rate": -.03 * min(pitch_rate * pitch_rate, 20), "action_change": -.06 * smoothness}


def walking_pass(summary):
    g = summary.get("gait")
    return bool(g and not summary["fell"] and summary["seconds"] >= 19.99 and summary["speed_error"] <= .35
                and min(g["steps_per_leg"]) >= 6 and all(.15 <= x <= .85 for x in g["contact_fraction"])
                and g["foot_slip_m_s"] <= .35 and g["pitch_rms"] <= .25)
