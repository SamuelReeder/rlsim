"""Mechanical actuator work integrated at physics substeps, not battery energy."""
import math
import numpy as np

class EffortMeter:
    def __init__(self):self.reset()

    def reset(self):
        self.absolute_work=self.positive_work=self.torque_squared=self.saturated=self.samples=self.seconds=0.

    def observe(self, force, velocity, torques, controls, dt):
        power=np.asarray(force)*np.asarray(velocity)
        work=float(np.sum(np.abs(power)))*dt
        self.absolute_work+=work
        self.positive_work+=float(np.sum(np.maximum(power,0)))*dt
        self.torque_squared+=float(np.sum(np.asarray(torques)**2))
        self.saturated+=int(np.count_nonzero(np.abs(controls)>=.98))
        self.samples+=len(controls);self.seconds+=dt
        return work

    def summary(self, distance, mass, gravity):
        return {'absolute_motor_work_j':self.absolute_work,'positive_motor_work_j':self.positive_work,
                'mean_motor_power_w':self.absolute_work/max(self.seconds,1e-12),
                'mechanical_cost_of_transport':self.absolute_work/(mass*gravity*distance) if distance>.1 else None,
                'torque_rms_nm':math.sqrt(self.torque_squared/max(1,self.samples)),
                'actuator_saturation_fraction':self.saturated/max(1,self.samples)}
