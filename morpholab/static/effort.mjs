/** Physics-substep mechanical work, matching effort.py. Not battery energy. */
export class EffortMeter{
  constructor(){this.reset();}
  reset(){this.absoluteWork=this.positiveWork=this.torqueSquared=this.saturated=this.samples=this.seconds=0;}
  observe(force,velocity,torques,controls,dt){
    let work=0;
    for(let i=0;i<force.length;i++){const power=force[i]*velocity[i];work+=Math.abs(power)*dt;this.positiveWork+=Math.max(power,0)*dt;}
    for(const t of torques)this.torqueSquared+=t*t;
    for(const c of controls)if(Math.abs(c)>=.98)this.saturated++;
    this.absoluteWork+=work;this.samples+=controls.length;this.seconds+=dt;return work;
  }
  summary(distance,mass,gravity){return {absolute_motor_work_j:this.absoluteWork,positive_motor_work_j:this.positiveWork,
    mean_motor_power_w:this.absoluteWork/Math.max(this.seconds,1e-12),
    mechanical_cost_of_transport:distance>.1?this.absoluteWork/(mass*gravity*distance):null,
    torque_rms_nm:Math.sqrt(this.torqueSquared/Math.max(1,this.samples)),actuator_saturation_fraction:this.saturated/Math.max(1,this.samples)};}
}
