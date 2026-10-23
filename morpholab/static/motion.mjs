/** Flat-floor template gait objective and measured motion, matching motion.py. */
const clip=(x,a,b)=>Math.min(b,Math.max(a,x));
export function phaseTargets(time,offset,legs,length){
  const phase=(time/(.7*Math.sqrt(length/.84))+offset)%1;
  const phases=Array.from({length:legs},(_,i)=>(phase+((Math.floor(i/2)+i%2)%2)*.5)%1);
  return {phase,contacts:phases.map(p=>p<=.6),heights:phases.map(p=>p>.6?.09*length*Math.sin(Math.PI*(p-.6)/.4):0)};
}
export function positionControls(action,qpos,qvel,body){
  return Array.from(action,(a,i)=>clip((25*((i%2?-.9:.45)+.8*a-qpos[i+3])-.7*qvel[i+3])*(body.body_mass/4)/body.motor_strength,-1,1));
}
export class MotionTracker{
  constructor(body){this.body=body;this.length=body.upper_length+body.lower_length;this.reset(Array.from({length:body.legs},()=>[0,0,0]));}
  reset(feet){
    this.previous=feet.map(f=>f.slice());this.contacts=feet.map(f=>f[2]<=.012);
    this.air=feet.map(()=>0);this.peak=feet.map(()=>0);this.liftX=feet.map(f=>f[0]);
    this.strides=feet.map(()=>0);this.contactSamples=feet.map(()=>0);
    this.slip=this.slipSamples=this.samples=this.pitch2=this.actionRate=0;
  }
  update(feet,dt,pitch,actionRate){
    const contacts=feet.map(f=>f[2]<=.012);let slip=0,count=0;
    feet.forEach((f,i)=>{
      const speed=(f[0]-this.previous[i][0])/dt;
      if(this.contacts[i]&&!contacts[i])this.liftX[i]=this.previous[i][0];
      if(!contacts[i])this.air[i]+=dt;
      this.peak[i]=Math.max(this.peak[i],f[2]);
      if(!this.contacts[i]&&contacts[i]&&this.air[i]>=.12&&this.peak[i]>=.04*this.length&&Math.abs(f[0]-this.liftX[i])>=.06*this.length)this.strides[i]++;
      if(contacts[i]){this.air[i]=0;this.peak[i]=0;this.contactSamples[i]++;}
      if(contacts[i]&&this.contacts[i]){this.slip+=Math.abs(speed);this.slipSamples++;slip+=Math.min(speed**2,4);count++;}
    });
    this.samples++;this.pitch2+=pitch*pitch;this.actionRate+=actionRate;
    this.contacts=contacts;this.previous=feet.map(f=>f.slice());return count?slip/count:0;
  }
  summary(){return {steps_per_leg:this.strides.slice(),contact_fraction:this.contactSamples.map(x=>x/Math.max(1,this.samples)),foot_slip_m_s:this.slip/Math.max(1,this.slipSamples),pitch_rms:Math.sqrt(this.pitch2/Math.max(1,this.samples)),action_change:this.actionRate/Math.max(1,this.samples)};}
}
export function walkingComponents(body,feet,contacts,time,offset,velocity,target,height,pitch,vz,pitchRate,slip,smoothness,footDriftWeight=.2){
  const length=body.upper_length+body.lower_length,desired=phaseTargets(time,offset,body.legs,length);
  const moving=Math.exp(-2*((velocity-target)/Math.max(target,.5))**2),gate=.25+.75*moving;
  return {gait_contacts:1.2*contacts.filter((c,i)=>c===desired.contacts[i]).length/body.legs*gate,
    foot_clearance:.8*feet.reduce((s,f,i)=>s+Math.exp(-(((f[2]-desired.heights[i])/(.04*length))**2)),0)/body.legs*gate,
    body_height:.3*Math.exp(-(((height-(.88*length+body.leg_radius*1.3))/(.15*length))**2)),
    level_body:.4*Math.exp(-8*pitch*pitch),foot_slip:-footDriftWeight*slip,body_bounce:-.08*Math.min(vz*vz,10),pitch_rate:-.03*Math.min(pitchRate*pitchRate,20),action_change:-.06*smoothness};
}
export function walkingPass(s){const g=s?.gait;return !!g&&!s.fell&&s.seconds>=19.99&&s.speed_error<=.35&&Math.min(...g.steps_per_leg)>=6&&g.contact_fraction.every(x=>x>=.15&&x<=.85)&&g.foot_slip_m_s<=.35&&g.pitch_rms<=.25;}
