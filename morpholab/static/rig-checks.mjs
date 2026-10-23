/** Short, isolated MuJoCo diagnostics. These test setup, not learned walking. */
import {ImportedWalker} from './imported-physics.mjs';
import {validatePhysicalRig} from './physical-rig.mjs';

export async function checkPhysicalRig(mj, body, environment={}, yieldNow=async()=>{}) {
  const checks=[], add=(id,level,message,details={})=>checks.push({id,level,message,...details});
  const result=()=>({format:'rlsim-physics-checks-v1',status:checks.some(c=>c.level==='error')?'needs-fixes':checks.some(c=>c.level==='warning')?'review':'passed',
    checks,scope:'Rest pose, collision geometry, motor response and two seconds without a learned controller. Does not certify walking, anatomy or physical accuracy.'});
  let env, buffer;
  try {
    if(body.physical_rig)validatePhysicalRig(body.physical_rig);
    env=new ImportedWalker(mj,body,environment,{randomize:false,seed:71,task:{preset:'balance',terminate_on_fall:false}});
    const m=env.model,d=env.data;
    add('structure','pass',`${body.physical_rig?'Connected physics model':'Physics model'} compiled: ${m.nbody-1} parts, ${m.njnt} joints, ${m.nu} controls.`);
    const skeletonNotes=body.metadata?.auto_rig?.skeleton_review?.findings;
    if(Array.isArray(skeletonNotes))for(const note of skeletonNotes.slice(0,3))add('skeleton-coverage','warning',String(note.message).slice(0,500));
    if(body.physical_rig?.profile==='generic')add('joint-assumptions','warning','Using a generic skeleton. Joint axes and limits need review; unnamed bones do not identify knees, ankles or other anatomical constraints.');
    const reset=()=>{mj.mj_resetData(m,d);mj.mj_forward(m,d);};
    reset();
    const masses=Array.from(m.body_mass).filter(x=>x>0),total=masses.reduce((a,b)=>a+b,0);
    if(!masses.length || !Number.isFinite(total))throw new Error('The body needs finite, positive mass.');
    const ratio=Math.max(...masses)/Math.min(...masses);
    add('mass',ratio>1000?'warning':'pass',ratio>1000?'Very uneven part masses may make simulation difficult. Review small parts and clothing.':`Mass distribution is finite (${total.toFixed(2)} kg total).`,{total_mass:total,mass_ratio:ratio});
    const height=Math.max(.05,...Array.from({length:m.ngeom},(_,i)=>d.geom_xpos[i*3+2]+m.geom_rbound[i]));
    const tolerance=Math.max(.002,height*.01);
    buffer=new mj.DoubleBuffer(6);
    let floorDepth=0;const overlaps=[];let tested=0,partial=false;
    const names=i=>mj.mj_id2name(m,mj.mjtObj.mjOBJ_BODY.value,m.geom_bodyid[i])||`part ${m.geom_bodyid[i]}`;
    for(let i=0;i<m.ngeom;i++)for(let j=i+1;j<m.ngeom;j++){
      const a=m.geom_bodyid[i],b=m.geom_bodyid[j];
      if(a===b)continue;
      const floor=m.geom_type[i]===0||m.geom_type[j]===0;
      if(!floor&&(m.body_parentid[a]===b||m.body_parentid[b]===a))continue;
      const separation=Math.hypot(...[0,1,2].map(k=>d.geom_xpos[i*3+k]-d.geom_xpos[j*3+k]));
      if(!floor && separation>m.geom_rbound[i]+m.geom_rbound[j])continue;
      if(tested++>=20000){partial=true;continue;}
      const distance=mj.mj_geomDistance(m,d,i,j,1,buffer);
      if(floor)floorDepth=Math.max(floorDepth,-distance);
      else if(distance < -tolerance && overlaps.length<12)overlaps.push(`${names(i)} / ${names(j)}`);
    }
    add('floor',floorDepth>tolerance?'warning':'pass',floorDepth>tolerance?'Collision shapes start below the floor. Review body placement before training.':'The rest pose has no deep floor penetration.',{penetration_m:floorDepth});
    add('overlaps',overlaps.length||partial?'warning':'pass',overlaps.length?`Non-neighbor collision shapes overlap: ${overlaps.join(', ')}. Self-contact remains disabled.`:partial?'Large rig: collision-pair check was capped; inspect the collision overlay.':'No deep overlaps found between non-neighbor collision shapes.',{pairs_tested:tested,partial});
    await yieldNow();
    const option=m.opt, gravity=Array.from(option.gravity);option.gravity.set([0,0,0]);option.delete();
    const contact=m.geom_contype.slice();m.geom_contype.fill(0);
    const responses=[];
    for(let actuator=0;actuator<env.actionSize;actuator++){
      let response=0;const traces=[];
      for(const direction of [-1,1]){
        reset();const action=Array(env.actionSize).fill(0);action[actuator]=direction*.15;
        const joint=m.actuator_trnid[actuator*2],dof=joint>=0?m.jnt_dofadr[joint]:0;
        const trace=[];
        for(let step=0;step<8;step++){
          const before=d.time;env.step(action);
          if(Math.abs(d.time-before-env.dt)>1e-7)throw new Error("MuJoCo reset during the motor probe. Review torque, mass and damping.");
          if(!d.qpos.every(Number.isFinite)||!d.qvel.every(Number.isFinite)||Math.max(...d.qvel.map(Math.abs))>1000)throw new Error('Motor probe became numerically unstable. Reduce torque or review joint limits, masses and damping.');
          // Compare opposite-command trajectories: passive dynamics or another
          // servo holding rest should not make an inert control appear useful.
          const width=joint>=0&&m.jnt_type[joint]===1?3:1;
          trace.push(...d.qvel.slice(dof,dof+width));
        }
        traces.push(trace);
      }
      for(let i=0;i<traces[0].length;i++)response=Math.max(response,Math.abs(traces[1][i]-traces[0][i])/2);
      responses.push({name:env.actionNames[actuator],peak_response_speed:response});
      if(actuator%8===0)await yieldNow();
    }
    const quiet=responses.filter(r=>r.peak_response_speed<1e-5);
    add('motors',!responses.length||quiet.length?'warning':'pass',!responses.length?'No motors: this body cannot learn active movement.':quiet.length?`${quiet.length} controls produced almost no joint movement. Review their limits and strength.`:`All ${responses.length} controls responded to small positive and negative commands in zero gravity.`,{responses});
    m.geom_contype.set(contact);const restored=m.opt;restored.gravity.set(gravity);restored.delete();
    env.reset(71);const start=env._position()[2];let fell=false,maxSpeed=0;
    for(let step=0;step<100;step++){
      const before=d.time,sample=env.step(Array(env.actionSize).fill(0));fell ||= sample.info.fallen;
      if(Math.abs(d.time-before-env.dt)>1e-7)throw new Error("MuJoCo reset during the passive check. Review the physics setup.");
      if(!sample.obs.every(Number.isFinite))throw new Error('The passive simulation produced non-finite values. Review masses, joint limits and collision geometry.');
      maxSpeed=Math.max(maxSpeed,...Array.from(d.qvel,Math.abs));
      if(step%20===0)await yieldNow();
    }
    if(maxSpeed>1000)throw new Error('The passive simulation became unstable. Review collision geometry and motor settings.');
    add('simulation','pass','Two seconds completed with finite physics states and no simulation resets.',{max_joint_speed:maxSpeed});
    add('support','info',fell?'The body falls without a learned controller. This alone is not a setup failure.':'The body stayed above the fall threshold during the short passive check.',{fell,initial_height:start,final_height:env._position()[2]});
    add('assumptions','info','Joint limits, mass distribution and motor strength are estimates. SkinTokens predicts an animation skeleton; it does not measure anatomy or guarantee locomotion.');
  }catch(error){add('invalid','error',error.message||String(error));}
  finally{buffer?.delete();env?.close();}
  return result();
}
