/** Serializable physics graph. Coordinates are metres; joint ranges are degrees.
 * Body poses are world-space rest poses. Colliders and joint axes are body-local.
 * This module intentionally has no rendering or ML dependency.
 */
const TYPES = new Set(['free', 'fixed', 'ball', 'hinge']);
const SHAPES = new Set(['capsule', 'sphere', 'box']);
const ID = /^[A-Za-z_][A-Za-z0-9_.-]{0,99}$/;
const finite = (value, min, max, label) => {
  if (typeof value !== 'number' || !Number.isFinite(value) || value < min || value > max) throw new Error(`${label} must be between ${min} and ${max}.`);
  return value;
};
function tuple(value, count, label, min=-10000, max=10000) {
  if (!Array.isArray(value) || value.length !== count) throw new Error(`${label} needs ${count} numbers.`);
  return value.map(number => finite(number, min, max, label));
}
function quaternion(value, label) {
  const q=tuple(value,4,label,-1,1), length=Math.hypot(...q);
  if (Math.abs(length-1)>1e-3) throw new Error(`${label} must be a unit quaternion.`);
  return q.map(x=>x/length);
}
const strings = (values, label) => {
  if (!Array.isArray(values) || values.length>256 || values.some(value=>typeof value!=='string'||value.length>1000)) throw new Error(`Invalid ${label}.`);
  return [...values];
};
export function validatePhysicalRig(input) {
  if (!input || input.version!==1 || !['humanoid','generic','rigid'].includes(input.profile)) throw new Error('Unsupported physical rig version or profile.');
  if (!Array.isArray(input.bodies) || !input.bodies.length || input.bodies.length>128) throw new Error('A physical rig needs 1–128 bodies.');
  const ids=new Set(), sources=new Set(); let controls=0;
  const bodies=input.bodies.map(source=>{
    if (!source || typeof source.id!=='string' || !ID.test(source.id) || ids.has(source.id)) throw new Error('Physical body IDs must be unique simple names.');
    ids.add(source.id);
    if (typeof source.label!=='string'||!source.label||source.label.length>200) throw new Error('Physical bodies need a short label.');
    if (source.parent!==null && (typeof source.parent!=='string'||!ID.test(source.parent))) throw new Error(`Invalid parent for ${source.label}.`);
    const j=source.joint;
    if (!j || !TYPES.has(j.type)) throw new Error(`Invalid joint type for ${source.label}.`);
    const axis=tuple(j.axis||[1,0,0],3,'Joint axis',-1,1), norm=Math.hypot(...axis);
    if (norm<1e-6) throw new Error('Joint axes must have a direction.');
    const range=tuple(j.range||[0,85],2,'Joint range',-180,180);
    if (j.type==='hinge'&&range[0]>=range[1]) throw new Error('Hinge lower limit must be below its upper limit.');
    if (j.type==='ball'&&(range[0]!==0||range[1]<=0)) throw new Error('Ball joint range must start at 0 with a positive angular limit.');
    const torque=finite(j.torque,0,10000,'Motor torque'), damping=finite(j.damping,0,1000,'Joint damping');
    if (torque>0) controls+=({ball:3,hinge:1,free:input.profile==='rigid'?6:0,fixed:0})[j.type];
    if (!Array.isArray(source.colliders)||!source.colliders.length||source.colliders.length>32) throw new Error(`Body ${source.label} needs 1–32 colliders.`);
    const colliders=source.colliders.map(c=>{
      if (!c||!SHAPES.has(c.type)) throw new Error('Unsupported collider shape.');
      return {type:c.type,position:tuple(c.position,3,'Collider position'),quaternion:quaternion(c.quaternion,'Collider quaternion'),
        size:tuple(c.size,{sphere:1,capsule:2,box:3}[c.type],'Collider dimensions',.00001,1000)};
    });
    const source_bones=strings(source.source_bones||[],'source bone paths');
    for(const path of source_bones){if(sources.has(path))throw new Error('A visual bone cannot map to two physical bodies.');sources.add(path);}
    return {id:source.id,label:source.label,parent:source.parent,position:tuple(source.position,3,'Body position'),
      quaternion:quaternion(source.quaternion,'Body quaternion'),mass:finite(source.mass,.000001,10000,'Body mass'),
      joint:{type:j.type,axis:axis.map(x=>x/norm),range,torque,damping},colliders,source_bones};
  });
  const byID=new Map(bodies.map(body=>[body.id,body]));
  if (!byID.has(input.root)) throw new Error('Physical root body does not exist.');
  for(const body of bodies){
    if(body.id===input.root){if(body.parent!==null||body.joint.type!=='free')throw new Error('The root body must have a free joint and no parent.');}
    else if(!byID.has(body.parent)||body.joint.type==='free')throw new Error('Every other body needs a parent and a non-free joint.');
    const seen=new Set();let node=body;
    while(node){if(seen.has(node.id))throw new Error('Physical rig parents contain a cycle.');seen.add(node.id);node=byID.get(node.parent);}
  }
  if(controls>256)throw new Error('The physical rig exceeds 256 motor controls. Reduce active joints or set some motor torques to zero.');
  let control;
  if(input.control!==undefined){
    const c=input.control;
    if(!c||!['torque','joint_target'].includes(c.mode))throw new Error('Control mode must be torque or joint_target.');
    if(c.mode==='joint_target'&&bodies.some(b=>b.joint.type==='free'&&b.joint.torque>0))
      throw new Error('Joint-target control cannot actuate a free root. Disable its motors first.');
    control={mode:c.mode,excursion:finite(c.excursion??30,1,85,'Target excursion'),
      stiffness:finite(c.stiffness??1,.1,3,'Servo stiffness'),damping_ratio:finite(c.damping_ratio??.7,.1,2,'Servo damping ratio')};
  }
  return {version:1,profile:input.profile,root:input.root,bodies,warnings:strings(input.warnings||[],'warnings'),repairs:strings(input.repairs||[],'repairs'),
    ...(control?{control}:{})};
}
export function colliderVolume(c){
  if(c.type==='box')return 8*c.size[0]*c.size[1]*c.size[2];
  const r=c.size[0];return 4*Math.PI*r*r*r/3+(c.type==='capsule'?Math.PI*r*r*2*c.size[1]:0);
}
const multiply=(a,b)=>[a[0]*b[0]-a[1]*b[1]-a[2]*b[2]-a[3]*b[3],a[0]*b[1]+a[1]*b[0]+a[2]*b[3]-a[3]*b[2],a[0]*b[2]-a[1]*b[3]+a[2]*b[0]+a[3]*b[1],a[0]*b[3]+a[1]*b[2]-a[2]*b[1]+a[3]*b[0]];
const conjugate=q=>[q[0],-q[1],-q[2],-q[3]];
const rotate=(q,p)=>multiply(multiply(q,[0,...p]),conjugate(q)).slice(1);
const nums=values=>values.map(x=>Number(x.toPrecision(9))).join(' ');
const escape=value=>String(value).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&apos;'}[c]));
export function physicalRigXML(name, input) {
  const rig=validatePhysicalRig(input), byID=new Map(rig.bodies.map(b=>[b.id,b])),motors=[];
  function bodyXML(body){
    const parent=byID.get(body.parent), inv=parent&&conjugate(parent.quaternion);
    const position=parent?rotate(inv,body.position.map((x,i)=>x-parent.position[i])):body.position;
    const q=parent?multiply(inv,body.quaternion):body.quaternion,j=body.joint;
    let joint='';
    if(j.type==='free')joint=`<freejoint name="root_free"/>`;
    else if(j.type!=='fixed')joint=`<joint name="joint_${body.id}" type="${j.type}" axis="${nums(j.axis)}" limited="true" range="${nums(j.range)}" damping="${j.damping}" armature="0.02"/>`;
    const axes=j.type==='ball'?3:j.type==='hinge'?1:j.type==='free'&&rig.profile==='rigid'?6:0;
    if(j.torque>0&&axes&&rig.control?.mode==='joint_target'){
      const angle=Math.min(rig.control.excursion,j.type==='ball'?j.range[1]*.9:rig.control.excursion)*Math.PI/180;
      const kp=j.torque/angle*rig.control.stiffness;
      if(j.type==='ball'){
        const bound=angle/Math.sqrt(3);
        motors.push(`<orientation name="target_${body.id}" joint="joint_${body.id}" kp="${kp}" dampratio="${rig.control.damping_ratio}" ctrlrange="${-bound} ${bound}" forcerange="0 ${j.torque}"/>`);
      }else{
        const low=Math.max(j.range[0]*Math.PI/180,-angle),high=Math.min(j.range[1]*Math.PI/180,angle);
        if(low>=high)throw new Error(`Target excursion does not overlap the limits of ${body.label}. Increase excursion or edit joint limits.`);
        motors.push(`<position name="target_${body.id}" joint="joint_${body.id}" kp="${kp}" dampratio="${rig.control.damping_ratio}" ctrlrange="${low} ${high}" forcerange="${-j.torque} ${j.torque}"/>`);
      }
    }else if(j.torque>0)for(let i=0;i<axes;i++){
      const gear=Array(6).fill(0);gear[i]=j.torque;
      motors.push(`<motor name="motor_${body.id}_${i}" joint="${j.type==='free'?'root_free':`joint_${body.id}`}" gear="${nums(gear)}" ctrllimited="true" ctrlrange="-1 1"/>`);
    }
    const totalVolume=body.colliders.reduce((sum,c)=>sum+colliderVolume(c),0);
    const geoms=body.colliders.map((c,i)=>`<geom name="geom_${body.id}_${i}" type="${c.type}" pos="${nums(c.position)}" quat="${nums(c.quaternion)}" size="${nums(c.size)}" mass="${body.mass*colliderVolume(c)/totalVolume}"/>`).join('');
    return `<body name="${body.id}" pos="${nums(position)}" quat="${nums(q)}">${joint}${geoms}${rig.bodies.filter(b=>b.parent===body.id).map(bodyXML).join('')}</body>`;
  }
  const tree=bodyXML(byID.get(rig.root));
  const schema=rig.control?.mode==='joint_target'?'<custom><text name="rlsim_action_schema" data="rlsim-joint-target-v1"/></custom>':'';
  return `<mujoco model="${escape(name)}"><compiler angle="degree" autolimits="true" inertiafromgeom="true"/><option timestep="0.005" gravity="0 0 -9.81" integrator="implicitfast"/>${schema}<default><geom contype="1" conaffinity="2" friction="1 0.05 0.01" rgba="0.45 0.55 0.61 1"/></default><worldbody><geom name="floor" type="plane" size="0 0 0.1" contype="2" conaffinity="1" rgba="0.94 0.94 0.92 1"/>${tree}</worldbody><actuator>${motors.join('')}</actuator></mujoco>`;
}
