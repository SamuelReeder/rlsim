export const PROPORTION_KEYS=['torso_length','upper_length','lower_length'];
export function validateMorphology(input={}){
  if(!input||typeof input!=="object"||Array.isArray(input))throw new Error("Invalid body-proportion settings.");
  const allowed=new Set(["enabled","width","ramp_steps","withhold_width","speed_range","boundary_fraction","robust_validation"]);
  if(Object.keys(input).some(k=>!allowed.has(k)))throw new Error("Unknown body-proportion setting.");
  const value={enabled:input.enabled??true,width:input.width??.2,ramp_steps:input.ramp_steps??131072};
  if(typeof value.enabled!=='boolean'||!Number.isFinite(value.width)||value.width<.05||value.width>.4||
     !Number.isInteger(value.ramp_steps)||value.ramp_steps<0||value.ramp_steps>2000000)
    throw new Error('Invalid body-proportion training settings.');
  if(input.withhold_width!=null){if(!Number.isFinite(input.withhold_width)||input.withhold_width<.05||input.withhold_width>value.width)throw new Error('Invalid withheld corner.');value.withhold_width=input.withhold_width;}
  if(input.speed_range!=null){const x=input.speed_range;if(!Array.isArray(x)||x.length!==2||x.some(v=>!Number.isFinite(v))||x[0]<.1||x[0]>x[1]||x[1]>3)throw new Error('Invalid speed range.');value.speed_range=x.slice();}
  if(input.boundary_fraction!=null){const x=input.boundary_fraction;if(!Number.isFinite(x)||x<0||x>.5)throw new Error('Invalid boundary fraction.');value.boundary_fraction=x;}
  if(input.robust_validation!=null){if(typeof input.robust_validation!=='boolean')throw new Error('Invalid validation setting.');value.robust_validation=input.robust_validation;}
  return value;
}
export function proportionRegion(scales,width=.2,withholdWidth=width){
  if(scales.some(s=>Math.abs(s-1)>width+1e-9))return 'outside training range';
  if(scales[1]>=1+withholdWidth/2-1e-9&&scales[2]<=1-withholdWidth/2+1e-9)return 'withheld region';
  return 'training range';
}
export function proportionBody(config,scales,validateBody){
  if(!config.morphology||config.body.kind==='mjcf')throw new Error('This controller does not declare a compatible proportion experiment.');
  if(!Array.isArray(scales)||scales.length!==3||scales.some(s=>!Number.isFinite(s)||s<.6||s>1.4))
    throw new Error('Choose three proportion multipliers between 0.6 and 1.4.');
  return validateBody({...config.body,...Object.fromEntries(PROPORTION_KEYS.map((key,i)=>[key,config.body[key]*scales[i]]))});
}
