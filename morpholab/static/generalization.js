"use strict";
let proportionDemo=null,proportionBusy=false,proportionLastResult=null;
const proportionInputs=['proportion-torso','proportion-upper','proportion-lower'];
const proportionRecords=new Map();
const currentProportions=()=>proportionInputs.map(id=>Number($(id).value));
function generalizationControls(){
  const locked=proportionBusy||state.busy||isTraining()||state.previewPending||!state.connected||state.engine!=='browser'||!proportionDemo;
  for(const id of [...proportionInputs,'generalization-policy','generalization-run','generalization-auto','generalization-speed'])$(id).disabled=locked;
  document.querySelectorAll('[data-proportions]').forEach(button=>button.disabled=locked);
}
async function updateProportions(){
  const {proportionRegion,PROPORTION_KEYS}=await import('./morphology.mjs');
  const scales=currentProportions();
  proportionInputs.forEach((id,i)=>{
    const length=(proportionDemo?.base_body?.[PROPORTION_KEYS[i]]??[.8,.42,.42][i])*scales[i];
    $(`${id}-value`).textContent=`${Math.round(scales[i]*100)}% · ${length.toFixed(2)} m`;
  });
  const entry=proportionDemo?.controllers[$('generalization-policy').value];
  const width=entry?.morphology?.width??proportionDemo?.width??.2;
  const withheld=entry?.morphology?.withhold_width??width;
  const range=`${Math.round((1-width)*100)}–${Math.round((1+width)*100)}%`;
  const region=proportionRegion(scales,width,withheld);
  $('proportion-region').textContent=region==='withheld region'
    ? `Withheld region: upper legs ≥${Math.round((1+withheld/2)*100)}% and lower legs ≤${Math.round((1-withheld/2)*100)}% were excluded from varied-body training.`
    : region==='outside training range'
      ? `Outside the ${range} training range. This tests extrapolation; failure is possible.`
      : `Within the ${range} training ranges. The varied policy saw many shapes here; the fixed policy trained only at 100%.`;
  const speeds=entry?.morphology?.speed_range;
  $('generalization-speed-field').hidden=!speeds;
  if(speeds){$('generalization-speed').min=speeds[0];$('generalization-speed').max=speeds[1];}
  else $('generalization-speed').value=1;
  $('generalization-speed-value').textContent=`${Number($('generalization-speed').value).toFixed(2)} m/s`;
  document.querySelector('[data-proportions="1,1.35,1.3"]').textContent=width>=.35?'Wide proportions':'Outside range';
}
async function generalizationRecord(key){
  if(proportionRecords.has(key))return proportionRecords.get(key);
  const entry=proportionDemo.controllers[key];
  const response=await fetch(`./demos/${entry.file}`);
  if(!response.ok)throw new Error('The bundled controller could not be loaded.');
  const bytes=await response.arrayBuffer();
  if(globalThis.crypto?.subtle){
    const hash=[...new Uint8Array(await crypto.subtle.digest('SHA-256',bytes))].map(n=>n.toString(16).padStart(2,'0')).join('');
    if(hash!==entry.sha256)throw new Error('The controller file does not match the experiment.');
  }
  const record=await browserBackend.call('validate',JSON.parse(new TextDecoder().decode(bytes)));
  record.id=`generalization-${key}`;record.name=entry.label;
  proportionRecords.set(key,record);
  return record;
}
async function runGeneralization(){
  if(proportionBusy||state.busy||isTraining()||state.previewPending||state.engine!=='browser'||!proportionDemo)return;
  proportionBusy=true;state.busy=true;controls();
  const key=$('generalization-policy').value,scales=currentProportions(),entry=proportionDemo.controllers[key];
  $('generalization-result').textContent='Loading the frozen controller…';
  try{
    const record=await generalizationRecord(key);
    await browserBackend.put(record);await refreshRuns(record.id);
    setEnvironment(record.config.environment);
    $('eval-speed').value=entry.morphology?.speed_range?Number($('generalization-speed').value):record.config.target_speed;$('eval-seed').value=proportionDemo.demo_seed;
    $('checkpoint-choice').value='latest';
    $('generalization-identity').textContent=`${entry.label} · checkpoint ${entry.fingerprint.slice(0,12)} at ${format(entry.steps)} steps · trained for ${format(entry.training_steps??entry.steps)} steps · seed ${entry.seed}. Weights and normalization are frozen.`;
    state.busy=false;
    const result=await infer(false,{proportions:scales});
    proportionLastResult=result?{...result,proportions:scales,controller:key}:null;
    if(result){
      const s=result.summary;
      const {walkingPass}=await import('./motion.mjs');
      $('generalization-result').textContent=`${s.fell?'Fell':s.speed_error<=.35&&s.seconds>=19.99?'Reached speed target':'Stayed up, but missed the speed target'} · ${format(s.distance,1)} m in ${format(s.seconds,1)} s · speed error ${format(s.speed_error,2)} m/s. No training performed.${s.gait?` Steps by leg: ${s.gait.steps_per_leg.join(" / ")}. Foot drift ${format(s.gait.foot_slip_m_s,2)} m/s. ${walkingPass(s)?"Passed":"Did not pass"} the gait check.`:""}`;
    }else $('generalization-result').textContent='Episode stopped or unavailable. Adjust the proportions and try again.';
  }catch(error){$('generalization-result').textContent=error.message;message(error.message,true);}
  finally{proportionBusy=false;state.busy=false;controls();}
}
async function loadGeneralization(){
  try{
    const response=await fetch('./demos/generalization.json');
    if(!response.ok)throw new Error('The proportion experiment is not included in this build.');
    proportionDemo=await response.json();
    $('generalization-controls').hidden=false;$('generalization-loading').hidden=true;
    await updateProportions();
    const result=await fetch('./demos/generalization-report.json');
    if(!result.ok)throw new Error('The experiment results could not be loaded.');
    const report=await result.json();
    const rows=[];
    for(const region of ['training range','withheld region','outside training range'])for(const condition of report.protocol.conditions){
      const runs=report.trials.filter(t=>t.condition===condition&&t.status==='completed');
      const values=runs.map(t=>{const cases=t.cases.filter(c=>c.region===region);return cases.reduce((sum,c)=>sum+c.success_rate,0)/cases.length;});
      const mean=values.reduce((a,b)=>a+b,0)/values.length;
      const sd=values.length>1?Math.sqrt(values.reduce((s,x)=>s+(x-mean)**2,0)/(values.length-1)):null;
      rows.push([region,condition,`${values.length}/${report.protocol.training_seeds.length}`,`${format(mean*100,0)}% ± ${format(sd===null?null:sd*100,0)}%`]);
    }
    $('generalization-evidence-content').replaceChildren(
      element('p',`${format(report.protocol.steps)} steps per run · ${report.protocol.training_seeds.length} paired training seeds · ${report.protocol.body_cases.length} test bodies · ${report.protocol.evaluation_seeds.length} episodes per body. Mean ± sample standard deviation across training runs.`,'control-hint'),
      resultsTable(['Test region','Training','Runs','Speed + survival'],rows),
      element('p',report.protocol.success_rule,'control-hint'),
      element('p','The original experiment measured speed and survival, not gait quality. These aggregate tests use native physics. The interactive demo uses browser physics and one checkpoint selected by validation, so a single replay can differ from the averages. Final test bodies were not used to select checkpoints.','control-hint'),
      ...(proportionDemo.notes||[]).map(note=>element('p',note,'control-hint')));
    const walkingResponse=await fetch('./demos/walking-study.json');
    if(walkingResponse.ok){
      const walking=await walkingResponse.json();
      if(walking.completed_at&&walking.controllers?.varied){
        proportionDemo.controllers.coordinated=walking.controllers.varied;
        const option=new Option('Coordinated · varied bodies','coordinated');
        $('generalization-policy').prepend(option);
        if(!proportionLastResult)$('generalization-policy').value='coordinated';
        const rows=[];
        for(const controller of ['varied','old varied'])for(const region of ['training range','withheld region','outside training range']){
          const cases=walking.cases.filter(c=>c.backend==='browser CPU'&&c.controller===controller&&c.region===region);
          const episodes=cases.flatMap(c=>c.episodes);
          rows.push([controller==='varied'?'Coordinated':'Original speed-only',region,`${cases.reduce((s,c)=>s+c.gait_passed,0)}/${episodes.length}`]);
        }
        const link=element('a','Download gait measurements');link.href='./demos/walking-study.json';link.download='';
        $('generalization-evidence-content').prepend(
          element('p','New coordinated-walking controller: the browser checks below require every leg to take repeated steps, limited foot drift and torso tilt, as well as speed and survival. One new training seed; several training changes together, so this is a development comparison.','control-hint'),
          resultsTable(['Controller','Test region','Gait checks passed'],rows),
          element('p',walking.gait_rule,'control-hint'),element('p',`Training budget: coordinated ${format(walking.controllers.varied.training_steps)} steps; original ${format(proportionDemo.controllers.varied.training_steps)} steps.`,'control-hint'),element('p',walking.limits,'control-hint'),link,
          element('p','Original three-seed experiment (speed and survival only):','control-hint'));
      }
    }
    // Optional broad-coverage study is published only after complete audits.
    const broadResponse=await fetch('./demos/locomotion-study.json');
    if(broadResponse.ok){
      const broad=await broadResponse.json();
      if(broad.completed_at&&broad.controller){
        proportionDemo.controllers.robust=broad.controller;
        $('generalization-policy').prepend(new Option('Coordinated · wider range','robust'));
        if(!proportionLastResult)$('generalization-policy').value='robust';
        const link=element('a','Download broad walking checks');link.href='./demos/locomotion-study.json';link.download='';
        $('generalization-evidence-content').prepend(
          element('p','Broader training: one frozen controller trained across 60–140% proportions and 0.5–1.5 m/s speed commands, excluding the marked upper/lower-leg corner. These are development results from one warm-started training seed.','control-hint'),
          resultsTable(['Controller','Audit','Gait checks passed'],broad.comparison_rows),
          element('p',broad.boundary_note,'control-hint'),element('p',broad.limits,'control-hint'),link);
      }
    }
    await updateProportions();
    const browserResponse=await fetch('./demos/generalization-browser.json');
    if(browserResponse.ok){
      const check=await browserResponse.json(),browserRows=[];
      for(const region of ['training range','withheld region','outside training range'])for(const controller of ['varied','fixed']){
        const cases=check.cases.filter(c=>c.region===region&&c.controller===controller);
        const episodes=cases.flatMap(c=>c.episodes);
        const passed=episodes.filter(e=>!e.fell&&e.seconds>=19.99&&e.speed_error<=.35).length;
        browserRows.push([region,controller==='varied'?'varied bodies':'fixed body',`${passed}/${episodes.length}`,`${episodes.filter(e=>e.fell).length}`]);
      }
      const download=element('a','Download browser checks');download.href='./demos/generalization-browser.json';download.download='';
      $('generalization-evidence-content').append(
        element('p','Bundled checkpoints in browser physics (CPU): one selected controller per condition, tested on the same body/episode suite. These are individual episodes, not independent training runs. WebGPU can produce different trajectories.','control-hint'),
        resultsTable(['Test region','Training','Passed episodes','Falls'],browserRows),download);
    }
  }catch(error){$('generalization-loading').hidden=false;$('generalization-loading').textContent=error.message;}
  finally{$('generalization-demo').setAttribute('aria-busy','false');controls();}
}
for(const id of proportionInputs){
  $(id).addEventListener('input',updateProportions);
  $(id).addEventListener('change',()=>{updateProportions();if($('generalization-auto').checked)runGeneralization();});
}
document.querySelectorAll('[data-proportions]').forEach(button=>button.addEventListener('click',()=>{
  button.dataset.proportions.split(',').forEach((value,i)=>$(proportionInputs[i]).value=value);
  updateProportions();runGeneralization();
}));
$('generalization-run').addEventListener('click',runGeneralization);
$('generalization-policy').addEventListener('change',async()=>{await updateProportions();runGeneralization();});
$('generalization-speed').addEventListener('input',updateProportions);
$('generalization-speed').addEventListener('change',()=>{if($('generalization-auto').checked)runGeneralization();});
loadGeneralization();
