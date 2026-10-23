"use strict";

const $ = id => document.getElementById(id);
const bodyInputs = [...document.querySelectorAll("input[data-field]")];
const advancedFields = [
  ["hidden-size","hidden_size"],["hidden-layers","hidden_layers"],
  ["learning-rate","learning_rate"],["batch-size","batch_size"],["epochs","epochs"],
  ["rollout-steps","rollout_steps"],["discount","gamma"],["entropy-coef","entropy_coef"],
  ["clip-range","clip_range"],
  ["target-kl","target_kl"],["evaluation-interval","evaluation_interval"],["evaluation-episodes","evaluation_episodes"],
];
const advancedInputs = advancedFields.map(([id])=>$(id));
const adviceFields = [
  ["hidden-size","hidden_size"],["hidden-layers","hidden_layers"],["training-budget","total_steps"],
  ["n-envs","n_envs"],["rollout-steps","rollout_steps"],["batch-size","batch_size"],
  ["learning-rate","learning_rate"],["epochs","epochs"],["discount","gamma"],
  ["clip-range","clip_range"],["entropy-coef","entropy_coef"],
  ["target-speed","target_speed"],["curriculum","curriculum"],["randomize","randomize"],
  ["curriculum-mode","curriculum_mode"],
];
const bodyImportFields = ["import-entry","import-scale","import-mass","import-motor-strength","import-rig","import-profile","import-floating-base","import-auto-actuators","import-add-floor"];
const rigEditorFields = ["rig-body","rig-parent","rig-mass","rig-joint","rig-limit-min","rig-limit-max","rig-torque","rig-damping","rig-axis-x","rig-axis-y","rig-axis-z"];
const rigControlFields=["rig-control-mode","rig-target-excursion","rig-servo-stiffness","rig-servo-damping"];
const locomotionFields = ["locomotion-mass","locomotion-strength","locomotion-lock-fingers"];
const taskFields = ["task-preset","foot-drift-weight","mechanical-work-weight","foot-only-support","episode-seconds","tracked-body","target-x","target-y","target-z","terminate-on-fall","min-height-ratio","max-tilt","reward-expression"];
const defaultReward="1.5*speed_tracking + 0.3*forward_progress + 0.15*alive + 0.15*upright + 0.04*energy + 0.02*smoothness + fall";
const defaultTask=()=>({preset:"velocity",episode_seconds:20,target:[2,0,.8],tracked_body:"",terminate_on_fall:true,min_height_ratio:.35,max_tilt_degrees:64});
const advancedDefaults = engine => ({
  hidden_size:engine==="server" ? 256 : 64, hidden_layers:2, learning_rate:0.0003,
  batch_size:engine==="server" ? 512 : 128, epochs:engine==="server" ? 10 : 4,
  rollout_steps:engine==="server" ? 256 : 128, gamma:0.99, entropy_coef:0.005, clip_range:0.2,
  target_kl:.03,evaluation_interval:16384,evaluation_episodes:3,best_checkpoint:true,
});
const activeStatuses = new Set(["starting", "training", "stopping"]);
const state = {
  body: null, presets: {}, environment: {gravity:9.81, friction:1}, hardware: {},
  connected: false, busy: false, previewPending: false, previewVersion: 0,
  status: {status:"idle",history:[]}, runs: [], selectedRun: "", tab:"body",
  mode:"preview", frames:[], replayTime:0, playing:false, pendingFrame:null, shownBody:null,
  engine:new URLSearchParams(location.search).get("engine")==="server" ? "server" : "browser",
  generation:0, statusRevision:0, initializing:false, defaults:{}, modelInfo:null,
  importBundle:null, renderVersion:0, rigDraft:null, rigDirty:false, modelSelectionCleared:false,
  replayLoading:false, replayRequest:0, replayBuffering:false, replayControl:null,
  advice:null, adviceUndo:null, adviceLoading:false, adviceRevision:0,
  locomotionAdvice:null, locomotionUndo:null, locomotionLoading:false, locomotionRevision:0,
};
const view = new PhysicsView($("simulation"));
let previewTimer, pollTimer, lastRunId, lastStatus, lastAnimation = performance.now();
let previewController;
let browserBackend;
const isTraining = () => activeStatuses.has(state.status.status);
const format = (n, digits=0) => Number.isFinite(n) ? n.toLocaleString(undefined,{maximumFractionDigits:digits}) : "—";
const selectedModel = () => state.runs.find(r => r.id === state.selectedRun);
const previewBodyConfig = () => state.tab === "run" && selectedModel() ? selectedModel().config.body : state.body;
const importedBody = body => body?.kind==="mjcf";
const bodyValidationInputs = () => [...(importedBody(state.body) ? [] : bodyInputs),"body-name"];

function message(text, error=false,detail="") {
  $("status-text").textContent = text;
  $("status-text").classList.toggle("error",error);
  $("status-text").title=detail;
}
async function api(path, body, signal, options) {
  const generation=state.generation,engine=state.engine;
  try {
    let result;
    if(engine==="browser") {
      browserBackend ||= new BrowserBackend();
      result=await browserBackend.request(path,body,options);
    } else {
      const response = await fetch(`.${path}`,{...(body===undefined ? {} : {method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(body)}),signal:signal || AbortSignal.timeout(60000)});
      result=await response.json().catch(()=>({}));
      if (!response.ok) {
        const detail = result.detail;
        throw new Error(Array.isArray(detail) ? detail.map(e=>`${e.loc.slice(1).join(".")}: ${e.msg}`).join("; ") : detail || `Request failed (${response.status})`);
      }
    }
    if(generation!==state.generation || signal?.aborted)throw new DOMException("Request superseded","AbortError");
    return result;
  } catch(error) {
    if(generation!==state.generation || signal?.aborted)throw new DOMException("Request superseded","AbortError");
    throw error;
  }
}
function showHardware(hardware) {
  state.hardware={...state.hardware,...hardware};
  const device=String(state.hardware.training_device || "cpu").toUpperCase();
  const location=state.hardware.execution_location==="browser" ? "This device" : "Server";
  $("hardware-label").textContent=`${device==="WEBGPU" ? "WebGPU" : device} · ${location}`;
  $("hardware-label").title=[
    `Training: ${device} on ${location.toLowerCase()}.`,
    state.hardware.gpu_name,
    "MuJoCo physics: CPU.",
    state.hardware.fallback_reason,
  ].filter(Boolean).join(" ");
}
function fieldsValid(ids, report=false) {
  for (const id of ids) {
    const input = typeof id === "string" ? $(id) : id;
    if (!input.checkValidity() || (input.type==="number" && !Number.isFinite(Number(input.value))) || input.value.trim()==="") {
      if (report) {
        const panel=input.closest("[role=tabpanel]");
        if(panel?.hidden)selectTab(panel.id.replace("-panel",""),false);
        let details=input.closest("details");
        while(details){details.open=true;details=details.parentElement.closest("details");}
        input.focus();input.reportValidity();
      }
      return false;
    }
  }
  return true;
}
function setAdvanced(config={}) {
  invalidateAdvice();
  const defaults={...advancedDefaults(state.engine),...state.defaults};
  for(const [id,key] of advancedFields)$(id).value=config[key] ?? defaults[key];
  $("best-checkpoint").checked=config.best_checkpoint ?? defaults.best_checkpoint;
  $("curriculum-mode").value=config.curriculum_mode??"speed";
}
function readAdvanced() {
  return {...Object.fromEntries(advancedFields.map(([id,key])=>[key,Number($(id).value)])),
    best_checkpoint:$("best-checkpoint").checked,curriculum_mode:$("curriculum-mode").value};
}
function taskVisibility() {
  const preset=$("task-preset").value;
  $("task-target-fields").hidden=!["target","custom"].includes(preset);
  $("custom-reward-field").hidden=preset!=="custom";
  $("walking-help").hidden=preset!=="walk";
  $("walking-drift-field").hidden=preset!=="walk";
  $("walking-effort-fields").hidden=preset!=="walk";
  if(preset==="walk")$("curriculum-mode").value="speed";
}
function updateTrackedBodies(info=state.modelInfo,selected=$("tracked-body").value) {
  const names=[...new Set((info?.bodies || []).map(body=>body.name).filter(Boolean))];
  if(selected && !names.includes(selected)){
    if(info)selected="";
    else names.push(selected);
  }
  $("tracked-body").replaceChildren(new Option("Root body",""),...names.map(name=>new Option(name,name)));
  $("tracked-body").value=selected;
}
function setTask(task=null) {
  invalidateAdvice();
  const config={...defaultTask(),...task};
  if(config.preset!=="custom" && config.weights!=null){
    config.preset="custom";
    config.expression=Object.entries(config.weights).map(([term,weight])=>`${weight}*${term}`).join(" + ") || "0";
  }
  $("foot-drift-weight").value=config.foot_drift_weight??.2;
  $("mechanical-work-weight").value=config.mechanical_work_weight??0;
  $("foot-only-support").checked=config.foot_only_support??false;
  $("task-preset").value=config.preset;$("episode-seconds").value=config.episode_seconds;
  for(const [index,id] of ["target-x","target-y","target-z"].entries())$(id).value=config.target[index];
  updateTrackedBodies(state.modelInfo,config.tracked_body);
  $("terminate-on-fall").checked=config.terminate_on_fall;
  $("min-height-ratio").value=config.min_height_ratio;$("max-tilt").value=config.max_tilt_degrees;
  $("reward-expression").value=config.expression || defaultReward;
  taskVisibility();
}
function readTask() {
  const task={
    preset:$("task-preset").value,episode_seconds:Number($("episode-seconds").value),
    target:["target-x","target-y","target-z"].map(id=>Number($(id).value)),
    tracked_body:$("tracked-body").value,terminate_on_fall:$("terminate-on-fall").checked,
    min_height_ratio:Number($("min-height-ratio").value),max_tilt_degrees:Number($("max-tilt").value),
  };
  if(task.preset==="walk"){
    task.foot_drift_weight=Number($("foot-drift-weight").value);
    task.mechanical_work_weight=Number($("mechanical-work-weight").value);
    task.foot_only_support=$("foot-only-support").checked;
  }
  if(task.preset==="custom")task.expression=$("reward-expression").value.trim();
  return JSON.stringify(task)===JSON.stringify(defaultTask()) ? null : task;
}
function taskValidationInputs() {
  return ["episode-seconds","min-height-ratio","max-tilt",
    ...($("task-preset").value==="walk"?["foot-drift-weight","mechanical-work-weight"]:[]),
    ...(!$("task-target-fields").hidden ? ["target-x","target-y","target-z"] : []),
    ...(!$("custom-reward-field").hidden ? ["reward-expression"] : [])];
}
function readEnvironment() {
  return {gravity:Number($("gravity").value),friction:Number($("friction").value)};
}
function setEnvironment(env) {
  invalidateAdvice();
  invalidateLocomotion();
  state.environment = {...{gravity:9.81,friction:1},...env};
  $("gravity").value=state.environment.gravity;
  $("friction").value=state.environment.friction;
}
function setBody(body,modelInfo=null) {
  invalidateAdvice();
  invalidateLocomotion();
  state.body=structuredClone(body);
  resetRigChecks();
  state.modelInfo=modelInfo || body.metadata?.model_info || null;
  $("body-name").value=body.name;
  $("import-warnings").hidden=true;$("import-warnings").textContent="";
  const imported=importedBody(body);
  $("body-name").maxLength=imported ? 120 : 60;
  $("body-dimensions").hidden=imported;$("body-mechanics").hidden=imported;
  $("imported-body-info").hidden=!imported;$("reset-design").hidden=imported;
  let option=$("body-preset").querySelector('[value="imported"]');
  if(!option){option=new Option("Imported body","imported");option.disabled=true;$("body-preset").add(option);}
  option.hidden=!imported;
  if(!imported)for (const input of bodyInputs) input.value=body[input.dataset.field];
  const preset=Object.keys(state.presets).find(k=>state.presets[k].legs===body.legs);
  $("body-preset").value=imported ? "imported" : preset || "quadruped";
  updateModelInfo(state.modelInfo);
  setPhysicalRig();
}
function clearSelectedModel() {
  state.selectedRun="";state.modelSelectionCleared=true;$("saved-run").value="";
}
function selectSavedModel(id) {
  const changed=id!==state.selectedRun;
  state.selectedRun=id;
  // Default a newly selected model to its trained command. A list refresh must
  // preserve any speed the user deliberately chose for the same model.
  const speed=selectedModel()?.config.target_speed;
  if(changed && Number.isFinite(speed))$("eval-speed").value=speed;
  if(changed)$("checkpoint-choice").value=selectedModel()?.best ? "best" : "latest";
}
function selectedRigBody() {
  return state.rigDraft?.bodies.find(body=>body.id===$("rig-body").value);
}
const rigNumber=value=>Number.isFinite(value) ? String(Number(value.toPrecision(7))) : "";
function setPhysicalRig() {
  const rig=state.body?.physical_rig;
  state.rigDraft=rig ? structuredClone(rig) : null;state.rigDirty=false;
  $("physical-rig-report").hidden=!rig;
  $("authored-physics-note").hidden=!importedBody(state.body) || Boolean(rig) || Boolean(state.body?.visual);
  $("legacy-physics-note").hidden=!importedBody(state.body) || Boolean(rig) || !state.body?.visual;
  $("physical-rig-editor").open=false;
  $("locomotion-setup").hidden=rig?.profile!=="humanoid";
  $("locomotion-setup").open=false;
  if(!rig)return;
  $("rig-control-mode").value=rig.control?.mode??"torque";
  $("rig-target-excursion").value=rig.control?.excursion??30;
  $("rig-servo-stiffness").value=rig.control?.stiffness??1;
  $("rig-servo-damping").value=rig.control?.damping_ratio??.7;
  $("rig-servo-fields").hidden=$("rig-control-mode").value!=="joint_target";
  $("locomotion-mass").value=rigNumber(rig.bodies.reduce((sum,body)=>sum+body.mass,0));
  $("locomotion-strength").value=1;$("locomotion-lock-fingers").checked=true;
  const profile={humanoid:"Humanoid",generic:"Generic skeleton",rigid:"Rigid body"}[rig.profile] || "Generated";
  $("rig-summary").textContent=`${profile} · ${rig.bodies.length} physical parts · ${format(rig.bodies.reduce((sum,body)=>sum+body.mass,0),2)} kg`;
  const report=state.body.metadata?.rig_report || {};
  const notes=[...(report.repairs || rig.repairs || []),...(report.warnings || rig.warnings || [])];
  $("rig-findings-list").replaceChildren(...notes.map(note=>{
    const item=document.createElement("li");
    item.textContent=typeof note==="string" ? note : note.message || JSON.stringify(note);
    return item;
  }));
  $("rig-findings").hidden=!notes.length;
  $("rig-body").replaceChildren(...rig.bodies.map(body=>new Option(body.label || body.id,body.id)));
  $("rig-body").value=rig.root;
  showRigBody();
}
function showRigBody() {
  const body=selectedRigBody();if(!body)return;
  const root=body.id===state.rigDraft.root;
  // Parent choices exclude descendants, so ordinary edits cannot introduce cycles.
  const descendants=new Set([body.id]);
  let previous=-1;
  while(previous!==descendants.size){
    previous=descendants.size;
    for(const part of state.rigDraft.bodies)if(descendants.has(part.parent))descendants.add(part.id);
  }
  $("rig-parent").replaceChildren(...(root ? [new Option("World (root)","")] :
    state.rigDraft.bodies.filter(part=>!descendants.has(part.id)).map(part=>new Option(part.label || part.id,part.id))));
  $("rig-parent").value=body.parent || "";
  $("rig-mass").value=rigNumber(body.mass);$("rig-joint").value=body.joint.type;
  $("rig-joint").querySelector('[value="free"]').hidden=!root;
  for(const option of $("rig-joint").options)if(option.value!=="free")option.hidden=root;
  const [min,max]=body.joint.range || [0,85];
  $("rig-limit-min").value=rigNumber(min);$("rig-limit-max").value=rigNumber(max);
  $("rig-torque").value=rigNumber(body.joint.torque ?? 60);$("rig-damping").value=rigNumber(body.joint.damping ?? 1);
  ["rig-axis-x","rig-axis-y","rig-axis-z"].forEach((id,index)=>{
    const value=(body.joint.axis || [1,0,0])[index];
    $(id).value=rigNumber(Math.abs(value)<1e-8 ? 0 : value);
  });
  $("rig-collider-fields").replaceChildren(...body.colliders.map((collider,index)=>{
    const section=document.createElement("div");section.className="rig-collider";
    const label=document.createElement("span");label.className="standalone-label";
    label.textContent=`Collider ${index+1} · ${collider.type} · metres`;section.append(label);
    const names=collider.type==="capsule" ? ["Radius","Half-length"] : collider.type==="box" ? ["Half X","Half Y","Half Z"] : ["Radius"];
    for(const [axis,name] of names.entries()){
      const row=document.createElement("div");row.className="field";
      const label=document.createElement("label"),input=document.createElement("input");
      input.id=`rig-collider-${index}-${axis}`;input.type="number";input.min="0.00001";input.max="1000";input.step="any";input.required=true;input.value=rigNumber(collider.size[axis]);
      input.dataset.rigCollider=String(index);input.dataset.rigDimension=String(axis);
      label.htmlFor=input.id;label.textContent=name;row.append(label,input);section.append(row);
    }
    return section;
  }));
  rigJointVisibility();updateRigDraftMessage();controls();
}
function rigJointVisibility() {
  const type=$("rig-joint").value;
  const articulated=["hinge","ball"].includes(type),rigidRoot=type==="free" && state.rigDraft?.profile==="rigid";
  $("rig-joint-fields").hidden=!articulated && !rigidRoot;
  $("rig-axis-fields").hidden=type!=="hinge";
  $("rig-limit-min").readOnly=type==="ball";
  $("rig-limit-min").closest(".field").hidden=type!=="hinge";
  $("rig-limit-max").closest(".field").hidden=!articulated;
  $("rig-damping").closest(".field").hidden=!articulated;
}
function updateRigDraftMessage(error="") {
  $("rig-draft-message").textContent=error || (state.rigDirty ? "Unapplied changes. Apply physics or discard before training." : "No unapplied changes.");
  $("rig-draft-message").classList.toggle("error",Boolean(error));
}
function editPhysicalRig(event) {
  const body=selectedRigBody();if(!body)return;
  invalidateLocomotion();
  const input=event.target,id=input.id,value=Number(input.value);
  // Display rounded values, but preserve every untouched property at full precision.
  if(rigControlFields.includes(id)){
    state.rigDraft.control={mode:$("rig-control-mode").value,excursion:Number($("rig-target-excursion").value),
      stiffness:Number($("rig-servo-stiffness").value),damping_ratio:Number($("rig-servo-damping").value)};
    $("rig-servo-fields").hidden=state.rigDraft.control.mode!=="joint_target";
  }else if(id==="rig-parent")body.parent=body.id===state.rigDraft.root ? null : input.value;
  else if(id==="rig-mass")body.mass=value;
  else if(id==="rig-joint"){
    if(input.value==="ball")body.joint.range[0]=0;
    else if(input.value==="hinge" && body.joint.type!=="hinge")body.joint.range[0]=-85;
    body.joint.type=input.value;$("rig-limit-min").value=rigNumber(body.joint.range[0]);
    rigJointVisibility();
  }else if(id==="rig-limit-min")body.joint.range[0]=value;
  else if(id==="rig-limit-max")body.joint.range[1]=value;
  else if(id==="rig-torque")body.joint.torque=value;
  else if(id==="rig-damping")body.joint.damping=value;
  else if(id.startsWith("rig-axis-"))body.joint.axis[["rig-axis-x","rig-axis-y","rig-axis-z"].indexOf(id)]=value;
  else if(input.dataset.rigCollider!==undefined){
    body.colliders[Number(input.dataset.rigCollider)].size[Number(input.dataset.rigDimension)]=value;
  }
  state.rigDirty=true;updateRigDraftMessage();controls();message("Unapplied physics edits in Body → Advanced physics.");
}
async function applyPhysicalRig() {
  if(!state.rigDraft || !state.rigDirty || state.busy || isTraining())return;
  const visibleFields=[...$("physical-rig-editor").querySelectorAll("input[type=number]")].filter(input=>!input.closest("[hidden]"));
  if(!fieldsValid(visibleFields,true))return;
  const generation=state.generation;
  state.busy=true;clearTimeout(previewTimer);state.previewVersion++;previewController?.abort();state.previewPending=false;
  controls();message("Validating physical rig…");
  try {
    if(!window.RLsimScene?.rebuildPhysicalRig)throw new Error("Physics editing is unavailable. Reload the updated app.");
    const body=window.RLsimScene.rebuildPhysicalRig(state.body,state.rigDraft);
    // Commit only after MuJoCo has compiled the proposed body successfully.
    const result=await api("/api/preview",{body,environment:readEnvironment()});
    if(generation!==state.generation)return;
    const selected=$("rig-body").value;
    setBody(result.body,result.model_info);
    if(state.rigDraft.bodies.some(part=>part.id===selected))$("rig-body").value=selected;
    showRigBody();$("physical-rig-editor").open=true;
    clearSelectedModel();clearReplay();mode("preview");
    showBody(state.body,result.model_info);setFrame(result.frame);view.reset(state.body);
    message("Physics applied. Train a new controller for this body.");
  } catch(error){
    if(generation===state.generation){message(error.message,true);updateRigDraftMessage(error.message);}
  } finally{if(generation===state.generation){state.busy=false;controls();}}
  if(generation===state.generation&&!state.rigDirty)await runRigChecks();
}
function invalidateLocomotion() {
  const hadAdvice=Boolean(state.locomotionAdvice || state.locomotionLoading);
  state.locomotionRevision++;state.locomotionAdvice=null;state.locomotionUndo=null;state.locomotionLoading=false;
  $("locomotion-review").hidden=true;$("undo-locomotion").hidden=true;
  if(hadAdvice)$("locomotion-status").textContent="Body or setup changed. Review physics setup again.";
}
function locomotionContext() {
  return {body:state.body,signature:JSON.stringify({
    environment:readEnvironment(),mass:$("locomotion-mass").value,
    strength:$("locomotion-strength").value,lock:$("locomotion-lock-fingers").checked,
  })};
}
function sameLocomotionContext(context) {
  const current=locomotionContext();
  return context.body===current.body && context.signature===current.signature;
}
function renderLocomotion() {
  const advice=state.locomotionAdvice;if(!advice)return;
  const a=advice.analysis;
  $("locomotion-summary").textContent=advice.summary;
  const metrics=[
    ["Total mass (kg)",a.total_mass_before,a.total_mass_after],
    ["Motor controls",a.controls_before,a.controls_after],
    ["New finger locks",0,a.fingers_locked],
  ];
  $("locomotion-metrics").replaceChildren(...metrics.map(([label,before,after])=>{
    const row=document.createElement("tr"),heading=document.createElement("th"),current=document.createElement("td"),next=document.createElement("td");
    heading.scope="row";heading.textContent=label;current.textContent=format(before,2);next.textContent=format(after,2);
    row.append(heading,current,next);return row;
  }));
  $("locomotion-changes").replaceChildren(...advice.changes.map(change=>{
    const li=document.createElement("li");
    li.textContent=`${change.label}: ${change.before.type} → ${change.after.type}; mass ${format(change.before.mass,3)} → ${format(change.after.mass,3)} kg; torque ${format(change.before.torque,2)} → ${format(change.after.torque,2)} N·m; damping ${format(change.before.damping,3)} → ${format(change.after.damping,3)}.`;
    return li;
  }));
  $("locomotion-warnings").replaceChildren(...advice.warnings.map(warning=>{
    const li=document.createElement("li");li.textContent=warning;return li;
  }));
  $("locomotion-warnings").hidden=!advice.warnings.length;
  $("apply-locomotion").hidden=!advice.rig;$("undo-locomotion").hidden=!state.locomotionUndo;
  $("locomotion-review").hidden=false;
}
async function recommendLocomotion() {
  if(state.busy || isTraining() || state.previewPending || state.rigDirty || !state.body?.physical_rig)return;
  if(!fieldsValid(["locomotion-mass","locomotion-strength","gravity","friction"],true))return;
  invalidateLocomotion();
  const generation=state.generation,revision=state.locomotionRevision,context=locomotionContext();
  state.locomotionLoading=true;controls();
  try {
    const {recommendLocomotionSetup}=await import("./locomotion-setup.mjs");
    if(generation!==state.generation || revision!==state.locomotionRevision || !sameLocomotionContext(context))return;
    const advice=recommendLocomotionSetup(state.body,{
      totalMass:Number($("locomotion-mass").value),strengthScale:Number($("locomotion-strength").value),
      lockFingers:$("locomotion-lock-fingers").checked,gravity:readEnvironment().gravity,
    });
    state.locomotionAdvice={...advice,context,applied:false};
    renderLocomotion();
    $("locomotion-status").textContent=advice.rig ? "Review the changes. These settings have not been optimized through training." : "This body needs manual physics setup.";
  }catch(error){message(`Locomotion setup: ${error.message}`,true);}
  finally{if(generation===state.generation && revision===state.locomotionRevision){state.locomotionLoading=false;controls();}}
}
async function applyLocomotion(undo=false) {
  const advice=state.locomotionAdvice;
  if(!advice || state.busy || isTraining() || state.previewPending || state.rigDirty)return;
  if(undo ? !state.locomotionUndo : !advice.rig || advice.applied)return;
  if(!sameLocomotionContext(advice.context)){invalidateLocomotion();controls();return;}
  const previous=state.body,restore=state.locomotionUndo,generation=state.generation;
  // Save reviewed values before setBody refreshes the editor.
  const fields=locomotionFields.map(id=>[$(id),$(id).type==="checkbox"?$(id).checked:$(id).value]);
  state.busy=true;clearTimeout(previewTimer);state.previewVersion++;previewController?.abort();state.previewPending=false;
  controls();message(undo ? "Restoring physics…" : "Validating locomotion setup…");
  try {
    const body=undo ? restore : window.RLsimScene.rebuildPhysicalRig(previous,advice.rig);
    const result=await api("/api/preview",{body,environment:readEnvironment(),task:readTask()});
    if(generation!==state.generation)return;
    setBody(result.body,result.model_info);
    for(const [input,value] of fields){if(input.type==="checkbox")input.checked=value;else input.value=value;}
    state.locomotionAdvice={...advice,context:locomotionContext(),applied:!undo};
    state.locomotionUndo=undo ? null : previous;
    clearSelectedModel();clearReplay();mode("preview");showBody(state.body,result.model_info);setFrame(result.frame);view.reset(state.body);
    $("locomotion-setup").open=true;renderLocomotion();
    $("locomotion-status").textContent=undo ? "Previous physics restored." : "Setup applied. Use Train → Recommend settings, then train a new controller.";
    message(undo ? "Previous physics restored." : "Locomotion setup applied. Train a new controller.");
  }catch(error){if(generation===state.generation)message(error.message,true);}
  finally{if(generation===state.generation){state.busy=false;controls();}}
}
function updateModelInfo(info) {
  state.modelInfo=info || null;
  $("body-count").textContent=info?.bodies?.length ?? "—";
  $("joint-count").textContent=info?.joints?.length ?? "—";
  $("actuator-count").textContent=info?.actuators?.length ?? info?.action_size ?? "—";
  $("observation-count").textContent=info?.observation_size ?? "—";
  const metadata=state.body?.metadata || {};
  $("body-source").textContent=importedBody(state.body) ? String(metadata.source || metadata.source_file || state.body.visual?.entry || metadata.format || metadata.source_format || "Imported MJCF body") : "";
  updateTrackedBodies(info);
}
function showBody(body,info=state.modelInfo) {
  const changed=state.shownBody!==body;
  state.shownBody=body;
  const count=Number.isFinite(body?.legs) ? body.legs*2 : info?.joints?.length ?? null;
  $("specimen-name").textContent=body ? `${body.name}${count!==null ? ` · ${count} joints` : " · Imported body"}` : "";
  $("physics-label").textContent=importedBody(body) ? "3D physics" : "Planar physics";
  if(changed && typeof view.setBody==="function"){
    const version=++state.renderVersion;
    Promise.resolve().then(()=>view.setBody(body)).catch(error=>{
      if(version===state.renderVersion)message(`Visual: ${error.message}`,true);
    });
  }
}
function mode(value) {
  state.mode=value;
  view.setMode?.(value);
  $("viewer-mode").textContent = {preview:"Preview",training:"Training samples",inference:"Model inference",
    baseline:state.modelInfo?.action_schema==="rlsim-joint-target-v1" ? "Rest targets" : importedBody(state.body) ? "Neutral controls" : "Zero torque"}[value];
  $("follow-camera-control").hidden=value==="training";
  $("training-camera-hint").hidden=value!=="training";
}
function clearReplay() {
  state.replayRequest++;state.replayLoading=false;state.replayBuffering=false;
  state.frames=[];state.replayTime=0;state.playing=false;
  state.pendingFrame=null;
  $("timeline").value=0;$("timeline").max=0;
  $("evaluation-summary").hidden=true;
  $("replay-progress").hidden=true;
}
function replayProgress(text) {
  const element=$("replay-progress");
  if(state.replayControl?.stopping)element.textContent="Stopping generation…";
  else if(text!==undefined)element.textContent=text;
  else if(state.frames.length)element.textContent=`${state.replayBuffering ? "Buffering" : "Generating replay"} · ${format(state.frames.at(-1).time,1)} s ready`;
  element.hidden=!state.replayLoading;
}
function setFrame(frame) {
  if (!frame) return;
  if(view.frame===frame)return;
  view.setFrame(frame);
  $("arena-empty").hidden=true;
  $("sim-time").textContent=`${format(frame.time || 0,2)} s`;
}
function controls() {
  if(typeof exploreControls==='function')exploreControls();
  const locked=state.busy || state.initializing || isTraining() || !state.body || rigWorkflow.active;
  $("best-checkpoint").disabled=locked;
  $("curriculum-mode").disabled=locked || !$("curriculum").checked;
  $("baseline-button").textContent=["rlsim-joint-target-v1","rlsim-joint-targets-v1"].includes(state.modelInfo?.action_schema) ? "Simulate rest targets" : "Simulate without control";
  const best=selectedModel()?.best;
  $("checkpoint-choice").options[0].disabled=!best;
  if(!best)$("checkpoint-choice").value="latest";
  $("checkpoint-choice").disabled=locked || !state.selectedRun;
  $("checkpoint-status").textContent=best
    ? `Best at ${format(best.steps)} steps · validation return ${format(best.validation?.mean_return,1)}. Download includes both checkpoints.`
    : "No evaluated checkpoint saved; using latest weights.";
  const adviceLocked=locked || !state.connected || state.previewPending || state.rigDirty || state.adviceLoading;
  $("recommend-training").disabled=adviceLocked;
  $("recommend-training").textContent=state.adviceLoading ? "Checking body…" : "Recommend settings";
  $("apply-advice").disabled=adviceLocked || !state.advice || state.advice.applied || !Object.keys(state.advice.settings).length;
  $("undo-advice").disabled=adviceLocked || !state.adviceUndo;
  const locomotionLocked=locked || !state.connected || state.previewPending || state.rigDirty || state.locomotionLoading;
  for(const id of locomotionFields)$(id).disabled=locked || state.locomotionLoading;
  $("recommend-locomotion").disabled=locomotionLocked || !state.body?.physical_rig;
  $("recommend-locomotion").textContent=state.locomotionLoading ? "Checking joints…" : "Review physics setup";
  $("apply-locomotion").disabled=locomotionLocked || !state.locomotionAdvice?.rig || state.locomotionAdvice.applied;
  $("undo-locomotion").disabled=locomotionLocked || !state.locomotionUndo;
  $("execution-engine").disabled=state.busy || isTraining() || state.initializing;
  $("import-model").hidden=state.engine!=="browser";
  $("export-interface").hidden=state.engine!=="browser";
  $("model-import-help").hidden=state.engine!=="browser";
  $("import-model").disabled=locked || !state.connected || state.previewPending;
  $("export-interface").disabled=locked || !state.connected || state.previewPending || state.rigDirty;
  for (const input of [...bodyInputs,...advancedInputs,$("body-name"),$("reset-advanced"),$("body-preset"),$("reset-design"),$("gravity"),$("friction"),$("training-budget"),$("target-speed"),$("seed"),$("n-envs"),$("curriculum"),$("randomize"),$("saved-run"),$("eval-speed"),$("eval-seed")]) input.disabled=locked;
  $("import-body").disabled=locked || !state.connected;
  $("export-body").disabled=locked || state.previewPending || state.rigDirty;
  for(const id of [...bodyImportFields,"choose-body-files","choose-body-folder","show-colliders"])$(id).disabled=locked;
  for(const id of [...rigEditorFields,...rigControlFields])$(id).disabled=locked || !state.rigDraft;
  for(const input of $("rig-collider-fields").querySelectorAll("input"))input.disabled=locked;
  const rigPart=selectedRigBody();
  if(rigPart?.id===state.rigDraft?.root){$("rig-parent").disabled=true;$("rig-joint").disabled=true;}
  $("import-profile").disabled=locked || $("import-rig").value==="rigid";
  $("apply-physical-rig").disabled=locked || !state.connected || !state.rigDirty;
  $("discard-physical-rig").disabled=locked || !state.rigDirty;
  for(const id of [...taskFields,"reset-task"])$(id).disabled=locked;
  $("apply-body-import").disabled=locked || !state.importBundle || !$("import-entry").value;
  $("train-button").hidden=isTraining();
  $("stop-button").hidden=!isTraining();
  $("train-button").disabled=locked || !state.connected || state.previewPending || state.rigDirty;
  $("stop-button").disabled=state.busy || state.status.status==="stopping";
  $("stop-button").textContent=state.status.status==="stopping" ? "Saving…" : "Stop and save";
  $("baseline-button").disabled=locked || !state.connected || state.previewPending || state.rigDirty;
  $("replay-button").disabled=locked || !state.connected || state.previewPending || !state.selectedRun;
  $("inference-controls").hidden=!state.replayLoading;
  $("stop-inference").disabled=Boolean(state.replayControl?.stopping);
  $("stop-inference").textContent=state.replayControl?.stopping ? "Stopping…" : state.replayControl?.baseline ? "Stop simulation" : "Stop inference";
  $("load-run").disabled=locked || !state.selectedRun;
  $("play-pause").disabled=Boolean(state.replayControl?.stopping) || (!state.frames.length && !(state.mode==="training" && isTraining()));
  $("play-pause").textContent=state.playing ? "Pause" : "Play";
  $("timeline").disabled=!state.frames.length;
  $("playback-speed").disabled=!state.frames.length;
  for (const [id,path] of [["checkpoint-link","checkpoint"],["body-link","body"]]) {
    const enabled=Boolean(state.selectedRun) && !state.busy && !(isTraining() && state.selectedRun===state.status.run_id);
    $(id).setAttribute("aria-disabled",String(!enabled));
    $(id).classList.toggle("disabled",!enabled);
    if (enabled) $(id).href=state.engine==="browser" ? "#" : `./api/${path}?run_id=${encodeURIComponent(state.selectedRun)}`;
    else $(id).removeAttribute("href");
  }
  updateRigWorkflowControls();
}
async function preview() {
  clearTimeout(previewTimer);
  if (!state.body || isTraining() || state.busy) return;
  if (!fieldsValid([...bodyValidationInputs(),"gravity","friction",...(state.tab==="run" && selectedModel() ? [] : taskValidationInputs())])) {
    state.previewPending=false;controls();return;
  }
  const version=++state.previewVersion;
  previewController?.abort();previewController=new AbortController();
  state.previewPending=true;clearReplay();mode("preview");controls();
  const body=previewBodyConfig(),environment=readEnvironment();
  try {
    const result=await api("/api/preview",{body,environment,task:state.tab==="run" && selectedModel() ? selectedModel().config.task ?? null : readTask()},previewController.signal);
    if (version!==state.previewVersion || isTraining()) return;
    state.environment=environment;
    if(body===state.body)updateModelInfo(result.model_info);
    setFrame(result.frame);showBody(body,result.model_info);
    message("Ready");
  } catch(error) {
    if(error.name!=="AbortError" && version===state.previewVersion) message(error.message,true);
  } finally {
    if (version===state.previewVersion) {state.previewPending=false;controls();}
  }
}
function schedulePreview() {
  invalidateAdvice();
  invalidateLocomotion();
  clearTimeout(previewTimer);
  state.previewVersion++;previewController?.abort();
  clearReplay();state.previewPending=true;controls();
  previewTimer=setTimeout(preview,180);
}
function selectTab(tab, refresh=true) {
  state.tab=tab;
  document.querySelectorAll("[data-tab]").forEach(button=>{
    const selected=button.dataset.tab===tab;
    button.setAttribute("aria-selected",String(selected));button.tabIndex=selected ? 0 : -1;
  });
  for(const name of ["body","train","run","explore"]) $(`${name}-panel`).hidden=name!==tab;
  if(tab==="train") drawChart();
  if(refresh && !isTraining()&&tab!=='explore') preview();
}
function invalidateAdvice() {
  const hadAdvice=Boolean(state.advice || state.adviceLoading);
  state.adviceRevision++;state.advice=null;state.adviceUndo=null;state.adviceLoading=false;
  $("training-advice").hidden=true;$("undo-advice").hidden=true;
  if(hadAdvice)$("advice-status").textContent="Body or settings changed. Recommend again for the current setup.";
}
function adviceSnapshot() {
  return Object.fromEntries(adviceFields.map(([id])=>[id,$(id).type==="checkbox" ? $(id).checked : $(id).value]));
}
function adviceContext() {
  return {body:state.body,signature:JSON.stringify({
    engine:state.engine,device:state.hardware.training_device,
    environment:readEnvironment(),task:readTask(),
    dimensions:importedBody(state.body) ? null : Object.fromEntries(bodyInputs.map(input=>[input.dataset.field,input.value])),
    settings:adviceSnapshot(),
  })};
}
function sameAdviceContext(context) {
  const current=adviceContext();
  return context.body===current.body && context.signature===current.signature;
}
function renderAdvice() {
  const advice=state.advice;if(!advice)return;
  const labels=Object.fromEntries(adviceFields.map(([id,key])=>[key,$(id).labels?.[0]?.textContent.trim() || key]));
  const display=value=>typeof value==="boolean" ? (value ? "On" : "Off") : format(Number(value),8);
  $("advice-summary").textContent=advice.summary;
  $("advice-settings").replaceChildren(...adviceFields.filter(([,key])=>key in advice.settings).map(([id,key])=>{
    const row=document.createElement("tr"),label=document.createElement("th"),current=document.createElement("td"),suggested=document.createElement("td");
    label.scope="row";label.textContent=labels[key];
    current.textContent=display($(id).type==="checkbox" ? $(id).checked : $(id).value);
    suggested.textContent=display(advice.settings[key]);row.append(label,current,suggested);return row;
  }));
  $("advice-reasons").replaceChildren(...advice.reasons.map(reason=>{
    const item=document.createElement("li");item.textContent=`${labels[reason.key] || reason.key}: ${reason.reason}`;return item;
  }));
  $("advice-warnings").replaceChildren(...advice.warnings.map(warning=>{
    const item=document.createElement("li");item.textContent=warning;return item;
  }));
  $("advice-warnings").hidden=!advice.warnings.length;
  $("apply-advice").hidden=!Object.keys(advice.settings).length;
  $("undo-advice").hidden=!state.adviceUndo;
  $("training-advice").hidden=false;
}
async function recommendSettings() {
  if(state.busy || isTraining() || state.previewPending || state.rigDirty || !state.body)return;
  if(!fieldsValid([...bodyValidationInputs(),"gravity","friction",...taskValidationInputs()],true))return;
  const generation=state.generation;
  invalidateAdvice();
  const revision=state.adviceRevision,context=adviceContext();
  state.adviceLoading=true;controls();
  try {
    const {recommendTraining}=await import("./training-advisor.mjs");
    if(generation!==state.generation || revision!==state.adviceRevision || !sameAdviceContext(context))return;
    const advice=recommendTraining({
      body:state.body,modelInfo:state.modelInfo || {},environment:readEnvironment(),task:readTask(),
      hardware:state.hardware,engine:state.engine,
      resources:state.engine==="browser" ? {hardwareConcurrency:navigator.hardwareConcurrency,deviceMemory:navigator.deviceMemory} : {},
    });
    state.advice={...advice,context,applied:false};state.adviceUndo=null;
    renderAdvice();$("training-advice").open=true;
    $("advice-status").textContent=Object.keys(advice.settings).length ? "Review the suggestions, then apply them for a new run." : "Check the body before choosing training settings.";
  }catch(error){message(`Recommendations: ${error.message}`,true);}
  finally {
    if(generation===state.generation && revision===state.adviceRevision){state.adviceLoading=false;controls();}
  }
}
function applyAdvice() {
  const advice=state.advice;
  if(!advice || state.busy || isTraining() || state.previewPending || state.rigDirty || advice.applied)return;
  if(!sameAdviceContext(advice.context)){invalidateAdvice();controls();return;}
  state.adviceUndo=adviceSnapshot();
  for(const [id,key] of adviceFields)if(key in advice.settings){
    if($(id).type==="checkbox")$(id).checked=advice.settings[key];
    else $(id).value=advice.settings[key];
  }
  // Keep all suggestions inside the same constraints as hand-edited settings.
  if(!fieldsValid(["training-budget","n-envs","target-speed",...advancedInputs])){
    for(const [id,value] of Object.entries(state.adviceUndo)){
      if($(id).type==="checkbox")$(id).checked=value;else $(id).value=value;
    }
    state.adviceUndo=null;message("Recommendation exceeds the current training limits.",true);return;
  }
  advice.applied=true;advice.context=adviceContext();renderAdvice();controls();
  $("advice-status").textContent="Applied. Settings remain editable; start training when ready.";
}
function undoAdvice() {
  if(!state.adviceUndo || !state.advice || state.busy || isTraining() || state.previewPending || state.rigDirty)return;
  if(!sameAdviceContext(state.advice.context)){invalidateAdvice();controls();return;}
  for(const [id,value] of Object.entries(state.adviceUndo)){
    if($(id).type==="checkbox")$(id).checked=value;else $(id).value=value;
  }
  state.adviceUndo=null;state.advice.applied=false;state.advice.context=adviceContext();
  renderAdvice();controls();$("advice-status").textContent="Previous settings restored.";
}
function drawChart() {drawRewardChart($("reward-chart"),state.status.history,state.status.total_steps);}
function updateStatus(status) {
  const wasActive=isTraining(),newRun=status.run_id!==lastRunId,changed=status.status!==lastStatus;
  state.status=status;
  if(status.hardware && (isTraining() || wasActive))showHardware(status.hardware);
  const m=status.metrics || {};
  $("run-status").textContent={idle:"Idle",starting:"Starting",training:"Training",stopping:"Saving",completed:"Completed",stopped:"Stopped",failed:"Failed"}[status.status] || status.status;
  $("step-count").textContent=`${format(status.steps || 0)} / ${format(status.total_steps || 0)}`;
  $("elapsed-value").textContent=`${format(status.elapsed || 0)} s`;
  $("reward-value").textContent=format(m.mean_reward,1);
  $("speed-value").textContent=`${format(m.speed,2)} m/s`;
  $("fps-value").textContent=`${format(m.fps)} steps/s`;
  $("validation-status").textContent=status.phase==="evaluating" ? "Evaluating checkpoint on validation episodes…"
    : status.best_validation ? `Best at ${format(status.best_checkpoint_steps)} steps · return ${format(status.best_validation.mean_return,1)} · survived ${format(status.best_validation.survival_rate*100)}% · speed error ${format(status.best_validation.speed_error,2)} m/s`
    : "";
  if(status.curriculum)$("validation-status").textContent+=` · Curriculum: ${status.curriculum.name} (${format(status.curriculum.speed,2)} m/s)`;
  const fraction=status.total_steps ? Math.min(1,status.steps/status.total_steps) : 0;
  if($("training-progress")) $("training-progress").value=fraction;
  if($("progress-percent")) $("progress-percent").textContent=`${Math.round(fraction*100)}%`;
  if($("chart-end")) $("chart-end").textContent=format(status.total_steps || 0);
  drawChart();
  if(isTraining()) {
    if(newRun || !wasActive) {
      clearTimeout(previewTimer);state.previewVersion++;previewController?.abort();state.previewPending=false;
      clearReplay();state.playing=true;mode("training");
      if(status.config) {
        setBody(status.config.body,status.model_info);setEnvironment(status.config.environment);
        setAdvanced(status.config);
        setTask(status.config.task);
        for(const [id,key] of [["training-budget","total_steps"],["target-speed","target_speed"],["seed","seed"],["n-envs","n_envs"]]) $(id).value=status.config[key] ?? $(id).value;
        $("curriculum").checked=status.config.curriculum;$("randomize").checked=status.config.randomize;
        showBody(status.config.body);
      }
    }
    if(status.model_info){
      updateModelInfo(status.model_info);
      showBody(state.shownBody,status.model_info);
    }
    if(status.frame) {
      if(state.playing) setFrame(status.frame); else state.pendingFrame=status.frame;
    }
  }
  if(changed || newRun) {
    if(status.status==="failed") message(status.error || "Training failed.",true);
    else if(isTraining()) message(status.status==="stopping" ? "Saving model…" : `Training on ${String(state.hardware.training_device || "cpu").toUpperCase()}`);
    else if(wasActive) {
      state.playing=false;message(status.status==="completed" ? "Training complete" : "Model saved");
    }
    refreshRuns(status.checkpoint_available ? status.run_id : null);
  }
  lastRunId=status.run_id;lastStatus=status.status;controls();
}
async function refreshRuns(preferred) {
  try {
    const result=await api("/api/runs");state.runs=result.runs || [];
    const saved=state.runs.filter(r=>r.checkpoint_available && !activeStatuses.has(r.status));
    const current=preferred || state.selectedRun || (state.modelSelectionCleared ? "" : saved[0]?.id) || "";
    if(preferred)state.modelSelectionCleared=false;
    const select=$("saved-run");
    select.replaceChildren(new Option(saved.length ? "Select model" : "No saved models",""));
    for(const r of saved) select.add(new Option(`${r.config.body.name} · ${format(r.steps)} steps · ${new Date(r.created_at).toLocaleString([], {month:"short",day:"numeric",hour:"2-digit",minute:"2-digit"})}`,r.id));
    selectSavedModel(saved.some(r=>r.id===current) ? current : "");select.value=state.selectedRun;controls();
    if(typeof comparisonModels==='function')comparisonModels();
  } catch(error) {if(error.name!=="AbortError")message(`Models: ${error.message}`,true);}
}
async function train() {
  if(rigWorkflow.active||rigWorkflow.checking)return;
  if(state.body?.metadata?.auto_rig&&(rigWorkflow.body!==state.body||rigWorkflow.report?.status==='needs-fixes')){
    message('Run Body → Check physics and resolve any errors before training this prepared body.',true);return;
  }
  if(!fieldsValid([...bodyValidationInputs(),"gravity","friction","training-budget","target-speed","seed","n-envs",...advancedInputs,...taskValidationInputs()],true)) return;
  const generation=state.generation;
  state.statusRevision++;
  state.busy=true;clearReplay();controls();message("Starting training…");
  try {
    const result=await api("/api/train",{body:state.body,environment:readEnvironment(),total_steps:Number($("training-budget").value),target_speed:Number($("target-speed").value),seed:Number($("seed").value),n_envs:Number($("n-envs").value),randomize:$("randomize").checked,curriculum:$("curriculum").checked,device:"auto",task:readTask(),...readAdvanced()});
    updateStatus(result);
  } catch(error) {if(error.name!=="AbortError")message(error.message,true);}
  finally {if(generation===state.generation){state.busy=false;controls();}}
}
async function stop() {
  const generation=state.generation;
  state.statusRevision++;
  state.busy=true;controls();
  try {updateStatus(await api("/api/stop",{}));} catch(error) {if(error.name!=="AbortError")message(error.message,true);}
  finally {if(generation===state.generation){state.busy=false;controls();}}
}
function useModelSettings() {
  const config=selectedModel()?.config;if(!config)return;
  setBody(config.body,selectedModel()?.model_info);setEnvironment(config.environment);
  setAdvanced(config);
  setTask(config.task);
  $("training-budget").value=config.total_steps ?? state.defaults.total_steps ?? $("training-budget").value;
  $("eval-speed").value=config.target_speed;
  $("target-speed").value=config.target_speed;
  $("seed").value=config.seed;
  $("n-envs").value=config.n_envs ?? state.defaults.n_envs ?? 4;
  $("curriculum").checked=config.curriculum;$("randomize").checked=config.randomize;
  preview();
}
async function stopInference() {
  const control=state.replayControl;
  if(!state.replayLoading || !control || control.stopping)return;
  control.stopping=true;state.playing=false;controls();replayProgress();
  message(control.baseline ? "Stopping simulation…" : "Stopping inference…");
  try{await api("/api/stop-replay",{replay_id:control.id});}
  catch(error){
    if(state.replayControl===control){
      control.stopping=false;controls();replayProgress();
      message(`Could not stop generation: ${error.message}`,true);
    }
  }
}
async function infer(baseline=false,replayOptions={}) {
  if(state.busy || isTraining())return;
  const ids=baseline ? [...bodyValidationInputs(),"gravity","friction",...taskValidationInputs()] : ["gravity","friction","eval-speed","eval-seed"];
  if(!fieldsValid(ids,true) || (!baseline && !state.selectedRun)) return;
  const generation=state.generation;
  clearTimeout(previewTimer);state.previewVersion++;previewController?.abort();state.previewPending=false;
  state.busy=true;clearReplay();state.replayLoading=true;
  const control={id:`replay-${Date.now()}-${Array.from(crypto.getRandomValues(new Uint8Array(12)),b=>b.toString(16).padStart(2,'0')).join('')}`,stopping:false,baseline};
  state.replayControl=control;
  const request=state.replayRequest;
  let started=false;
  controls();message(baseline ? "Simulating…" : "Loading model…");
  replayProgress(baseline ? "Preparing simulation…" : "Loading model…");
  const progress=event=>{
    if(generation!==state.generation || request!==state.replayRequest)return;
    if(event.phase==="loading")return;
    if(event.phase==="ready"){
      started=true;state.frames=[...event.frames];state.replayTime=0;state.playing=!control.stopping;
      lastAnimation=performance.now();
      mode(baseline ? "baseline" : "inference");
      showBody(event.body || state.body,event.model_info);setFrame(state.frames[0]);
      if(!control.stopping)message(baseline ? "Simulating…" : "Running inference…");
    }else if(event.phase==="simulating" && started){
      state.frames.push(...event.frames);
    }
    if(started){
      $("timeline").max=state.frames.at(-1).time;
      state.replayBuffering=state.playing && state.replayTime>=state.frames.at(-1).time;
      replayProgress();controls();
    }
  };
  try {
    const result=await api("/api/replay",{
      replay_id:control.id,
      ...(baseline ? {body:state.body} : {run_id:state.selectedRun}),
      ...(baseline ? {task:readTask()} : {}),
      environment:readEnvironment(),target_speed:baseline ? 1 : Number($("eval-speed").value),
      seed:baseline ? 10007 : Number($("eval-seed").value),deterministic:true,
      ...(!baseline?{checkpoint:$("checkpoint-choice").value}:{}),
      ...(!baseline&&replayOptions.proportions?{proportions:replayOptions.proportions}:{}),
    },undefined,{onProgress:progress});
    if(request!==state.replayRequest)return;
    if(result.cancelled || control.stopping){
      if(result.frames?.length){
        state.frames=result.frames;
        if(!started){state.replayTime=0;mode(baseline ? "baseline" : "inference");showBody(result.body || state.body,result.model_info);setFrame(result.frames[0]);}
        $("timeline").max=state.frames.at(-1).time;$("timeline").value=state.replayTime;
        $("evaluation-summary").textContent=`Stopped · ${format(state.frames.at(-1).time,1)} s generated · Partial episode`;
        $("evaluation-summary").hidden=false;
      }
      state.playing=false;
      message(baseline ? "Simulation stopped" : "Inference stopped");
      return;
    }
    if(!result.frames?.length)throw new Error("No simulation frames returned.");
    state.frames=result.frames;state.replayLoading=false;state.replayBuffering=false;
    if(!started){
      state.replayTime=0;state.playing=true;lastAnimation=performance.now();
      mode(baseline ? "baseline" : "inference");showBody(result.body || state.body,result.model_info);setFrame(result.frames[0]);
    }
    $("replay-progress").hidden=true;
    $("timeline").max=result.frames.at(-1).time;$("timeline").value=state.replayTime;
    const s=result.summary;
    $("evaluation-summary").textContent=`${format(s.distance,2)} m · ${format(s.mean_speed,2)} m/s · ${format(s.seconds,1)} s${s.fell ? " · Fell" : ""}${result.checkpoint ? ` · ${result.checkpoint==="best"?"Best":"Latest"} at ${format(result.checkpoint_steps)} steps` : ""}`;
    if(s.gait)$("evaluation-summary").textContent+=` · Steps by leg: ${s.gait.steps_per_leg.join(" / ")} · Foot drift ${format(s.gait.foot_slip_m_s,2)} m/s`;
    if(s.effort)$("evaluation-summary").textContent+=` · Motor work ${format(s.effort.absolute_motor_work_j,0)} J${s.effort.mechanical_cost_of_transport!=null?` · Work / weight / distance ${format(s.effort.mechanical_cost_of_transport,2)}`:""}`;
    $("evaluation-summary").hidden=false;
    const device=String(result.inference_device || (state.engine==="server" ? "cpu" : state.hardware.training_device || "cpu"));
    const deviceLabel=device==="webgpu" ? "WebGPU" : device.toUpperCase();
    message(baseline ? (["rlsim-joint-target-v1","rlsim-joint-targets-v1"].includes(result.model_info?.action_schema) ? "Rest-target simulation"
      : importedBody(result.body) ? "Neutral-control simulation" : "Zero-torque simulation")
      : `Inference complete · ${deviceLabel} · Seed ${result.seed ?? s.seed}`,false,result.fallback_reason || "");
    return result;
  } catch(error) {
    if(generation===state.generation && request===state.replayRequest){
      state.playing=false;
      if(error.name!=="AbortError")message(error.message,true);
    }
  }
  finally {
    if(generation===state.generation && request===state.replayRequest){
      state.replayLoading=false;state.replayBuffering=false;$("replay-progress").hidden=true;
      state.replayControl=null;
      state.busy=false;controls();
    }
  }
}
function seek(time) {
  if(!state.frames.length)return;
  const frames=state.frames;
  state.replayTime=Math.max(0,Math.min(time,frames.at(-1).time));
  let low=0,high=frames.length-1;
  while(low<high){const mid=Math.ceil((low+high)/2);if(frames[mid].time<=state.replayTime)low=mid;else high=mid-1;}
  setFrame(frames[low]);$("timeline").value=state.replayTime;
}
function playPause() {
  if(!state.replayLoading && state.frames.length && state.replayTime>=state.frames.at(-1).time)seek(0);
  state.playing=!state.playing;lastAnimation=performance.now();
  if(state.playing && state.pendingFrame){setFrame(state.pendingFrame);state.pendingFrame=null;}
  controls();
}
async function exportModel(event,type) {
  if(event.currentTarget.getAttribute("aria-disabled")==="true"){event.preventDefault();return;}
  if(state.engine!=="browser")return;
  event.preventDefault();
  const generation=state.generation;
  state.busy=true;controls();
  try {await browserBackend.exportModel(state.selectedRun,type);}
  catch(error){if(generation===state.generation)message(error.message,true);}
  finally{if(generation===state.generation){state.busy=false;controls();}}
}
function exportBody() {
  if(!fieldsValid([...bodyValidationInputs(),"gravity","friction"],true))return;
  const data={format:"rlsim-body-v1",body:state.body,environment:readEnvironment()};
  const blob=new Blob([JSON.stringify(data,null,2)],{type:"application/json"});
  const url=URL.createObjectURL(blob),link=document.createElement("a");
  link.href=url;link.download="rlsim-body.json";link.click();
  setTimeout(()=>URL.revokeObjectURL(url),10000);
}
async function importBody() {
  const file=$("body-file").files?.[0];
  if(!file || state.busy || isTraining())return;
  let generation=state.generation;
  state.busy=true;
  clearTimeout(previewTimer);state.previewVersion++;previewController?.abort();state.previewPending=false;
  controls();message("Importing body…");
  try {
    if(file.size>100*1024*1024)throw new Error("Body JSON must be smaller than 100 MB.");
    const data=JSON.parse(await file.text());
    if(!data || typeof data!=="object" || Array.isArray(data))throw new Error("Body JSON must contain an object.");
    if(data.format!==undefined && data.format!=="rlsim-body-v1")throw new Error("Unsupported body format. Use a body JSON export or a raw body configuration.");
    let body=data.format==="rlsim-body-v1" ? data.body : data;
    const environment=data.format==="rlsim-body-v1" ? data.environment ?? readEnvironment() : readEnvironment();
    if(!body || typeof body!=="object" || Array.isArray(body))throw new Error("Body JSON is missing its body configuration.");
    if(importedBody(body) && state.engine!=="browser"){
      state.busy=false;
      await initialize("browser",true);
      generation=state.generation;
      if(!state.connected)throw new Error("This device could not initialize for the imported body.");
      state.busy=true;controls();
    }
    if(body.physical_rig){
      if(!window.RLsimScene?.rebuildPhysicalRig)throw new Error("Physics editing is unavailable. Reload the updated app.");
      body=window.RLsimScene.rebuildPhysicalRig(body,body.physical_rig);
    }
    const result=await api("/api/preview",{body,environment});
    if(generation!==state.generation)return;
    setBody(result.body,result.model_info);setEnvironment(result.environment);
    clearSelectedModel();
    clearReplay();mode("preview");showBody(result.body);setFrame(result.frame);
    selectTab("body",false);view.reset(result.body);message("Body imported");
  } catch(error){if(error.name!=="AbortError" && generation===state.generation)message(error.message,true);}
  finally{if(generation===state.generation){state.busy=false;$("body-file").value="";controls();}}
}
async function prepareBodyFiles(input) {
  const files=[...input.files];
  if(!files.length || state.busy || isTraining())return;
  let generation=state.generation;
  state.busy=true;controls();message("Reading model files…");
  try {
    if(state.engine!=="browser"){
      state.busy=false;
      await initialize("browser",true);
      generation=state.generation;
      if(!state.connected)throw new Error("Select This device and allow it to initialize before importing model files.");
      state.busy=true;controls();message("Model files use This device.");
    }
    const bundle=await browserBackend.prepareBodyFiles(files);
    if(generation!==state.generation)return;
    if(!bundle.entries?.length)throw new Error("No supported model file found. Include an MJCF, URDF, GLB, glTF, FBX, OBJ or STL file.");
    state.importBundle=bundle;
    $("import-entry").replaceChildren(...bundle.entries.map(entry=>new Option(`${entry.path} (${entry.format})`,entry.path)));
    bodyImportVisibility();
    $("body-import-options").hidden=false;$("body-import").open=true;
    message(`${bundle.files.length} file${bundle.files.length===1 ? "" : "s"} loaded on This device`);
  } catch(error){if(generation===state.generation)message(error.message,true);}
  finally{if(generation===state.generation){state.busy=false;input.value="";controls();}}
}
function bodyImportVisibility() {
  const extension=$("import-entry").value.split(".").at(-1).toLowerCase();
  const visual=["glb","gltf","fbx","obj","stl"].includes(extension);
  const physics=["xml","mjcf","urdf"].includes(extension);
  for(const id of ["import-scale","import-mass","import-rig","import-profile"])$(id).closest(".field").hidden=!visual;
  $("import-motor-strength").closest(".field").hidden=!(visual || physics);
  $("import-floating-base").closest(".checkbox-fields").hidden=!physics;
}
async function applyBodyImport() {
  if(!state.importBundle || state.busy || isTraining())return;
  if(!fieldsValid(["import-scale","import-mass","import-motor-strength"].filter(id=>!$(id).closest(".field").hidden),true))return;
  let generation=state.generation;
  state.busy=true;
  clearTimeout(previewTimer);state.previewVersion++;previewController?.abort();state.previewPending=false;
  controls();message("Building physics body…");
  try {
    if(state.engine!=="browser"){
      state.busy=false;await initialize("browser",true);generation=state.generation;
      if(!state.connected)throw new Error("This device could not initialize for model import.");
      state.busy=true;controls();
    }
    const options={
      entry:$("import-entry").value,scale:Number($("import-scale").value),
      mass:Number($("import-mass").value),motor_strength:Number($("import-motor-strength").value),
      rig:$("import-rig").value,profile:$("import-profile").value,floating_base:$("import-floating-base").checked,
      auto_actuators:$("import-auto-actuators").checked,add_floor:$("import-add-floor").checked,
      environment:readEnvironment(),
    };
    const result=await browserBackend.importBody(state.importBundle,options);
    if(generation!==state.generation)return;
    if(!result.body || !result.frame)throw new Error("The importer did not return a validated physics body.");
    setBody(result.body,result.model_info);setEnvironment(result.environment || readEnvironment());
    clearSelectedModel();
    clearReplay();mode("preview");showBody(result.body,result.model_info);setFrame(result.frame);view.reset(result.body);
    $("body-source").textContent=options.entry;
    const noteText=item=>typeof item==="string" ? item : item.message || JSON.stringify(item);
    const report=result.body.metadata?.rig_report;
    const reported=new Set(report ? [...(report.repairs || []),...(report.warnings || [])].map(noteText) : []);
    const warnings=(result.warnings || []).map(noteText).filter(note=>!reported.has(note));
    $("import-warnings").textContent=warnings.join(" ");
    $("import-warnings").hidden=!warnings.length;
    $("body-import").open=false;selectTab("body",false);
    message("Body imported on This device");
  } catch(error){if(generation===state.generation)message(error.message,true);}
  finally{if(generation===state.generation){state.busy=false;controls();}}
  if(generation===state.generation)await runRigChecks();
}
async function importModel() {
  const file=$("model-file").files?.[0];
  if(!file || state.engine!=="browser")return;
  const generation=state.generation;
  state.busy=true;controls();message("Importing model…");
  try {
    const result=await browserBackend.importModel(file,{
      body:state.body,environment:readEnvironment(),task:readTask(),target_speed:Number($("target-speed").value),
    });
    if(generation!==state.generation)return;
    await refreshRuns(result?.id || result?.run_id);
    if(generation===state.generation)message("Model imported");
  } catch(error){if(generation===state.generation)message(error.message,true);}
  finally{if(generation===state.generation){state.busy=false;$("model-file").value="";controls();}}
}
async function exportInterface() {
  if(!fieldsValid([...bodyValidationInputs(),"gravity","friction","target-speed",...taskValidationInputs()],true))return;
  const generation=state.generation;
  state.busy=true;controls();
  try {
    await browserBackend.exportInterface({
      body:state.body,environment:readEnvironment(),task:readTask(),target_speed:Number($("target-speed").value),
    });
  } catch(error){if(generation===state.generation)message(error.message,true);}
  finally{if(generation===state.generation){state.busy=false;controls();}}
}
async function poll() {
  clearTimeout(pollTimer);
  const generation=state.generation,revision=state.statusRevision;
  try {
    const status=await api("/api/status");
    if(revision!==state.statusRevision)return;
    const reconnect=!state.connected;state.connected=true;updateStatus(status);if(reconnect)message("Connected");
  }
  catch(error){if(error.name!=="AbortError"){state.connected=false;message(`Connection: ${error.message}`,true);controls();}}
  finally{if(generation===state.generation)pollTimer=setTimeout(poll,isTraining() ? 800 : 3000);}
}
function animate(now) {
  const dt=Math.min((now-lastAnimation)/1000,.1);lastAnimation=now;
  if(state.frames.length && state.playing){
    seek(state.replayTime+dt*Number($("playback-speed").value));
    const atEnd=state.replayTime>=state.frames.at(-1).time;
    if(atEnd && !state.replayLoading){state.playing=false;controls();}
    else if(state.replayLoading && atEnd!==state.replayBuffering){state.replayBuffering=atEnd;replayProgress();}
  }
  view.draw(now);requestAnimationFrame(animate);
}
for(const button of document.querySelectorAll("[data-tab]")) {
  button.addEventListener("click",()=>selectTab(button.dataset.tab));
  button.addEventListener("keydown",event=>{
    const tabs=[...document.querySelectorAll("[data-tab]")],index=tabs.indexOf(button);
    const next=event.key==="ArrowRight" ? (index+1)%tabs.length : event.key==="ArrowLeft" ? (index+tabs.length-1)%tabs.length : null;
    if(next!==null){event.preventDefault();selectTab(tabs[next].dataset.tab);tabs[next].focus();}
  });
}
for(const input of bodyInputs)input.addEventListener("input",()=>{
  if(fieldsValid(bodyInputs))for(const field of bodyInputs)state.body[field.dataset.field]=Number(field.value);
  schedulePreview();
});
$("body-name").addEventListener("input",()=>{
  if(fieldsValid(["body-name"]))state.body.name=$("body-name").value;
  schedulePreview();
});
for(const id of ["gravity","friction"])$(id).addEventListener("input",schedulePreview);
$("body-preset").addEventListener("change",()=>{$("tracked-body").value="";setBody(state.presets[$("body-preset").value]);preview().then(()=>view.reset(state.body));});
$("reset-design").addEventListener("click",()=>{setBody(state.presets[$("body-preset").value]);preview().then(()=>view.reset(state.body));});
$("import-body").addEventListener("click",()=>$("body-file").click());
$("body-file").addEventListener("change",importBody);
$("export-body").addEventListener("click",exportBody);
$("choose-body-files").addEventListener("click",()=>$("asset-files").click());
$("choose-body-folder").addEventListener("click",()=>$("asset-folder").click());
for(const id of ["asset-files","asset-folder"])$(id).addEventListener("change",()=>prepareBodyFiles($(id)));
$("import-entry").addEventListener("change",()=>{bodyImportVisibility();controls();});
$("apply-body-import").addEventListener("click",applyBodyImport);
$("import-rig").addEventListener("change",controls);
$("rig-body").addEventListener("change",showRigBody);
for(const id of rigEditorFields.filter(id=>id!=="rig-body"))$(id).addEventListener("input",editPhysicalRig);
for(const id of rigControlFields)$(id).addEventListener("input",editPhysicalRig);
$("rig-collider-fields").addEventListener("input",editPhysicalRig);
$("apply-physical-rig").addEventListener("click",applyPhysicalRig);
$("recommend-locomotion").addEventListener("click",recommendLocomotion);
$("apply-locomotion").addEventListener("click",()=>applyLocomotion());
$("undo-locomotion").addEventListener("click",()=>applyLocomotion(true));
for(const id of locomotionFields)$(id).addEventListener("input",()=>{invalidateLocomotion();controls();});
$("discard-physical-rig").addEventListener("click",()=>{
  invalidateLocomotion();
  const selected=$("rig-body").value;
  setPhysicalRig();$("rig-body").value=selected;showRigBody();$("physical-rig-editor").open=true;controls();message("Physics edits discarded.");
});
$("show-colliders").addEventListener("change",()=>{view.showCollision=$("show-colliders").checked;view.dirty=true;});
$("train-button").addEventListener("click",train);$("stop-button").addEventListener("click",stop);
$("replay-button").addEventListener("click",()=>infer());$("baseline-button").addEventListener("click",()=>infer(true));
$("stop-inference").addEventListener("click",stopInference);
$("load-run").addEventListener("click",useModelSettings);
$("reset-advanced").addEventListener("click",()=>setAdvanced());
$("recommend-training").addEventListener("click",recommendSettings);
$("apply-advice").addEventListener("click",applyAdvice);
$("undo-advice").addEventListener("click",undoAdvice);
for(const [id] of adviceFields)$(id).addEventListener("input",()=>{invalidateAdvice();controls();});
$("task-preset").addEventListener("change",()=>{if($("task-preset").value==="walk"){$("foot-drift-weight").value=1;$("foot-only-support").checked=true;}taskVisibility();schedulePreview();});
for(const id of taskFields.filter(id=>id!=="task-preset"))$(id).addEventListener("input",schedulePreview);
$("reset-task").addEventListener("click",()=>{setTask();schedulePreview();});
$("checkpoint-link").addEventListener("click",event=>exportModel(event,"checkpoint"));
$("body-link").addEventListener("click",event=>exportModel(event,"body"));
$("import-model").addEventListener("click",()=>$("model-file").click());
$("model-file").addEventListener("change",importModel);
$("export-interface").addEventListener("click",exportInterface);
$("execution-engine").addEventListener("change",()=>{
  if(state.busy || isTraining()){$("execution-engine").value=state.engine;return;}
  if($("execution-engine").value==="server" && importedBody(state.body)){
    $("execution-engine").value=state.engine;
    message("Imported bodies currently run on This device; choose a built-in body to use Server.",true);
    return;
  }
  initialize($("execution-engine").value);
});
$("saved-run").addEventListener("change",()=>{selectSavedModel($("saved-run").value);state.modelSelectionCleared=!state.selectedRun;clearReplay();controls();preview();});
$("checkpoint-choice").addEventListener("change",clearReplay);
$("play-pause").addEventListener("click",playPause);
$("timeline").addEventListener("input",()=>{state.playing=false;seek(Number($("timeline").value));controls();});
$("playback-speed").addEventListener("change",()=>{lastAnimation=performance.now();});
$("reset-camera").addEventListener("click",()=>{view.reset(state.shownBody);$("view-angle").value="perspective";});
$("view-angle").addEventListener("change",()=>view.setAngle($("view-angle").value));
$("follow-camera").addEventListener("change",()=>{view.camera.follow=$("follow-camera").checked;if(view.frame)view.setFrame(view.frame);});
new ResizeObserver(drawChart).observe($("reward-chart"));
requestAnimationFrame(animate);
async function initialize(engine=state.engine,preserveEditor=false) {
  const previous=preserveEditor && state.body ? {
    body:structuredClone(state.body),environment:readEnvironment(),advanced:readAdvanced(),
    modelInfo:state.modelInfo,task:readTask(),
    training:Object.fromEntries([["training-budget","total_steps"],["target-speed","target_speed"],["seed","seed"],["n-envs","n_envs"]].map(([id,key])=>[key,$(id).value])),
    curriculum:$("curriculum").checked,randomize:$("randomize").checked,
  } : null;
  const generation=++state.generation;
  state.engine=engine;state.initializing=true;state.connected=false;state.busy=false;
  clearTimeout(previewTimer);clearTimeout(pollTimer);
  state.previewVersion++;previewController?.abort();state.previewPending=false;
  state.body=null;state.presets={};state.hardware={};state.runs=[];state.selectedRun="";state.modelSelectionCleared=false;
  state.rigDraft=null;state.rigDirty=false;
  state.defaults=advancedDefaults(engine);
  setAdvanced();$("advanced-training").open=false;
  setTask();$("task-settings").open=false;
  state.status={status:"idle",history:[]};state.pendingFrame=null;
  lastRunId=undefined;lastStatus=undefined;
  clearReplay();mode("preview");showBody(null);
  view.frame=null;view.dirty=true;
  $("saved-run").replaceChildren(new Option("No saved models",""));
  $("arena-empty").hidden=false;$("arena-empty").textContent="Loading simulation…";
  $("hardware-label").textContent="Connecting…";$("hardware-label").title="";
  $("sim-time").textContent="0.00 s";
  $("execution-engine").value=engine;
  $("run-status").textContent="Idle";$("step-count").textContent="0 / 0";
  $("elapsed-value").textContent="0 s";$("reward-value").textContent="—";
  $("speed-value").textContent="—";$("fps-value").textContent="—";
  $("training-progress").value=0;$("progress-percent").textContent="0%";
  drawChart();controls();message("");
  const url=new URL(location.href);
  if(engine==="server")url.searchParams.set("engine","server");
  else url.searchParams.delete("engine");
  history.replaceState(null,"",url);
  try {
    const config=await api("/api/config");
    if(!config.presets?.quadruped || !config.hardware)throw new Error("Training backend unavailable at this address.");
    state.presets=config.presets;state.connected=true;
    setBody(previous?.body || config.presets.quadruped,previous?.modelInfo);
    setEnvironment(previous?.environment || config.environment_defaults);
    const defaults={...advancedDefaults(engine),...config.defaults};
    state.defaults=defaults;setAdvanced(previous?.advanced || defaults);
    setTask(previous?.task || defaults.task);
    for(const [id,key] of [["training-budget","total_steps"],["target-speed","target_speed"],["seed","seed"],["n-envs","n_envs"]])$(id).value=previous?.training[key] ?? defaults[key] ?? $(id).value;
    $("curriculum").checked=previous?.curriculum ?? defaults.curriculum ?? true;
    $("randomize").checked=previous?.randomize ?? defaults.randomize ?? true;
    $("eval-speed").value=defaults.target_speed ?? 1;
    $("eval-seed").value=10007;
    showHardware(config.hardware);
    await preview();
    if(generation!==state.generation)return;
    view.reset(state.body);
    await refreshRuns();
    if(generation!==state.generation)return;
    await poll();
  } catch(error) {
    if(error.name!=="AbortError" && generation===state.generation){
      state.connected=false;
      message(engine==="server" ? "Server unavailable. Select This device to run locally." : error.message,true);
      $("hardware-label").textContent=engine==="browser" ? "This device unavailable" : "Server unavailable";
      $("hardware-label").title=error.message;
      $("arena-empty").textContent="Simulation unavailable";
    }
  } finally {
    if(generation===state.generation){state.initializing=false;controls();}
  }
}
selectTab("body",false);controls();initialize();
