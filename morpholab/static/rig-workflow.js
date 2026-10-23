"use strict";
const rigWorkflow={file:null,available:false,active:false,job:null,result:null,checking:false,report:null,body:null};
const rigEl=id=>document.getElementById(id);
const rigStatus=text=>rigEl('auto-rig-status').textContent=text;
async function rigFetch(path,options={}){
  const response=await fetch(path,options);
  if(!response.ok){let detail;try{detail=(await response.json()).detail;}catch{}throw new Error(detail||'Local SkinTokens service is unavailable. Run the native RLsim server with SkinTokens configured; static hosting supports prepared GLB imports.');}
  return response;
}
function resetRigChecks(){
  rigWorkflow.report=null;rigWorkflow.body=null;
  rigEl('rig-check-panel').hidden=state.body?.kind!=='mjcf';
  rigEl('rig-check-summary').textContent=state.body?.metadata?.auto_rig?'Run physics checks before training this prepared body.':'Physics checks have not run for this body.';
  rigEl('rig-check-details').hidden=true;
}
function updateRigWorkflowControls(){
  if(rigWorkflow.body===state.body&&rigWorkflow.environment!==JSON.stringify(readEnvironment()))resetRigChecks();
  const locked=state.busy||isTraining()||state.initializing||rigWorkflow.active;
  rigEl('run-rig-checks').disabled=locked||state.rigDirty||state.previewPending||rigWorkflow.checking||state.engine!=='browser';
  rigEl('start-auto-rig').disabled=locked||!rigWorkflow.file||!rigWorkflow.available;
  rigEl('auto-rig-file').disabled=locked;
  rigEl('check-rig-service').disabled=locked;
  rigEl('use-auto-rig').disabled=locked||!rigWorkflow.result;
  rigEl('prepare-import-rig').disabled=locked;
  rigEl('load-rig-example').disabled=locked;rigEl('rig-example').disabled=locked;
  const selected=rigEl('import-entry').value;
  rigEl('prepare-import-rig').hidden=!selected.toLowerCase().endsWith('.glb');
  if(rigWorkflow.checking||rigWorkflow.active||(rigWorkflow.body===state.body&&rigWorkflow.report?.status==='needs-fixes'))rigEl('train-button').disabled=true;
  if(state.body?.metadata?.auto_rig&&rigWorkflow.body!==state.body)rigEl('train-button').disabled=true;
}
async function runRigChecks(){
  if(state.busy||isTraining()||state.rigDirty||state.previewPending||rigWorkflow.checking||state.engine!=='browser'||state.body?.kind!=='mjcf')return;
  const body=state.body,signature=JSON.stringify(readEnvironment()),generation=state.generation;
  rigWorkflow.checking=true;state.busy=true;controls();
  rigEl('rig-check-summary').textContent='Checking joints, collision shapes and motor response…';
  try{
    const report=await browserBackend.call('check_rig',{body,environment:readEnvironment()});
    if(body!==state.body||generation!==state.generation||signature!==JSON.stringify(readEnvironment()))return;
    rigWorkflow.report=report;rigWorkflow.body=body;rigWorkflow.environment=signature;
    const labels={'passed':'Physics checks passed','review':'Physics checks complete — review the notes','needs-fixes':'Physics needs fixes before training'};
    rigEl('rig-check-summary').textContent=labels[report.status];
    rigEl('rig-check-list').replaceChildren(...report.checks.map(check=>{const li=document.createElement('li');li.textContent=`${{pass:'Pass',warning:'Review',error:'Fix',info:'Note'}[check.level]}: ${check.message}`;return li;}));
    rigEl('rig-check-details').hidden=false;rigEl('rig-check-details').open=report.status!=='passed';
  }catch(error){rigEl('rig-check-summary').textContent=`Checks could not finish: ${error.message}`;}
  finally{rigWorkflow.checking=false;if(generation===state.generation)state.busy=false;controls();}
}
async function checkRigService(){
  rigStatus('Checking local setup…');
  try{
    const status=await(await rigFetch('/api/rig-prepare/status')).json();
    rigWorkflow.available=Boolean(status.configured);
    rigStatus(status.configured?'SkinTokens files are available. Generate rig will check CUDA before starting.':`Setup needed: ${(status.issues||[]).join(' ')}`);
  }catch(error){rigWorkflow.available=false;rigStatus(error.message);}
  controls();
}
async function generateAutomaticRig(){
  if(!rigWorkflow.file||!rigWorkflow.available||rigWorkflow.active||state.busy||isTraining())return;
  const file=rigWorkflow.file;
  if(file.size>64*1024*1024){rigStatus('Choose a GLB smaller than 64 MB.');return;}
  rigWorkflow.active=true;rigWorkflow.result=null;rigWorkflow.job=null;
  rigEl('auto-rig-result').hidden=true;rigEl('cancel-auto-rig').hidden=false;rigEl('cancel-auto-rig').disabled=true;controls();
  rigStatus('Sending the model to your local SkinTokens helper…');
  try{
    let job=await(await rigFetch('/api/rig-prepare/jobs',{method:'POST',headers:{'Content-Type':'model/gltf-binary'},body:file})).json();
    rigWorkflow.job=job.id;rigEl('cancel-auto-rig').disabled=false;
    const labels={checking:'Checking CUDA and model files…',rigging:'SkinTokens is predicting the skeleton and skin weights. This can take several minutes…',verifying:'Checking shape alignment and retaining original geometry and materials…',stopping:'Stopping SkinTokens…'};
    while(!['completed','failed','cancelled'].includes(job.status)){
      rigStatus(labels[job.status]||'Preparing model…');
      await new Promise(resolve=>setTimeout(resolve,1200));
      job=await(await rigFetch(`/api/rig-prepare/jobs/${job.id}`)).json();
    }
    if(job.status==='cancelled'){rigStatus('Preparation cancelled. Your current body and source file are unchanged.');return;}
    if(job.status==='failed')throw new Error(job.error||'SkinTokens could not rig this model.');
    const blob=await(await rigFetch(`/api/rig-prepare/jobs/${job.id}/model`)).blob();
    rigWorkflow.result=new File([blob],file.name.replace(/\.glb$/i,'-rigged.glb'),{type:'model/gltf-binary'});
    const findings=job.fidelity?.skeleton_review?.findings||[];
    rigEl('auto-rig-fidelity').textContent=`Original geometry and materials retained. Predicted ${job.fidelity?.joints||'new'} joints. ${findings.length?findings.map(f=>f.message).join(' '):'Review body settings, then import to check the physics.'}`;
    rigEl('auto-rig-result').hidden=false;rigStatus('Rig prepared. Review it before training.');
  }catch(error){rigStatus(error.message);}
  finally{rigWorkflow.active=false;rigEl('cancel-auto-rig').hidden=true;controls();}
}
document.addEventListener('DOMContentLoaded',()=>{
  rigEl('run-rig-checks').addEventListener('click',runRigChecks);
  rigEl('check-rig-service').addEventListener('click',checkRigService);
  rigEl('auto-rig-file').addEventListener('change',()=>{rigWorkflow.file=rigEl('auto-rig-file').files[0]||null;rigWorkflow.result=null;rigEl('auto-rig-result').hidden=true;rigEl('auto-rig-selected').hidden=true;controls();});
  rigEl('start-auto-rig').addEventListener('click',generateAutomaticRig);
  rigEl('cancel-auto-rig').addEventListener('click',async()=>{
    if(!rigWorkflow.job)return;rigEl('cancel-auto-rig').disabled=true;
    try{await rigFetch(`/api/rig-prepare/jobs/${rigWorkflow.job}/cancel`,{method:'POST'});rigStatus('Stopping SkinTokens…');}catch(error){rigStatus(error.message);rigEl('cancel-auto-rig').disabled=false;}
  });
  rigEl('prepare-import-rig').addEventListener('click',async()=>{
    const source=state.importBundle?.files.find(f=>f.path===rigEl('import-entry').value);if(!source)return;
    rigWorkflow.file=new File([source.data],source.path.split('/').at(-1),{type:'model/gltf-binary'});
    rigEl('auto-rig-selected').textContent=`Selected: ${rigWorkflow.file.name}`;rigEl('auto-rig-selected').hidden=false;
    rigEl('auto-rig-file').value='';rigWorkflow.result=null;rigEl('auto-rig-result').hidden=true;
    rigEl('optional-rigging').open=true;rigEl('optional-rigging').scrollIntoView({block:'nearest'});
    await checkRigService();
  });
  rigEl('use-auto-rig').addEventListener('click',async()=>{
    if(!rigWorkflow.result)return;
    const file=rigWorkflow.result;await prepareBodyFiles({files:[file],value:''});
    rigEl('import-rig').value='skeleton';rigEl('import-profile').value='auto';
    rigEl('body-import').open=true;rigEl('body-import').scrollIntoView({block:'nearest'});
    message('Review scale, mass and motor torque, then import the prepared body.');
  });
  rigEl('download-auto-rig').addEventListener('click',()=>{
    if(!rigWorkflow.result)return;
    browserBackend.download(rigWorkflow.result,rigWorkflow.result.name,'model/gltf-binary');
  });
  rigEl('load-rig-example').addEventListener('click',async()=>{
    const name=rigEl('rig-example').value;if(!['humanoid','giraffe','hexapod'].includes(name)||state.busy||isTraining())return;
    state.busy=true;controls();
    try{
      const response=await fetch(`./demos/auto-rig-${name}.glb`);if(!response.ok)throw new Error('The prepared example is unavailable.');
      const file=new File([await response.blob()],`${name}-skintokens.glb`,{type:'model/gltf-binary'});
      state.busy=false;
      await prepareBodyFiles({files:[file],value:''});
      if(rigEl('import-entry').value!==file.name)return;
      rigEl('import-rig').value='skeleton';rigEl('import-profile').value='auto';rigEl('import-scale').value=1;rigEl('import-mass').value=name==='humanoid'?60:10;
      rigEl('body-import').open=true;rigEl('body-import').scrollIntoView({block:'nearest'});
      message('Review example settings, then import to run physics checks. No walking policy is included.');
    }catch(error){rigStatus(error.message);}finally{state.busy=false;controls();}
  });
});
