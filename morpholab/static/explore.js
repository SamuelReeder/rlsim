"use strict";
let galleryEntries=[],comparisonData=null,comparisonJob=null;
const element=(tag,text,className)=>{const node=document.createElement(tag);if(text!==undefined)node.textContent=text;if(className)node.className=className;return node;};
const comparisonLabel=record=>`${record.name||record.config.body.name} · ${record.id.slice(-8)}`;
function resultsTable(headers,rows){
  const wrapper=element('div',undefined,'results-scroll'),table=element('table',undefined,'results-table');
  wrapper.tabIndex=0;wrapper.setAttribute('role','region');wrapper.setAttribute('aria-label','Results table; scroll horizontally to see additional columns on narrow screens');
  const head=element('thead'),header=element('tr');headers.forEach(text=>header.append(element('th',text)));head.append(header);table.append(head);
  const body=element('tbody');for(const row of rows){const tr=element('tr');row.forEach(text=>tr.append(element('td',text)));body.append(tr);}table.append(body);wrapper.append(table);return wrapper;
}
async function loadExplore(){
  try{
    const response=await fetch('./demos/gallery.json');if(!response.ok)throw new Error('Demo files are unavailable. Rebuild the static site.');
    const manifest=await response.json();galleryEntries=manifest.entries;
    $('demo-gallery').replaceChildren(...galleryEntries.map(entry=>{
      const card=element('article',undefined,'demo-card');
      if(entry.thumbnail){const thumbnail=element('img');thumbnail.src=`./demos/${entry.thumbnail}`;thumbnail.alt=`${entry.title} in the physics simulator`;thumbnail.loading='lazy';card.append(thumbnail);}
      card.append(element('h3',entry.title),element('p',entry.description,'control-hint'));
      card.append(element('p',`Checkpoint at ${format(entry.steps)} steps · ${entry.provenance}`,'control-hint'));
      if(entry.browser_evaluation){const e=entry.browser_evaluation;card.append(element('p',`Browser check: ${e.survived}/${e.episodes.length} episodes without falling · ${format(e.mean_distance,1)} m mean distance${e.gait_passed==null?'':` · ${e.gait_passed}/${e.episodes.length} passed the gait check`}`,'control-hint'));}
      const button=element('button','Watch demo');button.type='button';button.dataset.demo=entry.id;button.addEventListener('click',()=>watchDemo(entry));card.append(button);return card;
    }));
    $('gallery-status').textContent='Examples are imported into your local model library. Choose This device to watch.';
    controls();
  }catch(error){$('gallery-status').textContent=error.message;}
  try{
    const response=await fetch('./demos/comparison.json');if(!response.ok)throw new Error('The published comparison is not available yet.');
    const report=await response.json(),{aggregateTrials}=await import('./experiment-results.mjs');
    const groups=aggregateTrials(report.trials);
    const rows=groups.map(g=>[g.body,g.condition,`${g.completed}/${g.trials.length}`,g.survival.mean==null?'—':`${format(g.survival.mean*100,0)}%`,`${format(g.speed_error.mean,2)} ± ${format(g.speed_error.sd,2)}`]);
    $('published-comparison').replaceChildren(element('p',`${format(report.protocol.steps)} steps/run · ${report.protocol.training_seeds.length} training seeds · ${report.protocol.evaluation_seeds.length} test episodes/run. Error is mean ± sample standard deviation across training seeds.`,'control-hint'),
      resultsTable(['Body','Curriculum','Finished','Survival','Speed error (m/s)'],rows),element('p',report.protocol.note,'control-hint'),
      ...(report.presentation_notes||[]).map(note=>element('p',note,'control-hint')));
  }catch(error){$('published-comparison').textContent=error.message;}
}
function exploreControls(){
  if(typeof generalizationControls==='function')generalizationControls();
  const locked=state.busy||isTraining()||!state.connected||state.previewPending;
  document.querySelectorAll('[data-demo]').forEach(button=>button.disabled=locked||state.engine!=='browser');
  $('resume-training').disabled=locked||!selectedModel()?.resumable;
  $('additional-steps').disabled=locked;
  $('continuation-status').textContent=selectedModel()?.resumable
    ? `Continue from ${format(selectedModel().steps)} steps using the latest training state. Checkpoint playback selection does not change the resume source.`
    : 'This model has no complete training state. Older, gallery and ONNX controllers support inference only.';
  $('run-comparison').disabled=locked;
  $('stop-comparison').hidden=!comparisonJob;
  $('stop-comparison').disabled=Boolean(comparisonJob?.stopped);
  document.querySelectorAll('[data-compare-model]').forEach(input=>input.disabled=locked);
}
function comparisonModels(){
  const selected=new Set([...document.querySelectorAll('[data-compare-model]:checked')].map(input=>input.value));
  $('comparison-models').replaceChildren(...state.runs.filter(r=>r.checkpoint_available&&!activeStatuses.has(r.status)).map(record=>{
    const label=element('label',undefined,'comparison-choice'),input=element('input');input.type='checkbox';input.value=record.id;input.dataset.compareModel='';input.checked=selected.has(record.id);
    label.append(input,document.createTextNode(`${comparisonLabel(record)} · ${format(record.steps)} steps · seed ${record.config.seed}`));return label;
  }));
}
async function watchDemo(entry){
  if(state.busy||isTraining()||state.engine!=='browser')return;
  state.busy=true;controls();message('Loading demo…');
  try{
    const response=await fetch(`./demos/${entry.file}`);if(!response.ok)throw new Error('Could not load this demo.');
    const record=await browserBackend.call('validate',await response.json());
    record.id=`gallery-${entry.id}`;record.name=entry.title;record.gallery_id=entry.id;
    await browserBackend.put(record);await refreshRuns(record.id);
    $('eval-speed').value=record.config.target_speed;$('eval-seed').value=entry.seed??500000007;
    selectTab('run',false);
  }catch(error){message(error.message,true);return;}
  finally{state.busy=false;controls();}
  await infer();
}
async function resumeTraining(){
  if(state.busy||isTraining()||!selectedModel()?.resumable||!fieldsValid(['additional-steps'],true))return;
  state.busy=true;state.statusRevision++;clearReplay();controls();message('Restoring training state…');
  try{
    const result=await api('/api/resume',{run_id:state.selectedRun,additional_steps:Number($('additional-steps').value)});
    selectTab('train',false);updateStatus(result);
  }catch(error){message(error.message,true);}
  finally{state.busy=false;controls();}
}
async function stopComparison(){
  const job=comparisonJob;if(!job||job.stopped)return;
  job.stopped=true;controls();$('comparison-status').textContent='Stopping comparison…';
  try{if(job.replayId)await api('/api/stop-replay',{replay_id:job.replayId});}
  catch(error){job.stopped=false;controls();$('comparison-status').textContent=`Could not stop: ${error.message}`;}
}
async function compareModels(){
  const ids=[...document.querySelectorAll('[data-compare-model]:checked')].map(input=>input.value);
  if(state.busy||isTraining())return;
  if(ids.length<2||ids.length>6){$('comparison-status').textContent='Select between 2 and 6 models.';return;}
  const {comparisonWarnings,COMPARISON_SEEDS}=await import('./experiment-results.mjs');
  if(state.busy||isTraining())return;
  const records=ids.map(id=>state.runs.find(r=>r.id===id));
  if(records.some(r=>!r)){ $('comparison-status').textContent='The model library changed. Select the models again.';return;}
  const warnings=comparisonWarnings(records);
  const seeds=[...COMPARISON_SEEDS];
  // Keep final test seeds distinct from checkpoint-selection seeds.
  const used=new Set(records.flatMap(r=>Array.from({length:r.config.evaluation_episodes??3},(_,i)=>(r.config.seed+1000003+i*997)%2147483647)));
  for(let i=0;i<seeds.length;i++)while(used.has(seeds[i])||seeds.slice(0,i).includes(seeds[i]))seeds[i]=(seeds[i]+101)%2147483647;
  const job={stopped:false,replayId:null};comparisonJob=job;state.busy=true;clearReplay();controls();
  comparisonData={format:'rlsim-model-comparison-v1',created_at:new Date().toISOString(),seeds,warnings,checkpoint:'best',runs:[]};
  $('comparison-results').replaceChildren();$('export-comparison').hidden=true;
  try{
    for(const record of records){
      if(job.stopped)break;
      const run={id:record.id,name:record.name,config:record.config,training_steps:record.steps,training_seed:record.config.seed,episodes:[],status:'evaluating'};
      comparisonData.runs.push(run);
      try{
        for(const seed of seeds){
          if(job.stopped)break;
          job.replayId=`compare-${Date.now()}-${seed}`;
          $('comparison-status').textContent=`${record.config.body.name}: episode ${run.episodes.length+1}/${seeds.length}`;
          const result=await api('/api/replay',{run_id:record.id,seed,checkpoint:'best',replay_id:job.replayId});
          if(result.cancelled||job.stopped)break;
          run.episodes.push({...result.summary,return:result.summary.return??result.summary.reward});
          run.checkpoint_steps=result.checkpoint_steps??record.steps;
        }
        run.status=run.episodes.length===seeds.length?'completed':'stopped';
      }catch(error){run.status='failed';run.error=error.message;}
      finally{job.replayId=null;}
    }
    comparisonData.status=job.stopped?'stopped':'completed';
    const rows=comparisonData.runs.map(run=>{
      const mean=key=>run.episodes.length?run.episodes.reduce((sum,e)=>sum+(e[key]??0),0)/run.episodes.length:null;
      return [comparisonLabel(run),`${format(run.training_steps)} / ${format(run.checkpoint_steps)}`,run.training_seed,`${run.episodes.length}/${seeds.length} · ${run.status}`,run.episodes.length?`${format(100*run.episodes.filter(e=>!e.fell).length/run.episodes.length)}%`:'—',format(mean('speed_error'),2),format(mean('return'),1)];
    });
    $('comparison-results').replaceChildren(...warnings.map(w=>element('p',w,'control-hint')),resultsTable(['Model','Trained / tested steps','Training seed','Episodes','Survival','Speed error','Return'],rows),
      ...comparisonData.runs.filter(run=>run.error).map(run=>element('p',`${run.config.body.name}: ${run.error}`,'control-hint')));
    $('comparison-status').textContent=job.stopped?'Comparison stopped. Partial results retained.':'Evaluation complete. Episode variation does not measure variation between training seeds.';
    $('export-comparison').hidden=false;
  }finally{comparisonJob=null;state.busy=false;controls();}
}
$('resume-training').addEventListener('click',resumeTraining);
$('run-comparison').addEventListener('click',compareModels);
$('stop-comparison').addEventListener('click',stopComparison);
$('export-comparison').addEventListener('click',()=>{
  const url=URL.createObjectURL(new Blob([JSON.stringify(comparisonData,null,2)],{type:'application/json'})),link=element('a');link.href=url;link.download='rlsim-comparison.json';link.click();setTimeout(()=>URL.revokeObjectURL(url),1000);
});
loadExplore();
