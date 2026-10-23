// Aggregate independent training runs, never treat evaluation episodes as seeds.
export function aggregateTrials(trials){
  const groups=new Map();
  for(const trial of trials){
    const key=JSON.stringify([trial.body,trial.condition]);
    if(!groups.has(key))groups.set(key,{body:trial.body,condition:trial.condition,trials:[]});
    groups.get(key).trials.push(trial);
  }
  return [...groups.values()].map(group=>{
    const completed=group.trials.filter(t=>t.status==='completed'&&t.metrics);
    const stats=key=>{
      const values=completed.map(t=>t.metrics[key]).filter(Number.isFinite);
      if(!values.length)return {mean:null,sd:null};
      const mean=values.reduce((a,b)=>a+b,0)/values.length;
      return {mean,sd:values.length>1?Math.sqrt(values.reduce((sum,v)=>sum+(v-mean)**2,0)/(values.length-1)):null};
    };
    return {...group,completed:completed.length,failed:group.trials.filter(t=>t.status==='failed').length,
      survival:stats('survival_rate'),speed_error:stats('speed_error'),distance:stats('distance'),reward:stats('return')};
  });
}
export function comparisonWarnings(records){
  const differences=[];
  for(const [label,value] of [
    ['bodies',r=>r.config.body],['tasks',r=>r.config.task??null],['world settings',r=>r.config.environment],
    ['target speeds',r=>r.config.target_speed],['training steps',r=>r.steps],['training backends',r=>r.hardware?.execution_location??r.source??'unknown']]){
    if(new Set(records.map(r=>JSON.stringify(value(r)))).size>1)differences.push(label);
  }
  const seeds=new Set(records.map(r=>r.config.seed));
  return [...(differences.length?[`Different ${differences.join(', ')}: this is a descriptive comparison, not a controlled experiment.`]:[]),
    ...(seeds.size<records.length?['Some models share a training seed or run lineage; they are not independent training repetitions.']:[])];
}
export const COMPARISON_SEEDS=[700000007,700000021,700000039];
