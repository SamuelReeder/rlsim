/** Competence gates use completed training episodes, never elapsed budget. */
export const LEARNING_STAGES=Object.freeze([
  {name:"Balance",preset:"balance",speed_fraction:0,randomization:0},
  {name:"Slow walk",preset:"velocity",speed_fraction:.25,randomization:.25},
  {name:"Walk",preset:"velocity",speed_fraction:.6,randomization:.6},
  {name:"Target speed",preset:"velocity",speed_fraction:1,randomization:1},
]);
export class LearningCurriculum {
  constructor(targetSpeed){this.targetSpeed=targetSpeed;this.index=0;this.episodes=[];}
  get stage(){return LEARNING_STAGES[this.index];}
  record(episode){
    if(!Number.isFinite(episode.speed_error))return;
    this.episodes.push(episode);if(this.episodes.length>12)this.episodes.shift();
  }
  advance(){
    if(this.index===LEARNING_STAGES.length-1||this.episodes.length<12)return false;
    const survival=this.episodes.filter(e=>!e.fell).length/this.episodes.length;
    const error=this.episodes.reduce((sum,e)=>sum+e.speed_error,0)/this.episodes.length;
    const tolerance=Math.max(.15,this.targetSpeed*this.stage.speed_fraction*.35);
    if(survival<.8||error>tolerance)return false;
    this.index++;this.episodes=[];return true;
  }
  snapshot(){return {index:this.index,name:this.stage.name,episodes:this.episodes.length,
    speed:this.targetSpeed*this.stage.speed_fraction,randomization:this.stage.randomization};}
  serialize(){return {index:this.index,episodes:this.episodes.map(e=>({...e}))};}
  restore(data){
    if(!data || !Number.isInteger(data.index)||data.index<0||data.index>=LEARNING_STAGES.length||!Array.isArray(data.episodes)||data.episodes.length>12 ||
      data.episodes.some(e=>typeof e.fell!=='boolean'||!Number.isFinite(e.speed_error)||e.speed_error<0))throw new Error('Invalid curriculum continuation state.');
    this.index=data.index;this.episodes=data.episodes.map(e=>({...e}));return this;
  }
}
