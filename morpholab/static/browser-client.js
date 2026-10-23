"use strict";
// Each origin has its own model library; no policy or training data is uploaded.
class BrowserBackend {
  constructor() {
    this.pending=new Map();this.replays=new Map();this.sequence=0;this.writes=Promise.resolve();this.sceneMeshes={};
    this.worker=new Worker(new URL("browser-worker.js",document.baseURI));
    this.database=new Promise((resolve,reject)=>{
      const request=indexedDB.open("morpholab-browser-v1",1);
      request.onupgradeneeded=()=>request.result.createObjectStore("runs",{keyPath:"id"});
      request.onsuccess=()=>resolve(request.result);
      request.onerror=()=>reject(new Error("Browser model storage is unavailable. Allow site storage to save models."));
    });
    this.database.catch(()=>{});
    this.worker.onmessage=async ({data})=>{
      if(data.event==='status'){
        this.hydrateStatus(data.status);
      }else if(data.event==='replay_progress'){
        const pending=this.pending.get(data.id);
        if(pending){
          const progress=data.progress;
          if(progress.phase==='ready')pending.replay={...progress,frames:[...progress.frames]};
          else if(progress.frames&&pending.replay)pending.replay.frames.push(...progress.frames);
          // A presentation callback must not break the worker protocol or strand
          // a request whose simulation is still running.
          try{pending.onProgress?.(progress);}catch(error){console.error('Replay progress callback failed:',error);}
        }
      }else if(data.event==="checkpoint") {
        this.writes=this.writes.catch(()=>{}).then(()=>this.put(data.record));
        try{await this.writes;this.worker.postMessage({op:"saved",token:data.token});}
        catch(error){this.worker.postMessage({op:"saved",token:data.token,error:error.message});}
      } else if(!data.event){const pending=this.pending.get(data.id);if(pending){this.pending.delete(data.id);data.error ? pending.reject(new Error(data.error)) : pending.resolve(data.result?.streamed&&pending.replay?{...pending.replay,...data.result,frames:pending.replay.frames}:data.result);}}
    };
    this.worker.onerror=event=>{
      this.failure=new Error(event.message || "Browser worker failed. Reload the page to restart it.");
      for(const item of this.pending.values())item.reject(this.failure);
      this.pending.clear();
    };
  }
  hydrateStatus(status){
    if(!status)return status;
    if(status.run_id!==this.meshRun){
      this.sceneMeshes={};this.sceneConfig=undefined;this.sceneModelInfo=undefined;this.meshRun=status.run_id;
    }
    if(status.config)this.sceneConfig=status.config;
    if(status.model_info)this.sceneModelInfo=status.model_info;
    Object.assign(this.sceneMeshes,status.frame?.meshes||{});
    return {...status,...(this.sceneConfig?{config:this.sceneConfig}:{}),...(this.sceneModelInfo?{model_info:this.sceneModelInfo}:{}),
      ...(status.frame&&Object.keys(this.sceneMeshes).length?{frame:{...status.frame,meshes:this.sceneMeshes}}:{})};
  }
  call(op,payload={},options={}){
    if(this.failure)return Promise.reject(this.failure);
    const id=++this.sequence;
    return new Promise((resolve,reject)=>{this.pending.set(id,{resolve,reject,onProgress:options.onProgress});this.worker.postMessage({id,op,payload});});
  }
  async put(record){
    const db=await this.database;
    return new Promise((resolve,reject)=>{
      const tx=db.transaction("runs","readwrite");tx.objectStore("runs").put(record);
      tx.oncomplete=()=>resolve(record);tx.onerror=()=>reject(new Error(`Could not save this model: ${tx.error?.message || "browser storage error"}`));
      tx.onabort=()=>reject(new Error("Model storage was interrupted. Free some site storage and try again."));
    });
  }
  async records(id){
    const db=await this.database;
    return new Promise((resolve,reject)=>{
      const store=db.transaction("runs").objectStore("runs"),request=id ? store.get(id) : store.getAll();
      request.onsuccess=()=>resolve(request.result);request.onerror=()=>reject(request.error);
    });
  }
  async request(path,payload={},options={}){
    const op=path.split("/").at(-1).split("?")[0];
    if(op==="runs"){
      await this.writes;
      return {runs:(await this.records()).sort((a,b)=>b.created_at.localeCompare(a.created_at)).map(({policy,controller,normalizer,body_xml,best,continuation,...summary})=>({...summary,resumable:continuation?.format==='rlsim-training-state-v1',
        ...(best?{best:{steps:best.steps,validation:best.validation}}:{}),controller_type:controller?"onnx":"ppo"}))};
    }
    if(op==="train")await this.database;
    if(op==='resume'){
      await this.writes;
      const record=await this.records(payload.run_id);
      if(!record)throw new Error('Select a saved model to continue.');
      return this.hydrateStatus(await this.call('resume',{record,additional_steps:payload.additional_steps}));
    }
    if(op==="stop-replay"){
      const replay=this.replays.get(payload.replay_id);
      if(replay){
        replay.cancelled=true;
        if(replay.started)await this.call('stop_replay',payload);
      }
      return {stopping:Boolean(replay)};
    }
    if(op==="replay"){
      const replay={cancelled:false,started:false};
      if(payload.replay_id)this.replays.set(payload.replay_id,replay);
      try{
      const stream=typeof options.onProgress==='function';
      if(stream){try{options.onProgress({phase:'loading',stage:payload.run_id?'storage':'physics'});}catch(error){console.error('Replay progress callback failed:',error);}}
      let record;
      if(payload.run_id){record=await this.records(payload.run_id);if(!record)throw new Error("This browser model was not found.");}
      if(replay.cancelled)return {cancelled:true,frames:[]};
      replay.started=true;
      return await this.call(op,{...payload,...(record?{record}:{}),stream},options);
      }finally{if(this.replays.get(payload.replay_id)===replay)this.replays.delete(payload.replay_id);}
    }
    const result=await this.call(op,payload,options);
    return op==='status'||op==='train'||op==='stop'?this.hydrateStatus(result):result;
  }
  download(contents,name,type='application/json'){
    const url=URL.createObjectURL(new Blob([contents],{type})),link=document.createElement('a');
    link.href=url;link.download=name;link.click();setTimeout(()=>URL.revokeObjectURL(url),10000);
  }
  async prepareBodyFiles(files){return (await import('./body-files.mjs')).prepareBodyFiles(files);}
  async importBody(bundle,options={}){
    const entry=options.entry||bundle.entries[0]?.path,format=entry?.split('.').at(-1).toLowerCase();
    if(['glb','gltf','fbx','obj','stl'].includes(format)){
      if(!window.RLsimScene?.importVisualFiles)throw new Error('3D importer is unavailable. Run npm run vendor:browser and reload.');
      const body=await window.RLsimScene.importVisualFiles(bundle.files,{...options,entry});
      return this.call('preview',{body,environment:options.environment});
    }
    if(format==='json'){
      const file=bundle.files.find(file=>file.path===entry),record=JSON.parse(new TextDecoder().decode(file.data));
      let body=record.body||record;
      if(body.physical_rig){
        if(!window.RLsimScene?.rebuildPhysicalRig)throw new Error('Physics rig editor is unavailable. Reload the page after rebuilding browser assets.');
        body=window.RLsimScene.rebuildPhysicalRig(body,body.physical_rig);
      }
      return this.call('preview',{body,environment:record.environment||options.environment});
    }
    return this.call('import_body',{files:bundle.files,options:{...options,entry},environment:options.environment});
  }
  async exportInterface(config){
    const result=await this.call('interface',config);
    this.download(JSON.stringify(result,null,2),'rlsim-controller-interface.json');return result;
  }
  async exportModel(id,type){
    const record=await this.records(id);if(!record)throw new Error("This browser model was not found.");
    if(type==='body' && record.config.body.kind==='mjcf' && record.config.body.assets.length){
      const {zipSync,strToU8}=await import('./vendor/zip.mjs'),{base64ToBytes}=await import('./body-files.mjs');
      const paths=new Set(record.config.body.assets.map(asset=>asset.path));
      let entry='body.xml',suffix=0;
      while(paths.has(entry))entry=`rlsim-body-${++suffix}.xml`;
      const files={[entry]:strToU8(record.body_xml)};
      for(const asset of record.config.body.assets)files[asset.path]=base64ToBytes(asset.data);
      this.download(zipSync(files),'rlsim-body.zip','application/zip');
    }else this.download(type==='body'?record.body_xml:JSON.stringify(record),type==='body'?'body.xml':`${id}.json`,type==='body'?'application/xml':'application/json');
  }
  async importModel(file,context={}){
    const limit=/\.json$/i.test(file.name)?100:64;
    if(file.size>limit*1024*1024)throw new Error(`Controller ${limit===100?"JSON files":"packages"} must be smaller than ${limit} MB.`);
    let source;
    if(/\.json$/i.test(file.name))source=JSON.parse(await file.text());
    else {
      const {bytesToBase64,safePath}=await import('./body-files.mjs');
      const bundle=await this.prepareBodyFiles([file]);
      const manifests=bundle.files.filter(file=>/\.json$/i.test(file.path)).flatMap(file=>{
        try{const value=JSON.parse(new TextDecoder().decode(file.data));return value.format==='rlsim-onnx-manifest-v1'?[{path:file.path,value}]:[];}catch{return [];}
      });
      if(manifests.length>1)throw new Error('Include exactly one RLsim ONNX manifest in the ZIP.');
      const manifest=manifests[0]?.value,folder=manifests[0]?.path.split('/').slice(0,-1).join('/');
      const models=bundle.files.filter(file=>/\.onnx$/i.test(file.path));
      const resolve=path=>{
        path=safePath(path);return bundle.files.find(file=>file.path===(folder?folder+'/':'')+path)||bundle.files.find(file=>file.path===path);
      };
      const model=manifest?.model?resolve(manifest.model):models.length===1?models[0]:null;
      if(!model)throw new Error('Include one .onnx model, or name it in an rlsim-onnx-manifest-v1 manifest.');
      const config=manifest?.config||context;
      if(!config.body)throw new Error('A controller needs a body. Select a body first or include config.body in its manifest.');
      const externalData=(manifest?.external_data||[]).map(item=>{
        const path=typeof item==='string'?item:item.path,file=resolve(typeof item==='string'?item:item.file||path);
        if(!file)throw new Error(`Missing ONNX weights: ${path}`);
        return {path:safePath(path),data:bytesToBase64(file.data)};
      });
      source={format:'morpholab-browser-run-v1',config,steps:manifest?.steps||0,
        controller:{format:'rlsim-onnx-v1',data:bytesToBase64(model.data),input_name:manifest?.input_name,output_name:manifest?.output_name,
          observation_schema:manifest?.observation_schema,action_schema:manifest?.action_schema,external_data:externalData},normalizer:manifest?.normalizer||null,
        history:[],name:manifest?.name||file.name,source:'onnx'};
    }
    const record=await this.call('validate',source);await this.put(record);return record;

  }
}
window.BrowserBackend=BrowserBackend;
