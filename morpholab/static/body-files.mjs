/** Local multi-file intake shared by robot, character and controller imports. */
import {unzipSync} from './vendor/zip.mjs';
export const MAX_BYTES=64*1024*1024;
export function safePath(input){
  if(typeof input!=='string'||!input||input.length>1024||/[\0\r\n]/.test(input))throw new Error('Invalid file path.');
  const path=input.replaceAll('\\','/');
  if(path.startsWith('/')||/^[a-z][a-z0-9+.-]*:/i.test(path)||path.split('/').includes('..'))throw new Error(`Use relative paths inside the imported folder: ${input}`);
  const clean=path.split('/').filter(part=>part&&part!=='.').join('/');if(!clean)throw new Error('Empty file path.');return clean;
}
export function bytesToBase64(bytes){let binary='';for(let i=0;i<bytes.length;i+=32768)binary+=String.fromCharCode(...bytes.subarray(i,i+32768));return btoa(binary);}
export function base64ToBytes(text){
  if(typeof text!=='string'||text.length>MAX_BYTES*1.4||!/^[A-Za-z0-9+/]*={0,2}$/.test(text))throw new Error('Invalid binary model data.');
  try{return Uint8Array.from(atob(text),c=>c.charCodeAt(0));}catch{throw new Error('Invalid binary model data.');}
}
export async function prepareBodyFiles(input){
  const files=[],paths=new Set();let total=0;
  function add(path,data){
    path=safePath(path);if(paths.has(path))throw new Error(`Duplicate file: ${path}`);
    if(files.length>=512)throw new Error('Import at most 512 files.');
    total+=data.length;if(total>MAX_BYTES)throw new Error('Imported files exceed 64 MB. Simplify meshes or textures.');
    paths.add(path);files.push({path,data});
  }
  for(const file of Array.from(input)){
    if(file.size>MAX_BYTES)throw new Error('Each file or ZIP must be smaller than 64 MB.');
    const bytes=new Uint8Array(await file.arrayBuffer());
    if(/\.(zip|mjz)$/i.test(file.name)){
      let size=total,count=files.length;
      const archive=unzipSync(bytes,{filter:entry=>{
        if(entry.name.endsWith('/')||entry.name.startsWith('__MACOSX/'))return false;
        safePath(entry.name);size+=entry.originalSize;
        if(++count>512||size>MAX_BYTES)throw new Error('Expanded archive exceeds 512 files or 64 MB.');return true;
      }});
      for(const [path,data]of Object.entries(archive))add(path,data);
    }else add(file.webkitRelativePath||file.name,bytes);
  }
  const decoder=new TextDecoder();
  const entries=files.flatMap(file=>{
    const format=file.path.split('.').at(-1).toLowerCase();
    if(!['xml','mjcf','urdf','glb','gltf','fbx','obj','stl','json','onnx'].includes(format))return [];
    if(['xml','mjcf'].includes(format)&&!/<\s*(mujoco|robot)(?:\s|\/?>)/.test(decoder.decode(file.data.subarray(0,16384))))return [];
    return [{path:file.path,format}];
  });
  if(!entries.length)throw new Error('No supported model found. Select MJCF, URDF, GLB/glTF, FBX, OBJ, STL, ONNX, or RLsim JSON with its assets.');
  return {files,entries};
}
