import assert from 'node:assert/strict';
import test from 'node:test';
import {zipSync} from '../morpholab/static/vendor/zip.mjs';
import {prepareBodyFiles,safePath} from '../morpholab/static/body-files.mjs';
const bytes=s=>new TextEncoder().encode(s);
const file=(name,data,path='')=>({name,webkitRelativePath:path,size:data.length,arrayBuffer:async()=>data.buffer.slice(data.byteOffset,data.byteOffset+data.byteLength)});
test('folder and ZIP imports preserve references and find entry models',async()=>{
  const content={'robot/main.xml':bytes('<mujoco/>'),'robot/fragment.xml':bytes('<mujocoinclude/>'),'robot/mesh.obj':bytes('v 0 0 0')};
  const result=await prepareBodyFiles([file('robot.zip',zipSync(content))]);
  assert.equal(result.files.length,3);assert.deepEqual(result.entries.map(e=>e.path),['robot/main.xml','robot/mesh.obj']);
  const folder=await prepareBodyFiles([file('robot.urdf',bytes('<robot/>'),'folder/robot.urdf')]);
  assert.equal(folder.files[0].path,'folder/robot.urdf');
});
test('ambiguous duplicate, traversal and unsupported inputs produce clear errors',async()=>{
  for(const path of ['../x','a/../../x','/absolute','C:\\x','https://x','a\0b'])assert.throws(()=>safePath(path));
  await assert.rejects(prepareBodyFiles([file('a.xml',bytes('<mujoco/>')),file('a.xml',bytes('<mujoco/>'))]),/Duplicate/);
  await assert.rejects(prepareBodyFiles([file('empty.txt',bytes('text'))]),/No supported/);
});
