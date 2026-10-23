import test from 'node:test';
import assert from 'node:assert/strict';
import load from '@mujoco/mujoco';
import {checkPhysicalRig} from '../morpholab/static/rig-checks.mjs';
import {preparePhysicsFiles} from '../morpholab/static/imported-physics.mjs';
const mj=await load();
async function check(xml){const body=await preparePhysicsFiles(mj,[{path:'body.xml',data:new TextEncoder().encode(xml)}]);return checkPhysicalRig(mj,body);}
const xml=`<mujoco><option timestep=".005"/><worldbody><geom type="plane" size="0 0 .1"/><body name="root" pos="0 0 1"><freejoint/><geom type="sphere" size=".15" mass="2"/><body name="limb" pos="0 0 -.3"><joint name="joint" type="hinge" range="-30 30"/><geom type="capsule" size=".07 .2" mass="1"/></body></body></worldbody><actuator><motor joint="joint" gear="10" ctrlrange="-1 1"/></actuator></mujoco>`;
test('real physics probes motor responses and separates falling from invalid setup',async()=>{
  const report=await check(xml.replace('pos="0 0 1"','pos="0 0 3"'));
  assert.notEqual(report.status,'needs-fixes',JSON.stringify(report));
  assert.equal(report.checks.find(c=>c.id==='motors').level,'pass');
  assert.equal(report.checks.find(c=>c.id==='support').fell,true);
  assert.equal(report.checks.find(c=>c.id==='support').level,'info');
});
test('detects deep floor penetration and inert motors',async()=>{
  const report=await check(xml.replace('pos="0 0 1"','pos="0 0 .05"').replace('gear="10"','gear="0"'));
  assert.equal(report.checks.find(c=>c.id==='floor').level,'warning');
  assert.equal(report.checks.find(c=>c.id==='motors').level,'warning');
});
test('rejects invalid physical graphs without throwing away the current body',async()=>{
  const report=await checkPhysicalRig(mj,{physical_rig:{version:9}});
  assert.equal(report.status,'needs-fixes');
});
