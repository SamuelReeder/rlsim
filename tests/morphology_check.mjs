import assert from 'node:assert/strict';
import test from 'node:test';
import {proportionBody,proportionRegion,validateMorphology} from '../morpholab/static/morphology.mjs';
import {PRESETS,validateBody} from '../morpholab/static/browser-physics.mjs';
test('proportion inference changes geometry while keeping topology and the source body',()=>{
  const config={body:PRESETS.quadruped,morphology:validateMorphology({})};
  const old=JSON.stringify(config),body=proportionBody(config,[.85,1.18,.82],validateBody);
  assert.equal(body.legs,4);assert.equal(body.upper_length,.42*1.18);
  assert.equal(body.body_mass,4);assert.equal(body.motor_strength,60);
  assert.equal(JSON.stringify(config),old);
  assert.equal(proportionRegion([.85,1.18,.82]),'withheld region');
  assert.equal(proportionRegion([1,1.1,.9]),'withheld region');
  assert.equal(proportionRegion([1,1.35,1.3]),'outside training range');
  assert.equal(proportionRegion([1,1,1]),'training range');
});
test('incompatible controllers and invalid proportion settings are rejected',()=>{
  for(const config of [{body:PRESETS.quadruped},{body:{kind:'mjcf'},morphology:{}}])
    assert.throws(()=>proportionBody(config,[1,1,1],validateBody),/compatible/);
  for(const values of [[1,1],[1,NaN,1],[1,1,2]])
    assert.throws(()=>proportionBody({body:PRESETS.quadruped,morphology:{}},values,validateBody));
  assert.throws(()=>validateMorphology({width:.7}),/Invalid/);
});

test('broader controllers retain the original excluded corner',()=>{
  assert.equal(proportionRegion([1,1.35,1.3],.4,.2),'training range');
  assert.equal(proportionRegion([1,1.16,.84],.4,.2),'withheld region');
  assert.equal(proportionRegion([1,1.16,.84],.4),'training range');
});
