import assert from 'node:assert/strict';
import test from 'node:test';
import loadMujoco from '@mujoco/mujoco';
import {physicalRigXML} from '../morpholab/static/physical-rig.mjs';
import {recommendLocomotionSetup} from '../morpholab/static/locomotion-setup.mjs';

const mj = await loadMujoco();
function humanoid() {
  const bodies = [];
  const add = (label, parent, position, type = 'ball') => {
    bodies.push({id: label, label, parent, position, quaternion: [1, 0, 0, 0], mass: 60 / 52,
      joint: {type, axis: [1, 0, 0], range: type === 'hinge' ? [-5, 145] : [0, 85],
        torque: ['free', 'fixed'].includes(type) ? 0 : 60, damping: 1},
      colliders: [{type: 'sphere', position: [0, 0, 0], quaternion: [1, 0, 0, 0], size: [.025]}],
      source_bones: [`scene/${label}`]});
  };
  add('Hips', null, [0, 0, 1], 'free');
  add('Spine', 'Hips', [0, 0, 1.15]);
  add('Spine1', 'Spine', [0, 0, 1.3]);
  add('Spine2', 'Spine1', [0, 0, 1.45]);
  add('Neck', 'Spine2', [0, 0, 1.6]);
  add('Head', 'Neck', [0, 0, 1.72], 'fixed');
  for (const [side, sign] of [['Left', 1], ['Right', -1]]) {
    add(`${side}UpLeg`, 'Hips', [.12 * sign, 0, 1]);
    add(`${side}LowerLeg`, `${side}UpLeg`, [.12 * sign, 0, .55], 'hinge');
    add(`${side}Foot`, `${side}LowerLeg`, [.12 * sign, 0, .1]);
    add(`${side}ToeBase`, `${side}Foot`, [.12 * sign, .15, .07]);
    add(`${side}Shoulder`, 'Spine2', [.1 * sign, 0, 1.45]);
    add(`${side}Arm`, `${side}Shoulder`, [.2 * sign, 0, 1.45]);
    add(`${side}ForeArm`, `${side}Arm`, [.5 * sign, 0, 1.45]);
    add(`${side}Hand`, `${side}ForeArm`, [.75 * sign, 0, 1.45]);
    for (const digit of ['Thumb', 'Index', 'Middle', 'Ring', 'Pinky']) {
      for (let i = 1; i <= 3; i++) add(`${side}Hand${digit}${i}`,
        i === 1 ? `${side}Hand` : `${side}Hand${digit}${i - 1}`, [( .75 + .04 * i) * sign, 0, 1.45]);
    }
  }
  return {physical_rig: {version: 1, profile: 'humanoid', root: 'Hips', bodies, warnings: [], repairs: []}};
}
const compile = rig => mj.MjModel.from_xml_string(physicalRigXML('Locomotion check', rig));
const part = (body, label) => body.physical_rig.bodies.find(b => b.label === label);

test('locks 30 finger DOFs and motors, retaining all bodies and rest transforms', () => {
  const body = humanoid(), original = structuredClone(body);
  const proposal = recommendLocomotionSetup(body), oldModel = compile(body.physical_rig), model = compile(proposal.rig);
  try {
    assert.equal(oldModel.nu, 146);
    assert.equal(model.nu, 56);
    assert.equal(oldModel.nv - model.nv, 90, 'Fixed fingers remove actual DOFs, not only motor controls');
    assert.equal(model.nbody, oldModel.nbody);
    assert.equal(proposal.analysis.fingers_locked, 30);
    assert.deepEqual(body, original, 'Proposal cannot mutate caller state');
    for (const b of proposal.rig.bodies) {
      const old = part(original, b.label);
      for (const key of ['id', 'parent', 'position', 'quaternion', 'colliders', 'source_bones', 'mass'])
        assert.deepEqual(b[key], old[key], `${b.label}: preserved ${key}`);
      assert.deepEqual(b.joint.axis, old.joint.axis);
      assert.deepEqual(b.joint.range, old.joint.range);
      if (!/Hand(Thumb|Index|Middle|Ring|Pinky)/.test(b.label)) assert.equal(b.joint.type, old.joint.type);
    }
    assert.equal(proposal.rig.bodies.find(b => b.label === 'LeftToeBase').joint.type, 'ball');
    assert.equal(proposal.rig.bodies.find(b => b.label === 'LeftHand').joint.type, 'ball');
  } finally {model.delete(); oldModel.delete();}
});

test('fixed fingers follow a moving hand and preserve relative transforms in real physics', () => {
  const proposal = recommendLocomotionSetup(humanoid()), model = compile(proposal.rig), data = new mj.MjData(model);
  try {
    const hand = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY.value, 'LeftHand');
    const digit = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY.value, 'LeftHandIndex3');
    const handJoint = mj.mj_name2id(model, mj.mjtObj.mjOBJ_JOINT.value, 'joint_LeftHand');
    mj.mj_forward(model, data);
    const before = Array.from(data.xpos.slice(digit * 3, digit * 3 + 3));
    const q = model.jnt_qposadr[handJoint];
    data.qpos[q] = Math.cos(.3); data.qpos[q + 3] = Math.sin(.3);
    mj.mj_forward(model, data);
    assert.ok(Math.hypot(...before.map((v, i) => v - data.xpos[digit * 3 + i])) > .04);
    const relativeDistance = Math.hypot(...Array.from({length: 3}, (_, i) => data.xpos[hand * 3 + i] - data.xpos[digit * 3 + i]));
    assert.ok(Math.abs(relativeDistance - .12) < 1e-8);
    for (let i = 0; i < 200; i++) {
      for (let a = 0; a < model.nu; a++) data.ctrl[a] = .05 * Math.sin(i * .03 + a);
      mj.mj_step(model, data);
    }
    assert.ok(data.qpos.every(Number.isFinite));
    assert.ok(data.qvel.every(Number.isFinite));
    for (const kind of ['mjWARN_BADQPOS', 'mjWARN_BADQVEL', 'mjWARN_BADQACC']) {
      const warning = data.warning.get(mj.mjtWarning[kind].value);
      assert.equal(warning.number, 0, kind);
      warning.delete();
    }
  } finally {data.delete(); model.delete();}
});

test('mass scales existing shares and joint strength/damping respond to load', () => {
  const body = humanoid(), base = recommendLocomotionSetup(body), heavy = recommendLocomotionSetup(body, {totalMass: 120});
  assert.ok(Math.abs(heavy.rig.bodies.reduce((sum, b) => sum + b.mass, 0) - 120) < 1e-10);
  for (const b of heavy.rig.bodies) assert.ok(Math.abs(b.mass / part(body, b.label).mass - 2) < 1e-12);
  const lookup = (p, name) => p.rig.bodies.find(b => b.label === name).joint;
  assert.equal(lookup(heavy, 'LeftLowerLeg').torque, 2 * lookup(base, 'LeftLowerLeg').torque);
  assert.ok(lookup(heavy, 'LeftLowerLeg').damping > lookup(base, 'LeftLowerLeg').damping);
  assert.ok(lookup(base, 'LeftUpLeg').torque > lookup(base, 'LeftLowerLeg').torque);
  assert.ok(lookup(base, 'LeftLowerLeg').torque > lookup(base, 'LeftFoot').torque);
  assert.ok(lookup(base, 'LeftFoot').torque > lookup(base, 'LeftHand').torque);
  for (const b of heavy.rig.bodies.filter(b => b.joint.torque > 0)) assert.ok(b.joint.damping > 0);
});

test('explicit strength/gravity scale torque and finger opt-out leaves finger motors', () => {
  const body = humanoid(), base = recommendLocomotionSetup(body);
  const scaled = recommendLocomotionSetup(body, {strengthScale: .5, gravity: 4.905, lockFingers: false});
  assert.equal(scaled.analysis.controls_after, 146);
  const knee = p => p.rig.bodies.find(b => b.label === 'LeftLowerLeg').joint.torque;
  assert.equal(knee(scaled), knee(base) * .25);
  const finger = scaled.rig.bodies.find(b => b.label === 'LeftHandThumb1');
  assert.deepEqual(finger.joint, part(body, finger.label).joint);
});

test('rejects unsafe inputs before returning a changed rig', () => {
  for (const options of [
    {totalMass: NaN}, {totalMass: 0}, {totalMass: 501}, {totalMass: '60'},
    {strengthScale: 0}, {strengthScale: 2.1}, {gravity: 0}, {gravity: Infinity}, {lockFingers: 'true'},
  ]) assert.throws(() => recommendLocomotionSetup(humanoid(), options), /must be/);
  const body = humanoid(); part(body, 'LeftHand').mass = -1;
  assert.throws(() => recommendLocomotionSetup(body), /mass/);
});

test('generic, ambiguous, disconnected or implausibly scaled anatomy cannot get an automatic setup', () => {
  for (const mutate of [
    b => b.physical_rig.profile = 'generic',
    b => part(b, 'LeftLowerLeg').parent = 'Hips',
    b => part(b, 'LeftUpLeg').label = 'Unknown',
    b => part(b, 'LeftFoot').position[2] = -20,
    b => part(b, 'LeftHand').label = 'LeftFoot',
  ]) {
    const body = humanoid(); mutate(body);
    const proposal = recommendLocomotionSetup(body);
    assert.equal(proposal.rig, null); assert.equal(proposal.analysis.supported, false);
    assert.ok(proposal.warnings.length);
  }
});

test('keeps unrecognized, passive and existing fixed joints; locks only fingers under verified hands', () => {
  const body = humanoid();
  part(body, 'LeftHandIndex1').parent = 'Hips';
  part(body, 'RightHandThumb1').label = 'ExtraThing';
  part(body, 'LeftForeArm').joint.type = 'fixed';
  part(body, 'RightForeArm').joint.torque = 0;
  const proposal = recommendLocomotionSetup(body);
  const byID = new Map(proposal.rig.bodies.map(b => [b.id, b]));
  assert.equal(byID.get('LeftHandIndex1').joint.type, 'ball');
  assert.deepEqual(byID.get('RightHandThumb1').joint, part(body, 'ExtraThing').joint);
  assert.deepEqual(byID.get('LeftForeArm').joint, part(body, 'LeftForeArm').joint);
  assert.deepEqual(byID.get('RightForeArm').joint, part(body, 'RightForeArm').joint);
  assert.ok(proposal.warnings.some(w => w.includes('LeftHandIndex1') && w.includes('ExtraThing')));
  assert.ok(proposal.warnings.some(w => w.includes('unpowered') && w.includes('RightForeArm')));
});

test('common suffix and Mixamo finger names work without freezing hands or toes', () => {
  const body = humanoid();
  part(body, 'LeftHandIndex1').label = 'IndexFinger1L';
  part(body, 'LeftHandIndex2').label = 'IndexFinger2L';
  part(body, 'RightHandIndex1').label = 'mixamorig:RightHandIndex1';
  const proposal = recommendLocomotionSetup(body);
  assert.equal(proposal.analysis.fingers_locked, 30);
  assert.equal(proposal.rig.bodies.find(b => b.id === 'LeftHand').joint.type, 'ball');
  assert.equal(proposal.rig.bodies.find(b => b.id === 'LeftToeBase').joint.type, 'ball');
});
