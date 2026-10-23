import {validatePhysicalRig} from './physical-rig.mjs';

/**
 * Deterministic starting settings for an already fitted humanoid, not an
 * optimizer or a physiological model. No geometry or visual binding changes.
 *
 * Leg motors must support the body during stance, so their scale uses whole
 * body mass rather than only distal segment mass. These dimensionless torque
 * coefficients and the damping factor are engineering choices, not fitted
 * human strength measurements. A ball joint's torque is PER AXIS.
 * Damping resists velocity; it does not hold a rest pose or assist standing.
 */
const MOTOR = {
  thigh: {coefficient: .35, load: .5},
  shin: {coefficient: .28, load: .5},
  foot: {coefficient: .13, load: .5},
  toe: {coefficient: .025, load: .12},
  spine: {coefficient: .16, load: .45},
  shoulder: {coefficient: .07, load: .08},
  upperarm: {coefficient: .055, load: .06},
  forearm: {coefficient: .035, load: .035},
  hand: {coefficient: .012, load: .01},
  neck: {coefficient: .018, load: .08},
  head: {coefficient: .018, load: .08},
};
const clamp = (x, low, high) => Math.max(low, Math.min(high, x));
const distance = (a, b) => Math.hypot(...a.map((x, i) => x - b[i]));
const round = n => Number(n.toPrecision(7));
const controls = rig => rig.bodies.reduce((sum, b) =>
  sum + (b.joint.torque > 0 ? ({ball: 3, hinge: 1})[b.joint.type] || 0 : 0), 0);
function numberOption(value, low, high, name) {
  if (typeof value !== 'number' || !Number.isFinite(value) || value < low || value > high)
    throw new Error(`${name} must be between ${low} and ${high}.`);
  return value;
}
function anatomy(label) {
  let name = label.toLowerCase().replace(/^mixamorig[:_]?/, '').replace(/_\d+$/, '')
    .replace(/[^a-z0-9]/g, '');
  let side = '';
  if (name.includes('left')) {side = 'L'; name = name.replace('left', '');}
  else if (name.includes('right')) {side = 'R'; name = name.replace('right', '');}
  else if (/[lr]$/.test(name)) {side = name.at(-1).toUpperCase(); name = name.slice(0, -1);}
  else if (/^[lr](upper|lower|thigh|calf|foot|hand|arm|leg|shoulder|toe|finger)/.test(name)) {
    side = name[0].toUpperCase(); name = name.slice(1);
  }
  // A name alone never authorizes a finger lock: the caller also checks that
  // the chain descends from an identified hand of the same side.
  if (/^(?:hand)?(?:thumb|index|middle|ring|pinky|little)(?:finger)?\d*$/.test(name)
      || /^(?:hand)?finger\d+$/.test(name)) return {role: 'finger', side};
  if (side) {
    for (const [role, pattern] of [
      ['thigh', /^(upperleg|upleg|thigh|hip)$/],
      ['shin', /^(lowerleg|leg|shin|calf|knee)$/],
      ['foot', /^(foot|ankle)$/],
      ['toe', /^(toebase|toe\d*|toes)$/],
      ['shoulder', /^(shoulder|clavicle|collar)$/],
      ['upperarm', /^(upperarm|arm)$/],
      ['forearm', /^(lowerarm|forearm|elbow)$/],
      ['hand', /^(hand|wrist|fist)$/],
    ]) if (pattern.test(name)) return {role, side};
  }
  if (/^(spine\d*|chest\d*|upperchest|abdomen\d*|torso\d*)$/.test(name)) return {role: 'spine', side};
  if (/^neck\d*$/.test(name)) return {role: 'neck', side};
  if (/^head\d*$/.test(name)) return {role: 'head', side};
  return {role: null, side};
}
function legEvidence(rig, roles) {
  const legs = [];
  for (const side of ['L', 'R']) {
    const matches = role => rig.bodies.filter(b => roles.get(b.id).role === role && roles.get(b.id).side === side);
    const [thighs, shins, feet] = ['thigh', 'shin', 'foot'].map(matches);
    if ([thighs, shins, feet].some(parts => parts.length !== 1)) return null;
    const [thigh, shin, foot] = [thighs[0], shins[0], feet[0]];
    if (shin.parent !== thigh.id || foot.parent !== shin.id) return null;
    const upper = distance(thigh.position, shin.position), lower = distance(shin.position, foot.position);
    const length = upper + lower;
    if (length < .15 || length > 2.5 || upper / length < .2 || upper / length > .8
        || thigh.position[2] < foot.position[2] + .05) return null;
    legs.push({thigh, shin, foot, upper, lower, length});
  }
  if (legs[0].thigh.parent !== legs[1].thigh.parent
      || Math.max(...legs.map(l => l.length)) / Math.min(...legs.map(l => l.length)) > 1.4) return null;
  return legs;
}
function isFinger(part, roles, byID) {
  const candidate = roles.get(part.id);
  if (candidate.role !== 'finger') return false;
  for (let parent = byID.get(part.parent); parent; parent = byID.get(parent.parent)) {
    const role = roles.get(parent.id);
    if (role.role === 'hand') return !candidate.side || candidate.side === role.side;
    if (role.role && role.role !== 'finger') return false;
  }
  return false;
}
function colliderReach(part) {
  return Math.max(...part.colliders.map(c => Math.hypot(...c.position)
    + (c.type === 'sphere' ? c.size[0] : c.type === 'capsule' ? c.size[0] + c.size[1] : Math.hypot(...c.size))));
}
const snapshot = b => ({type: b.joint.type, torque: b.joint.torque, damping: b.joint.damping, mass: b.mass});

/**
 * @returns {{rig: object|null, changes: object[], warnings: string[],
 * summary: string, analysis: object}}
 * Unsupported or ambiguous anatomy returns rig:null. Invalid options/rigs
 * throw before making a proposal. Omitted totalMass preserves the current sum.
 */
export function recommendLocomotionSetup(body, options = {}) {
  const strength = numberOption(options.strengthScale ?? 1, .25, 2, 'Strength scale');
  const gravity = numberOption(options.gravity ?? 9.81, .1, 50, 'Gravity');
  const lockFingers = options.lockFingers ?? true;
  if (typeof lockFingers !== 'boolean') throw new Error('Lock fingers must be true or false.');
  if (options.totalMass !== undefined) numberOption(options.totalMass, .1, 500, 'Total mass (kg)');
  const unsupported = reason => ({rig: null, changes: [], warnings: [reason],
    summary: 'A humanoid locomotion setup is unavailable for this body.', analysis: {supported: false, reason}});
  const input = body?.physical_rig;
  if (!input || input.profile !== 'humanoid')
    return unsupported('This setup needs an editable humanoid physical rig. Review other body types manually.');
  validatePhysicalRig(input);
  // Preserve exact rest transforms, colliders, source mappings and other
  // authored metadata; validation above need not normalize/rewrite them.
  const rig = structuredClone(input), before = new Map(input.bodies.map(b => [b.id, b]));
  const byID = new Map(rig.bodies.map(b => [b.id, b]));
  const roles = new Map(rig.bodies.map(b => [b.id, anatomy(b.label)]));
  const legs = legEvidence(rig, roles);
  if (!legs) return unsupported('Could not confidently identify two connected thigh, knee and foot chains at a plausible metre scale. Review names, parents and scale first.');
  const originalMass = rig.bodies.reduce((sum, b) => sum + b.mass, 0);
  const mass = numberOption(options.totalMass ?? originalMass, .1, 500, 'Total mass (kg)');
  const legLength = legs.reduce((sum, l) => sum + l.length, 0) / 2;
  const warnings = [
    'These are starting estimates, not measured optimal or physiological strengths. Validate and compare short training trials before a long run.',
    'Ball-joint torque is per axis. Damping only resists motion; it does not hold a standing posture.',
    'Existing collider-based mass shares, joint limits and collision shapes are retained; clothing geometry can still bias the physical model.',
  ];
  const changedMass = mass !== originalMass;
  if (changedMass) {
    for (const part of rig.bodies) part.mass *= mass / originalMass;
    // Place roundoff in the largest body, never in a tiny finger segment.
    const largest = rig.bodies.reduce((best, part) => part.mass > best.mass ? part : best);
    largest.mass += mass - rig.bodies.reduce((sum, part) => sum + part.mass, 0);
  }
  const unknown = [], passive = [], fixedMajor = [], capped = [];
  let fingersLocked = 0;
  const newRepairs = [];
  for (const part of rig.bodies) {
    const {role} = roles.get(part.id), joint = part.joint;
    if (lockFingers && isFinger(part, roles, byID)) {
      if (joint.type !== 'fixed') fingersLocked++;
      joint.type = 'fixed'; joint.torque = 0;
      continue;
    }
    if (joint.type === 'fixed' || joint.type === 'free') {
      if (joint.type === 'fixed' && ['thigh', 'shin', 'foot'].includes(role)) fixedMajor.push(part.label);
      continue;
    }
    if (role === 'finger') {
      if (lockFingers) unknown.push(part.label);
      continue;
    }
    if (!MOTOR[role]) {unknown.push(part.label); continue;}
    // Preserve deliberately passive joints; a setup must not silently invent
    // new actuators or change the observation/action layout except finger locks.
    if (joint.torque === 0) {passive.push(part.label); continue;}
    const rule = MOTOR[role];
    const torque = mass * gravity * legLength * rule.coefficient * strength;
    if (torque > 1000) capped.push(part.label);
    joint.torque = round(clamp(torque, .02, 1000));
    const children = rig.bodies.filter(b => b.parent === part.id);
    const observedReach = Math.max(colliderReach(part), ...children.map(child => distance(part.position, child.position)));
    const lever = ['thigh', 'shin', 'foot'].includes(role) ? legLength : clamp(observedReach, legLength * .05, legLength);
    const inertia = Math.max(part.mass, mass * rule.load) * lever * lever / 3;
    joint.damping = round(clamp(.2 * Math.sqrt(inertia * joint.torque), .02, 30));
  }
  if (unknown.length) warnings.push(`Unrecognized or unverified joints kept their motor settings: ${unknown.join(', ')}.`);
  if (passive.length) warnings.push(`Deliberately unpowered joints remain unpowered: ${passive.join(', ')}.`);
  if (fixedMajor.length) warnings.push(`Existing fixed leg joints remain fixed: ${fixedMajor.join(', ')}. Review whether they need to move for this task.`);
  if (capped.length) warnings.push(`Torque was capped at 1,000 Nm per axis for: ${capped.join(', ')}. Review body units and strength.`);
  if (lockFingers && !fingersLocked) warnings.push('No additional confidently identified finger joints needed locking.');
  if (fingersLocked) newRepairs.push(`Locomotion setup fixed ${fingersLocked} finger joints in their imported rest pose, retaining their bodies, masses and visual bindings.`);
  newRepairs.push(`Locomotion motor estimates use ${round(mass)} kg, ${gravity} m/s² gravity and ${strength}× strength; anatomy and geometry are unchanged.`);
  rig.repairs = [...new Set([...(rig.repairs || []), ...newRepairs])].slice(-256);
  validatePhysicalRig(rig);
  const changes = rig.bodies.flatMap(part => {
    const prior = snapshot(before.get(part.id)), after = snapshot(part);
    return Object.keys(prior).some(key => prior[key] !== after[key])
      ? [{id: part.id, label: part.label, role: roles.get(part.id).role || 'unrecognized', before: prior, after}] : [];
  });
  const analysis = {
    supported: true, total_mass_before: originalMass, total_mass_after: rig.bodies.reduce((sum, b) => sum + b.mass, 0),
    controls_before: controls(input), controls_after: controls(rig), fingers_locked: fingersLocked,
    leg_length: legLength, strength_scale: strength, gravity,
  };
  return {rig, changes, warnings,
    summary: `${fingersLocked} finger joints locked; ${analysis.controls_before} → ${analysis.controls_after} motor controls. Starting motor strengths for ${round(mass)} kg.`,
    analysis};
}
