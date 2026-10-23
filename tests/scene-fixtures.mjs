const encoder = new TextEncoder();

/** Small original fixtures, generated here rather than downloaded assets. */
export function tinyRigGLB(boneCount = 3) {
  const chunks = [], bufferViews = [], accessors = [];
  let byteLength = 0;
  const add = (array, componentType, type, count, extras = {}) => {
    const padding = (4 - byteLength % 4) % 4;
    if (padding) { chunks.push(new Uint8Array(padding)); byteLength += padding; }
    bufferViews.push({buffer: 0, byteOffset: byteLength, byteLength: array.byteLength});
    chunks.push(new Uint8Array(array.buffer, array.byteOffset, array.byteLength));
    byteLength += array.byteLength;
    accessors.push({bufferView: bufferViews.length - 1, componentType, type, count, ...extras});
    return accessors.length - 1;
  };
  const positions = add(new Float32Array([0,0,0, -.4,1.5,0, .3,2,0]), 5126, "VEC3", 3,
    {min: [-.4,0,0], max: [.3,2,0]});
  const joints = add(new Uint16Array([0,0,0,0, 1,0,0,0, boneCount-1,0,0,0]), 5123, "VEC4", 3);
  const weights = add(new Float32Array([1,0,0,0, 1,0,0,0, 1,0,0,0]), 5126, "VEC4", 3);
  const inverse = new Float32Array(boneCount * 16);
  for (let i = 0; i < boneCount; i++) {
    inverse.set([1,0,0,0, 0,1,0,0, 0,0,1,0, 0,-(1+i/(boneCount-1)),0,1], i*16);
  }
  const inverseBindMatrices = add(inverse, 5126, "MAT4", boneCount);
  const nodes = [{name: "Character", children: [1, boneCount + 1]}];
  for (let i = 0; i < boneCount; i++) {
    nodes.push({name: `Bone_${i}`, translation: [0, i ? 1/(boneCount-1) : 1, 0],
      ...(i < boneCount - 1 ? {children: [i + 2]} : {})});
  }
  nodes.push({name: "Skin", mesh: 0, skin: 0});
  const gltf = {
    asset: {version: "2.0", generator: "RLsim test fixture"}, scene: 0, scenes: [{nodes: [0]}], nodes,
    buffers: [{byteLength}], bufferViews, accessors,
    meshes: [{primitives: [{attributes: {POSITION: positions, JOINTS_0: joints, WEIGHTS_0: weights}, material: 0}]}],
    skins: [{joints: Array.from({length: boneCount}, (_, i) => i+1), skeleton: 1, inverseBindMatrices}],
    materials: [{doubleSided: true, pbrMetallicRoughness: {baseColorFactor: [.2,.5,.8,1], metallicFactor: 0, roughnessFactor: .8}}],
  };
  let json = encoder.encode(JSON.stringify(gltf));
  if (json.length % 4) {
    const padded = new Uint8Array(json.length + 4 - json.length % 4).fill(32);
    padded.set(json); json = padded;
  }
  const binary = new Uint8Array(Math.ceil(byteLength / 4) * 4);
  let offset = 0;
  for (const chunk of chunks) { binary.set(chunk, offset); offset += chunk.length; }
  const output = new Uint8Array(12 + 8 + json.length + 8 + binary.length), view = new DataView(output.buffer);
  view.setUint32(0, 0x46546c67, true); view.setUint32(4, 2, true); view.setUint32(8, output.length, true);
  view.setUint32(12, json.length, true); view.setUint32(16, 0x4e4f534a, true); output.set(json, 20);
  const binaryOffset = 20 + json.length;
  view.setUint32(binaryOffset, binary.length, true); view.setUint32(binaryOffset + 4, 0x004e4942, true);
  output.set(binary, binaryOffset + 8);
  return output;
}

export const tinyOBJ = encoder.encode(`mtllib cube.mtl
o Cube
v -0.5 0 -0.5
v 0.5 0 -0.5
v 0.5 1 -0.5
v -0.5 1 -0.5
v -0.5 0 0.5
v 0.5 0 0.5
v 0.5 1 0.5
v -0.5 1 0.5
usemtl blue
f 1 4 3 2
f 5 6 7 8
f 1 2 6 5
f 4 8 7 3
f 1 5 8 4
f 2 3 7 6
`);
export const tinyMTL = encoder.encode("newmtl blue\nKd 0.2 0.4 0.8\n");
export const tinySTL = encoder.encode(`solid triangle
facet normal 0 0 1
outer loop
vertex 0 0 0
vertex 1 0 0
vertex 0 1 0
endloop
endfacet
endsolid triangle
`);
export const tinyFBX = encoder.encode(`; FBX 7.4.0 project file
FBXHeaderExtension:  {
\tFBXHeaderVersion: 1003
\tFBXVersion: 7400
}
GlobalSettings:  {
\tVersion: 1000
\tProperties70:  {
\t\tP: "UpAxis", "int", "Integer", "",1
\t\tP: "UnitScaleFactor", "double", "Number", "",100
\t}
}
Objects:  {
\tGeometry: 1, "Geometry::Triangle", "Mesh" {
\t\tVertices: *9 {
\t\t\ta: 0,0,0,1,0,0,0,1,0
\t\t}
\t\tPolygonVertexIndex: *3 {
\t\t\ta: 0,1,-3
\t\t}
\t}
\tModel: 2, "Model::Triangle", "Mesh" {
\t\tVersion: 232
\t\tProperties70:  {
\t\t\tP: "Lcl Translation", "Lcl Translation", "", "A",0,0,0
\t\t\tP: "Lcl Rotation", "Lcl Rotation", "", "A",0,0,0
\t\t\tP: "Lcl Scaling", "Lcl Scaling", "", "A",1,1,1
\t\t}
\t\tShading: T
\t\tCulling: "CullingOff"
\t}
}
Connections:  {
\tC: "OO",1,2
\tC: "OO",2,0
}
`);

/** Original humanoid fixture with IK-style root feet, leaf shins and a posed
 * node state different from its bind pose. No downloaded asset is required. */
export function helperHumanoidGLB() {
  const defs=[
    ['Bone',null,[0,0,0]],['Pelvis',0,[0,1,0]],
    ['UpperLegL',1,[.15,1,0]],['LowerLegL',2,[.15,.55,.02]],['FootL',0,[.15,.08,0]],
    ['UpperLegR',1,[-.15,1,0]],['LowerLegR',5,[-.15,.55,.02]],['FootR',0,[-.15,.08,0]],
    ['Spine',1,[0,1.3,0]],['Head',8,[0,1.6,0]],['PoleTargetL',0,[.2,.5,.8]],
  ];
  const points=[],indices=[],weights=[],bounds={min:[Infinity,Infinity,Infinity],max:[-Infinity,-Infinity,-Infinity]};
  defs.forEach(([name,,position],i)=>{
    if(i===0||name.startsWith('Pole'))return;
    const center=[...position];if(name.startsWith('UpperLeg'))center[1]-=.18;if(name.startsWith('LowerLeg'))center[1]-=.2;
    if(name.startsWith('Foot'))center[2]+=.04;
    const size=name==='Head'?[.12,.12,.12]:name.startsWith('Foot')?[.06,.04,.12]:[.06,.1,.07];
    for(let corner=0;corner<8;corner++){
      for(let axis=0;axis<3;axis++){const p=center[axis]+((corner>>axis)&1?1:-1)*size[axis];points.push(p);bounds.min[axis]=Math.min(bounds.min[axis],p);bounds.max[axis]=Math.max(bounds.max[axis],p);}
      indices.push(i,0,0,0);weights.push(1,0,0,0);
    }
  });
  const chunks=[],views=[],accessors=[];let length=0;
  function add(array,componentType,type,count,extra={}){
    while(length%4){chunks.push(new Uint8Array(1));length++;}
    views.push({buffer:0,byteOffset:length,byteLength:array.byteLength});chunks.push(new Uint8Array(array.buffer));length+=array.byteLength;
    accessors.push({bufferView:views.length-1,componentType,type,count,...extra});return accessors.length-1;
  }
  const position=add(new Float32Array(points),5126,'VEC3',points.length/3,bounds);
  const joints=add(new Uint16Array(indices),5123,'VEC4',indices.length/4),skinWeights=add(new Float32Array(weights),5126,'VEC4',weights.length/4);
  const inverse=new Float32Array(defs.length*16);
  defs.forEach(([, ,p],i)=>inverse.set([1,0,0,0,0,1,0,0,0,0,1,0,-p[0],-p[1],-p[2],1],i*16));
  const ibm=add(inverse,5126,'MAT4',defs.length);
  const nodes=[{name:'Outer',translation:[.4,.1,-.2],scale:[.8,.8,.8],children:[1,defs.length+1]}];
  defs.forEach(([name,parent,p],i)=>{
    const translation=parent===null?p:p.map((x,axis)=>x-defs[parent][2][axis]);
    const children=defs.flatMap(([,parent],child)=>parent===i?[child+1]:[]);
    nodes.push({name,translation, ...(children.length?{children}:{})});
  });
  nodes[5].translation=[.15,.12,.4];nodes[8].translation=[-.15,.15,-.3];nodes[2].translation=[0,.85,0];
  nodes.push({name:'Skin',mesh:0,skin:0});
  const gltf={asset:{version:'2.0',generator:'RLsim original helper-rig fixture'},scene:0,scenes:[{nodes:[0]}],nodes,
    buffers:[{byteLength:length}],bufferViews:views,accessors,
    meshes:[{primitives:[{mode:0,attributes:{POSITION:position,JOINTS_0:joints,WEIGHTS_0:skinWeights}}]}],
    skins:[{joints:defs.map((_,i)=>i+1),skeleton:1,inverseBindMatrices:ibm}]};
  // TRIANGLES avoids Points, whose loader does not create SkinnedMesh.
  gltf.meshes[0].primitives[0].mode=4;
  let json=encoder.encode(JSON.stringify(gltf));const padded=new Uint8Array(Math.ceil(json.length/4)*4).fill(32);padded.set(json);json=padded;
  const bin=new Uint8Array(Math.ceil(length/4)*4);let at=0;for(const c of chunks){bin.set(c,at);at+=c.length;}
  const result=new Uint8Array(28+json.length+bin.length),view=new DataView(result.buffer);
  view.setUint32(0,0x46546c67,true);view.setUint32(4,2,true);view.setUint32(8,result.length,true);
  view.setUint32(12,json.length,true);view.setUint32(16,0x4e4f534a,true);result.set(json,20);
  at=20+json.length;view.setUint32(at,bin.length,true);view.setUint32(at+4,0x004e4942,true);result.set(bin,at+8);
  return result;
}
