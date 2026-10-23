import copy
import json
import subprocess
from pathlib import Path

import numpy as np
import pytest

from morpholab.rig_fidelity import accessor, preserve_original, read_glb, world_matrices, write_glb, review_skeleton
from test_rig_prepare import glb_bytes


def inputs(tmp_path):
    source, prediction, output = [tmp_path / name for name in ['source.glb', 'prediction.glb', 'output.glb']]
    source.write_bytes(glb_bytes(skinned=False)); prediction.write_bytes(glb_bytes())
    return source, prediction, output


def test_original_geometry_materials_and_buffer_are_preserved(tmp_path):
    source, prediction, output = inputs(tmp_path)
    doc, binary = read_glb(source)
    doc['materials'] = [{'name': 'Original', 'pbrMetallicRoughness': {'baseColorFactor': [.2, .4, .6, 1]}}]
    doc['meshes'][0]['primitives'][0]['material'] = 0
    write_glb(source, doc, binary)
    before = source.read_bytes()
    report = preserve_original(source, prediction, output)
    result, result_binary = read_glb(output)
    assert source.read_bytes() == before
    assert result_binary[:len(binary)] == binary
    assert result['materials'] == doc['materials']
    assert result['meshes'][0] == doc['meshes'][0]
    mesh = result['meshes'][result['nodes'][0]['mesh']]
    attrs = mesh['primitives'][0]['attributes']
    assert attrs['POSITION'] == 0
    weights = accessor(result, result_binary, attrs['WEIGHTS_0'])
    assert np.allclose(weights.sum(axis=1), 1)
    assert report['original_geometry_preserved'] and report['max_surface_error'] == 0
    assert result['extras']['rlsim_rig_fidelity'] == report


def test_rest_binding_preserves_transformed_mesh(tmp_path):
    source, prediction, output = inputs(tmp_path)
    for path in [source, prediction]:
        doc, raw = read_glb(path)
        doc['nodes'][0]['translation'] = [3, 2, 1]
        doc['nodes'][0]['scale'] = [2, 2, 2]
        write_glb(path, doc, raw)
    preserve_original(source, prediction, output)
    doc, raw = read_glb(output)
    world, _ = world_matrices(doc)
    node = doc['nodes'][0]; skin = doc['skins'][node['skin']]
    inverse = accessor(doc, raw, skin['inverseBindMatrices']).reshape(-1, 4, 4).transpose(0, 2, 1)
    for joint, matrix in zip(skin['joints'], inverse):
        np.testing.assert_allclose(world[joint] @ matrix, world[0], atol=1e-6)
    # Exercise the actual Three.js importer and its rest-pose recovery too.
    script="""
import {readFileSync} from 'node:fs';
import {GLTFLoader} from 'three/addons/loaders/GLTFLoader.js';
import {Vector3} from 'three';
import {recoverRestPose} from './morpholab/static/rig-inference.mjs';
const bytes=readFileSync(process.argv[1]);const {scene}=await new GLTFLoader().parseAsync(bytes.buffer.slice(bytes.byteOffset,bytes.byteOffset+bytes.byteLength),'');
recoverRestPose(scene);scene.updateMatrixWorld(true);const points=[];
scene.traverse(mesh=>{if(mesh.isSkinnedMesh){for(let i=0;i<mesh.geometry.attributes.position.count;i++)points.push(mesh.getVertexPosition(i,new Vector3()).applyMatrix4(mesh.matrixWorld).toArray());}});
console.log(JSON.stringify(points));
"""
    result=subprocess.run(['node','--input-type=module','-e',script,str(output)],cwd=Path(__file__).resolve().parents[1],capture_output=True,text=True,check=True)
    np.testing.assert_allclose(json.loads(result.stdout),[[3,2,1],[5,2,1],[3,4,1]],atol=1e-5)


def test_shape_change_is_rejected_without_creating_output(tmp_path):
    source, prediction, output = inputs(tmp_path)
    doc, raw = read_glb(prediction)
    doc['nodes'][0]['translation'] = [.1, 0, 0]
    write_glb(prediction, doc, raw)
    with pytest.raises(ValueError, match='moved or changed shape'):
        preserve_original(source, prediction, output)
    assert not output.exists()


def test_existing_skin_is_not_replaced_implicitly(tmp_path):
    source, prediction, output = inputs(tmp_path)
    source.write_bytes(glb_bytes())
    with pytest.raises(ValueError, match='already has a skeleton'):
        preserve_original(source, prediction, output)


def test_invalid_weights_are_rejected(tmp_path):
    source, prediction, output = inputs(tmp_path)
    doc, raw = read_glb(prediction)
    raw = bytearray(raw)
    offset = doc['bufferViews'][2]['byteOffset']
    raw[offset:offset+48] = bytes(48)
    write_glb(prediction, doc, raw)
    with pytest.raises(ValueError, match='invalid skin weights'):
        preserve_original(source, prediction, output)


def test_symmetric_surface_flags_missing_joint_counterpart():
    points=np.array([[x,y,z] for x in [-1,1] for y in [0,1] for z in [-1,0,1]],dtype=float)
    joints=np.array([[0,1,0],[-1,.5,-1],[-1,.5,0],[-1,.5,1],[1,.5,0],[1,.5,1]])
    report=review_skeleton(points,joints,[str(i) for i in range(len(joints))])
    assert any(f['axis']=='X' and '1' in f['unmatched_joints'] for f in report['findings'])
