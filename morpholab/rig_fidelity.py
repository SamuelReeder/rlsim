"""Attach predicted skinning to the original GLB without re-exporting its art.

Only skeleton nodes, skins and joint/weight accessors are added. Existing mesh
positions, indices, normals, UVs, textures, materials and transforms are retained.
The predicted export must match the original rest surface before weights transfer.
"""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import struct

import numpy as np
from scipy.spatial import cKDTree


def read_glb(path):
    raw = Path(path).read_bytes()
    if len(raw) < 20 or struct.unpack_from('<4sII', raw) != (b'glTF', 2, len(raw)):
        raise ValueError('Expected a self-contained GLB 2.0 file.')
    document, binary, offset = None, b'', 12
    while offset < len(raw):
        size, kind = struct.unpack_from('<I4s', raw, offset)
        offset += 8
        if offset + size > len(raw):
            raise ValueError('Truncated GLB chunk.')
        chunk = raw[offset:offset + size]
        if kind == b'JSON':
            document = json.loads(chunk)
        elif kind == b'BIN\0':
            binary = chunk
        offset += size
    if document is None or any(b.get('uri') for b in document.get('buffers', [])):
        raise ValueError('Embed all model resources in one GLB before automatic rigging.')
    if len(document.get('buffers', [])) != 1:
        raise ValueError('Automatic rigging requires one embedded geometry buffer.')
    if any(image.get('uri') and not image['uri'].startswith('data:') for image in document.get('images', [])):
        raise ValueError('Embed textures in the GLB before automatic rigging.')
    return document, binary


def write_glb(path, doc, binary):
    doc['buffers'] = [{'byteLength': len(binary)}]
    payload = json.dumps(doc, separators=(',', ':'), allow_nan=False).encode()
    payload += b' ' * (-len(payload) % 4)
    binary = bytes(binary) + b'\0' * (-len(binary) % 4)
    Path(path).write_bytes(struct.pack('<4sII', b'glTF', 2, 28 + len(payload) + len(binary))
                          + struct.pack('<I4s', len(payload), b'JSON') + payload
                          + struct.pack('<I4s', len(binary), b'BIN\0') + binary)


def accessor(doc, binary, index):
    a = doc['accessors'][index]
    if 'sparse' in a or 'bufferView' not in a:
        raise ValueError('Expand sparse/compressed geometry before automatic rigging.')
    types = {5120: 'i1', 5121: 'u1', 5122: '<i2', 5123: '<u2', 5125: '<u4', 5126: '<f4'}
    dtype = np.dtype(types[a['componentType']])
    width = {'SCALAR': 1, 'VEC2': 2, 'VEC3': 3, 'VEC4': 4, 'MAT4': 16}[a['type']]
    view = doc['bufferViews'][a['bufferView']]
    if view.get('buffer', 0) != 0:
        raise ValueError('Only embedded GLB buffers are supported.')
    stride = view.get('byteStride', width * dtype.itemsize)
    offset = view.get('byteOffset', 0) + a.get('byteOffset', 0)
    count = a['count']
    if not 0 < count <= 2_000_000 or stride < width * dtype.itemsize:
        raise ValueError('Invalid or oversized geometry accessor.')
    end = a.get('byteOffset', 0) + (count - 1) * stride + width * dtype.itemsize
    if end > view['byteLength'] or offset + (count - 1) * stride + width * dtype.itemsize > len(binary):
        raise ValueError('Geometry accessor exceeds its buffer.')
    value = np.ndarray((count, width), dtype=dtype, buffer=binary, offset=offset,
                       strides=(stride, dtype.itemsize)).copy()
    if a.get('normalized') and dtype.kind in 'ui':
        value = np.maximum(value.astype(float) / np.iinfo(dtype).max, -1)
    if not np.isfinite(value).all():
        raise ValueError('Geometry contains non-finite values.')
    return value


def world_matrices(doc):
    nodes = doc['nodes']
    parents = {}
    for i, node in enumerate(nodes):
        for child in node.get('children', []):
            if child in parents or not 0 <= child < len(nodes):
                raise ValueError('Skeleton must be a tree with valid nodes.')
            parents[child] = i
    world, active = {}, set()
    def visit(i):
        if i in active:
            raise ValueError('Skeleton contains a cycle.')
        if i in world:
            return world[i]
        active.add(i)
        node = nodes[i]
        if 'matrix' in node:
            local = np.array(node['matrix'], dtype=float).reshape(4, 4).T
        else:
            x, y, z, w = node.get('rotation', [0, 0, 0, 1])
            rotation = np.array([[1-2*y*y-2*z*z, 2*x*y-2*z*w, 2*x*z+2*y*w],
                                 [2*x*y+2*z*w, 1-2*x*x-2*z*z, 2*y*z-2*x*w],
                                 [2*x*z-2*y*w, 2*y*z+2*x*w, 1-2*x*x-2*y*y]])
            local = np.eye(4)
            local[:3, :3] = rotation @ np.diag(node.get('scale', [1, 1, 1]))
            local[:3, 3] = node.get('translation', [0, 0, 0])
        world[i] = visit(parents[i]) @ local if i in parents else local
        active.remove(i)
        return world[i]
    for i in range(len(nodes)):
        visit(i)
    return world, parents


def mesh_points(doc, binary, world):
    for i, node in enumerate(doc['nodes']):
        if 'mesh' not in node:
            continue
        for primitive in doc['meshes'][node['mesh']]['primitives']:
            if primitive.get('extensions') or primitive.get('targets'):
                raise ValueError('Automatic rigging currently requires uncompressed meshes without morph targets.')
            positions = accessor(doc, binary, primitive['attributes']['POSITION'])
            points = positions @ world[i][:3, :3].T + world[i][:3, 3]
            yield i, primitive, points


def review_skeleton(points, joint_positions, joint_names):
    """Flag missing counterparts only when the source surface is near symmetric.

    This is a review heuristic, not anatomical classification or a limb counter.
    """
    extent = np.ptp(points, axis=0)
    diagonal = float(np.linalg.norm(extent))
    center = (points.min(axis=0) + points.max(axis=0)) / 2
    samples = points[::max(1, len(points)//20000)]
    surface_tree, bone_tree = cKDTree(points), cKDTree(joint_positions)
    findings=[]
    for axis in range(3):
        reflected=samples.copy(); reflected[:,axis]=2*center[axis]-reflected[:,axis]
        distances,_=surface_tree.query(reflected)
        if np.quantile(distances,.99)>diagonal*.008:
            continue
        mirrored=joint_positions.copy();mirrored[:,axis]=2*center[axis]-mirrored[:,axis]
        bone_distances,_=bone_tree.query(mirrored)
        missing=[joint_names[i] for i,distance in enumerate(bone_distances) if distance>diagonal*.06]
        if missing:
            findings.append({'axis':'XYZ'[axis],'unmatched_joints':missing,
                             'message':f'The mesh is nearly symmetric across {"XYZ"[axis]}, but some predicted joints have no nearby counterpart. A limb branch may be missing; inspect or regenerate the rig.'})
    return {'method':'near-symmetric surface / mirrored joint proximity','findings':findings,
            'scope':'Flags some missing branches. Does not verify limb count, anatomy or intended joint limits.'}


def preserve_original(source, prediction, destination):
    original, raw = read_glb(source)
    generated, generated_raw = read_glb(prediction)
    if any('skin' in n for n in original['nodes']):
        raise ValueError('This model already has a skeleton. Import it with Rig: Auto to preserve its existing skinning.')
    # A single predicted skeleton may be shared by multiple mesh primitives.
    used_skins = {n['skin'] for n in generated['nodes'] if 'mesh' in n and 'skin' in n}
    if len(used_skins) != 1:
        raise ValueError('SkinTokens must return one coherent skeleton for this model.')
    skin_index = next(iter(used_skins))
    skin = generated['skins'][skin_index]
    joints = skin['joints']
    if not 2 <= len(joints) <= 128:
        raise ValueError('The predicted skeleton needs 2–128 joints. Simplify the model and retry.')
    source_world, _ = world_matrices(original)
    predicted_world, parents = world_matrices(generated)
    points, joint_indices, weights = [], [], []
    for node, primitive, positions in mesh_points(generated, generated_raw, predicted_world):
        if generated['nodes'][node].get('skin') != skin_index:
            continue
        attrs = primitive['attributes']
        j = accessor(generated, generated_raw, attrs['JOINTS_0']).astype(int)
        w = accessor(generated, generated_raw, attrs['WEIGHTS_0']).astype(float)
        if j.shape != w.shape or len(j) != len(positions) or j.min() < 0 or j.max() >= len(joints) or w.min() < 0 or (w.sum(axis=1) <= 0).any():
            raise ValueError('SkinTokens returned invalid skin weights.')
        points.append(positions); joint_indices.append(j); weights.append(w / w.sum(axis=1)[:, None])
    if not points:
        raise ValueError('SkinTokens returned no weighted mesh.')
    points = np.concatenate(points)
    joint_indices, weights = np.concatenate(joint_indices), np.concatenate(weights)
    tree = cKDTree(points)
    source_meshes = list(mesh_points(original, raw, source_world))
    all_original = np.concatenate([row[2] for row in source_meshes])
    diagonal = float(np.linalg.norm(np.ptp(all_original, axis=0)))
    tolerance = max(diagonal * 0.0001, 1e-6)
    distances, _ = tree.query(all_original)
    reverse, _ = cKDTree(all_original).query(points)
    max_error = float(max(distances.max(), reverse.max()))
    if max_error > tolerance:
        raise ValueError(f'Predicted mesh moved or changed shape ({max_error:.6g} units; tolerance {tolerance:.6g}). Original preserved; no rig applied.')
    doc, binary = copy.deepcopy(original), bytearray(raw)
    def add(values, kind, component=5126):
        values = np.asarray(values, dtype='<u2' if component == 5123 else '<f4')
        binary.extend(b'\0' * (-len(binary) % 4))
        view = len(doc.setdefault('bufferViews', []))
        doc['bufferViews'].append({'buffer': 0, 'byteOffset': len(binary), 'byteLength': values.nbytes})
        binary.extend(values.tobytes())
        index = len(doc.setdefault('accessors', []))
        doc['accessors'].append({'bufferView': view, 'componentType': component, 'count': len(values), 'type': kind})
        return index
    mapping = {old: len(doc['nodes']) + i for i, old in enumerate(joints)}
    joint_parents = {}
    for old in joints:
        parent = parents.get(old)
        while parent is not None and parent not in mapping:
            parent = parents.get(parent)
        joint_parents[old] = parent
        local = predicted_world[old] if parent is None else np.linalg.inv(predicted_world[parent]) @ predicted_world[old]
        doc['nodes'].append({'name': generated['nodes'][old].get('name', f'Bone_{old}'),
                             'matrix': local.T.reshape(-1).tolist(),
                             'extras': {'rlsim_rest_pose': True},
                             'children': [mapping[j] for j in joints if parents.get(j) == old]})
    # Link through any non-joint intermediary nodes that were flattened above.
    for old in joints:
        doc['nodes'][mapping[old]]['children'] = [mapping[j] for j in joints if joint_parents[j] == old]
    roots = [mapping[j] for j in joints if joint_parents[j] is None]
    for scene in doc.get('scenes', []):
        scene.setdefault('nodes', []).extend(roots)
    new_meshes = {}
    for node_index, primitive, positions in source_meshes:
        if node_index not in new_meshes:
            node = doc['nodes'][node_index]
            mesh = copy.deepcopy(doc['meshes'][node['mesh']])
            new_meshes[node_index] = mesh
            node['mesh'] = len(doc['meshes']); doc['meshes'].append(mesh)
            ibm = np.stack([(np.linalg.inv(predicted_world[j]) @ source_world[node_index]).T.reshape(-1) for j in joints])
            node['skin'] = len(doc.setdefault('skins', []))
            doc['skins'].append({'joints': list(mapping.values()), 'inverseBindMatrices': add(ibm, 'MAT4')})
        original_primitives = original['meshes'][original['nodes'][node_index]['mesh']]['primitives']
        primitive_index = next(i for i, p in enumerate(original_primitives) if p is primitive)
        target = new_meshes[node_index]['primitives'][primitive_index]
        _, nearest = tree.query(positions)
        target['attributes']['JOINTS_0'] = add(joint_indices[nearest], 'VEC4', 5123)
        target['attributes']['WEIGHTS_0'] = add(weights[nearest], 'VEC4')
    report = {'format': 'rlsim-rig-fidelity-v1', 'source_sha256': hashlib.sha256(Path(source).read_bytes()).hexdigest(),
              'prediction_sha256': hashlib.sha256(Path(prediction).read_bytes()).hexdigest(),
              'original_geometry_preserved': True, 'original_materials_preserved': True,
              'original_vertices': len(all_original), 'joints': len(joints),
              'max_surface_error': max_error, 'surface_tolerance': tolerance,
              'weight_transfer': 'nearest coincident rest vertex',
              'skeleton_review': review_skeleton(all_original, np.stack([predicted_world[j][:3,3] for j in joints]),
                                                  [generated['nodes'][j].get('name',str(j)) for j in joints]),
              'note': 'Preserves rest geometry and materials; predicted deformation and physics still need review.'}
    doc.setdefault('extras', {})['rlsim_rig_fidelity'] = report
    write_glb(destination, doc, binary)
    return report
