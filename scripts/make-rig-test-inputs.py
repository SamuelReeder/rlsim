"""Original, reproducible unrigged shapes for the automatic-rigging audit.

Run in the separate SkinTokens environment (trimesh). No skeleton is supplied.
"""
from pathlib import Path
import argparse
import numpy as np
import trimesh


def ellipsoid(center, radii, color):
    mesh = trimesh.creation.icosphere(subdivisions=3)
    mesh.vertices = mesh.vertices * radii + center
    mesh.visual.face_colors = color
    return mesh


def limb(a, b, radius, color):
    a, b = np.array(a), np.array(b)
    mesh = trimesh.creation.cylinder(radius=radius, height=np.linalg.norm(b-a), sections=24)
    transform = trimesh.geometry.align_vectors([0,0,1], b-a)
    transform[:3,3] = (a+b)/2
    mesh.apply_transform(transform)
    mesh.visual.face_colors = color
    return [mesh, ellipsoid(a,[radius]*3,color), ellipsoid(b,[radius]*3,color)]


def build(output):
    output.mkdir(parents=True, exist_ok=True)
    blue, skin, dark = [65,125,180,255], [190,145,115,255], [45,52,65,255]
    human = [ellipsoid([0,1.2,0],[.22,.3,.13],blue), ellipsoid([0,1.73,0],[.12,.16,.12],skin)]
    human += limb([0,1.4,0],[0,1.6,0],.065,skin)
    for sign in [-1,1]:
        human += limb([sign*.13,.98,0],[sign*.15,.55,.025],.085,dark)
        human += limb([sign*.15,.55,.025],[sign*.15,.13,0],.062,dark)
        human += [ellipsoid([sign*.15,.07,.07],[.075,.07,.15],dark)]
        human += limb([sign*.18,1.42,0],[sign*.5,1.39,0],.063,blue)
        human += limb([sign*.5,1.39,0],[sign*.78,1.37,0],.048,skin)
        human += [ellipsoid([sign*.84,1.37,0],[.08,.045,.065],skin)]
    trimesh.util.concatenate(human).export(output/'humanoid.glb')
    green = [65,135,105,255]
    crawler=[ellipsoid([0,.48,0],[.25,.15,.45],green),ellipsoid([0,.5,.48],[.14,.12,.15],dark)]
    for z in [-.3,0,.3]:
        for sign in [-1,1]:
            crawler += limb([sign*.18,.48,z],[sign*.5,.4,z*1.35],.045,green)
            crawler += limb([sign*.5,.4,z*1.35],[sign*.72,.05,z*1.6],.032,dark)
    trimesh.util.concatenate(crawler).export(output/'hexapod.glb')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('output',type=Path);args=parser.parse_args();build(args.output)
