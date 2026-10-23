"""Opt-in WebGL renderer and local-texture integration checks.

RLSIM_SCENE_TEST=1 .venv/bin/pytest -q tests/test_scene_visual.py
Run npm run vendor:browser first. Uses an isolated static server and Chromium.
"""
import base64
import functools
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import struct
import subprocess
import threading
import zlib

import pytest

ROOT = Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.skipif(not os.getenv("RLSIM_SCENE_TEST"), reason="Opt-in WebGL scene test")


def one_pixel_png():
    def chunk(kind, data):
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", 1, 1, 8, 6, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(bytes([0, 20, 180, 60, 255]))) + chunk(b"IEND", b""))


def test_local_textures_skin_and_cached_physics_geometry():
    pw = pytest.importorskip("playwright.sync_api")
    fixtures = json.loads(subprocess.check_output([
        "node", "--input-type=module", "-e",
        "import {tinyRigGLB,tinyOBJ} from './tests/scene-fixtures.mjs';"
        "console.log(JSON.stringify({glb:Buffer.from(tinyRigGLB()).toString('base64'),"
        "obj:Buffer.from(tinyOBJ).toString('base64')}));",
    ], cwd=ROOT, text=True))
    fixtures["texture"] = base64.b64encode(one_pixel_png()).decode()
    fixtures["tga"] = base64.b64encode(bytes([0,0,2,0,0,0,0,0,0,0,0,0,1,0,1,0,24,32,60,180,20])).decode()

    class QuietHandler(SimpleHTTPRequestHandler):
        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), functools.partial(
        QuietHandler, directory=str(ROOT / "morpholab/static")))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        with pw.sync_playwright() as runtime:
            browser = runtime.chromium.launch(headless=True)
            page = browser.new_page(viewport={"width": 900, "height": 650})
            page.goto(f"http://127.0.0.1:{server.server_port}/vendor/NOTICE.txt")
            page.set_content('<style>body{margin:0}canvas{display:block;width:900px;height:650px}</style><canvas></canvas>')
            page.add_script_tag(url=f"http://127.0.0.1:{server.server_port}/vendor/scene.js")
            errors, external_requests = [], []
            page.on("pageerror", lambda error: errors.append(str(error)))
            page.on("request", lambda request: external_requests.append(request.url) if request.url.startswith("http") else None)
            result = page.evaluate("""async fixtures => {
              const {importVisualFiles,SceneView,decodeAsset,loadVisualScene,disposeScene} = RLsimScene;
              const encoder = new TextEncoder();
              const files = [
                {path:"model/cube.obj",data:decodeAsset(fixtures.obj)},
                {path:"model/cube.mtl",data:encoder.encode("newmtl blue\\nKd 1 1 1\\nmap_Kd ../textures/green.png\\n")},
                {path:"textures/green.png",data:decodeAsset(fixtures.texture)},
              ];
              const body = await importVisualFiles(files);
              const view = new SceneView(document.querySelector("canvas"));
              await view.setBody(body); view.reset(body); view.draw();
              let imageWidth=0;
              view.visual.scene.traverse(object => { if(object.isMesh) imageWidth=object.material.map?.image?.width || imageWidth; });
              const tga=await loadVisualScene([
                {path:"cube.obj",data:decodeAsset(fixtures.obj)},
                {path:"cube.mtl",data:encoder.encode("newmtl blue\\nmap_Kd green.tga\\n")},
                {path:"green.tga",data:decodeAsset(fixtures.tga)},
              ]);
              let tgaWidth=0;
              tga.scene.traverse(object => {if(object.isMesh)tgaWidth=object.material.map?.image?.width||tgaWidth;});
              disposeScene(tga.scene);
              const glb = await importVisualFiles([{path:"character.glb",data:decodeAsset(fixtures.glb)}]);
              const loading = view.setBody(glb), loadingVersion = view.bodyVersion;
              await Promise.all([loading, view.setBody(structuredClone(glb))]);
              const sharedLoad = view.bodyVersion === loadingVersion;
              const visualBefore = view.visual;
              await view.setBody(structuredClone(glb));
              const retainedVisual = view.visual === visualBefore;
              const bones = view.visual.bones.length;
              view.setMode("training");
              const movedRoot=glb.visual.bind_root.slice(12,15).map((v,i)=>v+[8,3,0][i]);
              view.setFrame({time:1,geoms:[],bodies:[{name:glb.root_body,pos:movedRoot,quat:[1,0,0,0]}]});
              view.reset(glb);
              const movedReset=view.camera.target.every((v,i)=>Math.abs(v-
                ((glb.visual.bounds.min[i]+glb.visual.bounds.max[i])/2+[8,3,0][i]))<1e-8);
              view.setMode("preview");
              // Exercise the first-frame metadata cache and each geometry type.
              const mat=[1,0,0,0,1,0,0,0,1], rgba=[.3,.5,.7,1];
              const mesh={vertices:[0,0,0,1,0,0,0,1,0],faces:[0,1,2]};
              const geoms=["box","capsule","cylinder","sphere","ellipsoid","mesh","hfield","plane"].map((type,i)=>({
                type,pos:[i*.4,0,.5],mat,rgba,size:[.15,.2,.25],...([5,6].includes(i)?{mesh:"shared"}:{})
              }));
              view.setFrame({geoms,bodies:[],meshes:{shared:mesh}});
              const cached=view.geomObjects.get(5).object.geometry;
              view.setFrame({geoms,bodies:[]});
              const retained=view.geomObjects.get(5).object.geometry===cached && view.meshes.get("shared")===mesh;
              view.showCollision=true;view.draw();
              const overlay=view.physics.visible && view.geomObjects.get(0).object.material.wireframe;
              // Reproduce frame-before-setBody UI ordering: ordinary physics stays visible.
              await view.setBody({kind:"mjcf",name:"Geometry test"});
              view.setFrame({geoms,bodies:[],meshes:{shared:mesh}});
              const count=view.geomObjects.size;
              view.setFrame({geoms,bodies:[],task:{preset:"target",target:[2,3,1]}});
              const targetShown=view.targetMarker.visible && view.targetMarker.position.toArray().join(",")==="2,3,1";
              view.setFrame({geoms,bodies:[],task:{preset:"velocity",target:[2,3,1]}});
              const velocityHidden=!view.targetMarker.visible;
              view.setFrame({geoms,bodies:[],target:[-1,0,.8]});
              const planarTarget=view.targetMarker.visible && view.targetMarker.position.x===-1;
              view.setFrame({geoms,bodies:[]});
              const targetHidden=!view.targetMarker.visible;
              view.reset({kind:"mjcf"});view.draw();
              const anchor=[...view.camera.target];
              view.setMode("training");
              view.setFrame({time:.8,geoms:[{type:"sphere",pos:[4,2,.1],mat,rgba,size:[.2,.2,.2]}],torso_position:[4,2,.1]});
              view.draw(0);
              view.setFrame({time:0,geoms:[{type:"sphere",pos:[-2,-1,2],mat,rgba,size:[.2,.2,.2]}],torso_position:[-2,-1,2]});
              view.draw(100);
              const trainingCameraStable=view.camera.target.every((v,i)=>v===anchor[i]);
              const unfilteredPhysics=view.geomObjects.get(0).object.position.toArray().join(",")==="-2,-1,2";
              view.setMode("inference");
              view.setFrame({time:0,geoms:[],torso_position:[0,0,1]});view.draw(0);
              view.setFrame({time:.02,geoms:[],torso_position:[1,2,.1]});view.draw(100);
              const smoothFollow=view.camera.target[0]>0 && view.camera.target[0]<1 && view.camera.target[2]===anchor[2];
              const gridSeparated=view.grid.position.z>0 && !view.grid.material.depthWrite && view.grid.renderOrder===1;
              // Long unsplit lines disappeared in the foreground on software WebGL.
              view.setFrame({geoms:[
                {type:"plane",pos:[0,0,0],mat,size:[5,5,1],rgba:[.15,.2,.22,1]},
                {type:"box",pos:[0,0,.5],mat,size:[.5,.2,.3],rgba:[.5,.65,.7,1]}
              ]});view.reset({kind:"mjcf"});
              const capture=()=>{
                view.dirty=true;view.draw();
                const copy=document.createElement("canvas");copy.width=view.canvas.width;copy.height=view.canvas.height;
                const context=copy.getContext("2d");context.drawImage(view.canvas,0,0);
                return context.getImageData(0,0,copy.width,copy.height).data;
              };
              const withGrid=capture();view.grid.visible=false;const withoutGrid=capture();view.grid.visible=true;
              let foregroundGridPixels=0;
              for(let i=Math.floor(withGrid.length/2/4)*4;i<withGrid.length;i+=4) {
                if(Math.abs(withGrid[i]-withoutGrid[i])+Math.abs(withGrid[i+1]-withoutGrid[i+1])>20)foregroundGridPixels++;
              }
              window.renderTestView=view;
              return {imageWidth,tgaWidth,bones,sharedLoad,retainedVisual,movedReset,retained,overlay,count,targetShown,velocityHidden,planarTarget,targetHidden,
                trainingCameraStable,unfilteredPhysics,smoothFollow,gridSeparated,foregroundGridPixels,
                webgl:view.renderer.getContext().getParameter(view.renderer.getContext().VERSION)};
            }""", fixtures)
            assert result["imageWidth"] == 1 and result["tgaWidth"] == 1
            assert result["bones"] == 3 and result["retained"] and result["overlay"], result
            assert all(result[key] for key in ("sharedLoad", "retainedVisual", "movedReset", "trainingCameraStable",
                                              "unfilteredPhysics", "smoothFollow", "gridSeparated")), result
            assert result["count"] == 8 and result["webgl"].startswith("WebGL 2"), result
            assert result["foregroundGridPixels"] > 500, result
            assert all(result[key] for key in ("targetShown", "velocityHidden", "planarTarget", "targetHidden")), result
            before = page.evaluate("renderTestView.camera.azimuth")
            page.mouse.move(450, 325)
            page.mouse.down()
            page.mouse.move(500, 340)
            page.mouse.up()
            assert page.evaluate("renderTestView.camera.azimuth") != before
            page.evaluate("renderTestView.draw()")
            page.add_script_tag(content=(ROOT / "morpholab/static/renderer.js").read_text())
            fallback = page.evaluate("""async () => {
              window.RLsimScene={SceneView:class {constructor(){throw new Error("Test WebGL unavailable");}}};
              const original=document.querySelector("canvas");
              const view=new PhysicsView(original);
              view.resize();view.camera.follow=false;
              await view.setBody({kind:"mjcf",root_body:"root"});
              const mat=[1,0,0,0,1,0,0,0,1],size=[.2,.3,.4],pos=[0,0,.6];
              function pixels(frame) {
                view.setFrame(frame);view.draw();
                return view.ctx.getImageData(0,0,view.canvas.width,view.canvas.height).data;
              }
              const baseline=pixels({geoms:[]});
              function changed(frame) {
                const image=pixels(frame);let count=0;
                for(let i=0;i<image.length;i+=4)if(image[i]!==baseline[i]||image[i+1]!==baseline[i+1]||image[i+2]!==baseline[i+2])count++;
                return count;
              }
              const primitiveChanges=["box","cylinder","ellipsoid"].map(type=>changed({geoms:[{type,pos,mat,size}]}));
              const mesh={vertices:[-.2,0,0,.2,0,0,0,0,.4],faces:[0,1,2]};
              pixels({geoms:[{type:"mesh",pos,mat,size,mesh:"cached"}],meshes:{cached:mesh}});
              const meshChanges=changed({geoms:[{type:"mesh",pos,mat,size,mesh:"cached"}]});
              const targetChanges=changed({geoms:[],task:{preset:"target",target:[1,0,.8]}});
              view.camera.follow=true;view.setFrame({geoms:[],bodies:[{name:"root",pos:[2,3,1]}]});
              const follow=[...view.camera.target],anchor=[...view.camera.target];
              view.setMode("training");
              view.setFrame({time:.8,geoms:[],bodies:[{name:"root",pos:[8,6,.1]}]});view.draw(0);
              view.setFrame({time:0,geoms:[],bodies:[{name:"root",pos:[0,0,1]}]});view.draw(100);
              const stable=view.camera.target.every((v,i)=>v===anchor[i]);
              view.setMode("inference");
              view.setFrame({time:0,geoms:[],bodies:[{name:"root",pos:[0,0,1]}]});view.draw(0);
              view.setFrame({time:.02,geoms:[],bodies:[{name:"root",pos:[1,2,.1]}]});view.draw(100);
              const smooth=view.camera.target[0]>0 && view.camera.target[0]<1 && view.camera.target[2]===anchor[2];
              view.setMode("training");
              view.setFrame({time:.1,geoms:[],bodies:[{name:"root",pos:[8,3,.5]}]});
              view.reset({root_body:"root",visual:{bounds:{min:[-1,-1,0],max:[1,1,2]},
                bind_root:[1,0,0,0,0,1,0,0,0,0,1,0,0,0,.5,1]}});
              const movedReset=view.camera.target.join(",")==="8,3,1";
              return {replaced:original!==view.canvas,primitiveChanges,meshChanges,targetChanges,follow,stable,smooth,movedReset};
            }""")
            assert fallback["replaced"] and min(fallback["primitiveChanges"]) > 100, fallback
            assert fallback["meshChanges"] > 50 and fallback["targetChanges"] > 50, fallback
            assert fallback["follow"][:2] == [2, 3], fallback
            assert fallback["stable"] and fallback["smooth"] and fallback["movedReset"], fallback
            assert not external_requests and not errors, (external_requests, errors)
            browser.close()
    finally:
        server.shutdown()
        server.server_close()
        thread.join()
