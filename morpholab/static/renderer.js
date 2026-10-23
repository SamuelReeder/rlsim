"use strict";

// The renderer projects MuJoCo's world-space geometry; it never simulates motion.
class CanvasPhysicsView {
  constructor(canvas) {
    this.canvas = canvas;
    this.ctx = canvas.getContext("2d");
    this.frame = null;
    this.meshes = new Map();
    this.camera = { azimuth: -1.15, elevation: .30, distance: 4.2, follow: true, target: [0, 0, .45] };
    this.followCamera = new globalThis.RLsimCameraFollow(this.camera);
    this.dirty = true;
    this.width = this.height = 0;
    new ResizeObserver(() => this.resize()).observe(canvas);
    let pointer = null;
    canvas.addEventListener("pointerdown", e => {
      pointer = [e.clientX, e.clientY];
      canvas.setPointerCapture(e.pointerId);
    });
    canvas.addEventListener("pointermove", e => {
      if (!pointer) return;
      this.camera.azimuth -= (e.clientX - pointer[0]) * .008;
      this.camera.elevation = Math.max(.025, Math.min(1.45, this.camera.elevation + (e.clientY - pointer[1]) * .007));
      pointer = [e.clientX, e.clientY];
      this.dirty = true;
    });
    for (const name of ["pointerup", "pointercancel", "lostpointercapture"]) canvas.addEventListener(name, () => { pointer = null; });
    canvas.addEventListener("wheel", e => {
      e.preventDefault();
      this.camera.distance = Math.max(1.4, Math.min(18, this.camera.distance * Math.exp(e.deltaY * .001)));
      this.dirty = true;
    }, { passive: false });
  }
  resize() {
    const r = this.canvas.getBoundingClientRect();
    this.width = r.width;
    this.height = r.height;
    const dpr = Math.min(devicePixelRatio || 1, 2);
    this.canvas.width = Math.round(r.width * dpr);
    this.canvas.height = Math.round(r.height * dpr);
    this.ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    this.dirty = true;
  }
  setFrame(frame) {
    if (!frame?.geoms) return;
    this.frame = frame;
    for (const [name, mesh] of Object.entries(frame.meshes || {})) this.meshes.set(name, mesh);
    this.followCamera.setFrame(frame, this.body);
    this.dirty = true;
  }
  async setBody(body) {
    if (body !== this.body) { this.meshes.clear(); this.followCamera.reset(); }
    this.body = body;
    if (this.frame) this.setFrame(this.frame);
    this.dirty = true;
  }
  setMode(mode) { this.followCamera.setMode(mode); }
  reset(body = this.body) {
    this.followCamera.reset();
    if (body?.visual?.bounds) {
      const {min, max} = body.visual.bounds;
      this.camera.target = globalThis.RLsimCameraFollow.visualCenter(body, this.frame);
      this.camera.distance = Math.max(.1, Math.hypot(...max.map((v, i) => v - min[i])) * 2.4);
      this.setAngle("perspective");
      return;
    }
    this.camera.distance = Math.max(3, Math.max((body?.torso_length || .8) * 1.3, ((body?.upper_length || .42) + (body?.lower_length || .42)) * 1.8) * 2.3);
    const root = this.frame?.bodies?.find(item => item.name === body?.root_body)?.pos ||
      this.frame?.torso_position || [this.frame?.position || 0, 0, .8];
    this.camera.target = [root[0], root[1], root[2] * .55];
    this.setAngle("perspective");
  }
  setAngle(angle) {
    const angles = { perspective: [-1.15, .30], side: [-Math.PI / 2, .025], front: [0, .025] };
    [this.camera.azimuth, this.camera.elevation] = angles[angle] || angles.perspective;
    this.dirty = true;
  }
  projector() {
    const c = this.camera;
    const eye = [Math.cos(c.azimuth) * Math.cos(c.elevation), Math.sin(c.azimuth) * Math.cos(c.elevation), Math.sin(c.elevation)].map((v, i) => c.target[i] + v * c.distance);
    const normalize = a => { const d = Math.hypot(...a); return a.map(v => v / d); };
    const dot = (a, b) => a.reduce((s, v, i) => s + v * b[i], 0);
    const cross = (a, b) => [a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0]];
    const forward = normalize(c.target.map((v, i) => v - eye[i]));
    const right = normalize(cross(forward, [0, 0, 1]));
    const up = cross(right, forward);
    const focal = Math.min(this.width, this.height * 1.5) * 1.05;
    return point => {
      const p = point.map((v, i) => v - eye[i]);
      const depth = dot(p, forward);
      if (depth < .08) return null;
      const scale = focal / depth;
      return { x: this.width / 2 + dot(p, right) * scale, y: this.height / 2 - dot(p, up) * scale, scale, depth };
    };
  }
  line(a, b, project, stroke, thickness = 1) {
    const p = project(a), q = project(b);
    if (!p || !q) return;
    const ctx = this.ctx;
    ctx.beginPath(); ctx.moveTo(p.x, p.y); ctx.lineTo(q.x, q.y);
    ctx.strokeStyle = stroke; ctx.lineWidth = thickness; ctx.stroke();
  }
  draw(now) {
    if (this.followCamera.advance(now)) this.dirty = true;
    if (!this.dirty || !this.width || !this.height) return;
    this.dirty = false;
    const ctx = this.ctx, project = this.projector();
    ctx.clearRect(0, 0, this.width, this.height);
    ctx.fillStyle = "#f7f7f5"; ctx.fillRect(0, 0, this.width, this.height);
    const x = Math.round(this.camera.target[0]), extent = Math.min(12, Math.ceil(this.camera.distance * 1.6));
    ctx.lineCap = "butt";
    for (let i = -extent; i <= extent; i++) {
      this.line([x+i, -extent, 0], [x+i, extent, 0], project, "#deded9", .7);
      this.line([x-extent, i, 0], [x+extent, i, 0], project, i === 0 ? "#b8b8b1" : "#deded9", i === 0 ? 1.2 : .7);
    }
    ctx.font = "11px ui-monospace, monospace"; ctx.fillStyle = "#71716c"; ctx.textAlign = "center";
    for (let i = -3; i <= 3; i++) {
      const p = project([x+i, -1, .005]);
      if (p && p.y > 10 && p.y < this.height - 15) ctx.fillText(`${x+i} m`, p.x, p.y);
    }
    const geoms = (this.frame?.geoms || []).filter(g => g.type !== "plane");
    geoms.map((geom, index) => ({ geom, index, depth: project(geom.pos)?.depth || 0 })).sort((a,b) => b.depth - a.depth).forEach(({geom:g, index}) => {
      const p = project(g.pos);
      if (!p) return;
      const body = index === 0;
      const fill = body ? "#9ba5ab" : (g.rgba?.[1] > .7 ? "#7d96a0" : "#526b77");
      const mat = g.mat || [1,0,0,0,1,0,0,0,1];
      const world = v => g.pos.map((origin, row) => origin + mat[row*3]*v[0] + mat[row*3+1]*v[1] + mat[row*3+2]*v[2]);
      const mesh = typeof g.mesh === "string" ? this.meshes.get(g.mesh) : g.mesh;
      if (mesh) {
        const vertex = i => world(mesh.vertices.slice(i*3, i*3+3));
        const faces = mesh.faces || Array.from({length: mesh.vertices.length / 3}, (_, i) => i);
        // Keep the non-WebGL fallback responsive for large collision meshes.
        const stride = Math.max(1, Math.ceil(faces.length / 18000));
        for (let i = 0; i + 2 < faces.length; i += 3 * stride) {
          const vertices = [vertex(faces[i]), vertex(faces[i+1]), vertex(faces[i+2])];
          for (let j = 0; j < 3; j++) this.line(vertices[j], vertices[(j+1)%3], project, "#526b77", .8);
        }
      } else if (g.type === "box") {
        const vertices = Array.from({length: 8}, (_, i) => world(g.size.map((size, axis) => size * (i & (1 << axis) ? 1 : -1))));
        const faces = [[0,1,3,2],[4,6,7,5],[0,4,5,1],[2,3,7,6],[0,2,6,4],[1,5,7,3]];
        faces.map(face => face.map(i => project(vertices[i]))).filter(face => face.every(Boolean))
          .sort((a,b) => b.reduce((s,v)=>s+v.depth,0)-a.reduce((s,v)=>s+v.depth,0)).forEach(face => {
            ctx.beginPath();face.forEach((v,i)=>i?ctx.lineTo(v.x,v.y):ctx.moveTo(v.x,v.y));ctx.closePath();
            ctx.fillStyle=fill;ctx.fill();ctx.strokeStyle="#43535b";ctx.lineWidth=1;ctx.stroke();
          });
      } else if (g.type === "ellipsoid" || g.type === "cylinder") {
        const cylinder = g.type === "cylinder";
        for (let ring = 0; ring < (cylinder ? 2 : 3); ring++) {
          let previous;
          for (let i = 0; i <= 32; i++) {
            const angle = i * Math.PI / 16, point = cylinder
              ? [g.size[0]*Math.cos(angle),g.size[0]*Math.sin(angle),g.size[1]*(ring?1:-1)]
              : [0,0,0].map((_,axis) => axis===ring ? 0 : g.size[axis] * (axis===(ring+1)%3 ? Math.cos(angle) : Math.sin(angle)));
            const current = world(point);
            if (previous) this.line(previous,current,project,"#526b77",1.3);
            previous=current;
          }
        }
        if (cylinder) for (let i=0;i<4;i++) {
          const a=i*Math.PI/2,xy=[g.size[0]*Math.cos(a),g.size[0]*Math.sin(a)];
          this.line(world([...xy,-g.size[1]]),world([...xy,g.size[1]]),project,"#526b77",1.3);
        }
      } else if (g.type === "capsule") {
        const axis = [g.mat[2], g.mat[5], g.mat[8]];
        const a = project(g.pos.map((v,i) => v + axis[i] * g.size[1]));
        const b = project(g.pos.map((v,i) => v - axis[i] * g.size[1]));
        if (!a || !b) return;
        const radius = Math.max(1, g.size[0] * (a.scale+b.scale) / 2);
        ctx.lineCap = "round";
        ctx.beginPath(); ctx.moveTo(a.x,a.y); ctx.lineTo(b.x,b.y);
        ctx.strokeStyle = "#394c55"; ctx.lineWidth = radius*2 + 1.5; ctx.stroke();
        ctx.strokeStyle = fill; ctx.lineWidth = radius*2; ctx.stroke();
        // Endpoints mark the joint positions for inspection.
        if (!body) for (const q of [a,b]) {
          ctx.beginPath(); ctx.arc(q.x,q.y,Math.max(2,radius*.35),0,Math.PI*2);
          ctx.fillStyle = "#e8edef"; ctx.fill(); ctx.strokeStyle = "#43535b"; ctx.lineWidth = 1; ctx.stroke();
        }
      } else if (g.type === "sphere") {
        ctx.beginPath(); ctx.arc(p.x,p.y,Math.max(1,g.size[0]*p.scale),0,Math.PI*2);
        ctx.fillStyle = "#b5bec3"; ctx.fill(); ctx.strokeStyle = "#43535b"; ctx.lineWidth = 1; ctx.stroke();
      }
    });
    const target = this.frame?.target || (["target","custom"].includes(this.frame?.task?.preset) ? this.frame.task.target : null);
    if (target?.length === 3 && Array.from(target).every(Number.isFinite)) {
      const point = project(target);
      if (point) {
        const radius = Math.max(5,.2*point.scale);
        ctx.beginPath();ctx.arc(point.x,point.y,radius,0,2*Math.PI);
        ctx.moveTo(point.x-radius-4,point.y);ctx.lineTo(point.x+radius+4,point.y);
        ctx.moveTo(point.x,point.y-radius-4);ctx.lineTo(point.x,point.y+radius+4);
        ctx.strokeStyle="#de752f";ctx.lineWidth=1.5;ctx.stroke();
      }
    }
  }
}

window.PhysicsView = class {
  constructor(canvas) {
    if (window.RLsimScene?.SceneView) {
      try { return new window.RLsimScene.SceneView(canvas); }
      catch (error) {
        console.warn("3D rendering unavailable; using the canvas view.", error.message);
        // A failed WebGL setup may already own the canvas context.
        const replacement = canvas.cloneNode(true);
        canvas.replaceWith(replacement);
        canvas = replacement;
      }
    }
    return new CanvasPhysicsView(canvas);
  }
};

window.drawRewardChart = function(canvas, history, totalSteps) {
  const rect = canvas.getBoundingClientRect();
  if (!rect.width || !rect.height) return;
  const ctx = canvas.getContext("2d"), ratio = Math.min(devicePixelRatio || 1,2);
  canvas.width = rect.width * ratio; canvas.height = rect.height * ratio;
  ctx.setTransform(ratio,0,0,ratio,0,0);
  const w=rect.width, h=rect.height;
  ctx.clearRect(0,0,w,h);
  const points = (history || []).filter(p=>Number.isFinite(p.reward));
  if (!points.length) return;
  let low=Math.min(...points.map(p=>p.reward)), high=Math.max(...points.map(p=>p.reward));
  const margin=Math.max((high-low)*.1,1); low-=margin; high+=margin;
  const left=42, right=w-5, top=10, bottom=h-20;
  ctx.font="10px ui-monospace,monospace"; ctx.fillStyle="#62625d"; ctx.textAlign="right";
  for (const [value,y] of [[high,top],[low,bottom]]) {
    ctx.fillText(Math.round(value),left-5,y+3);
    ctx.beginPath();ctx.moveTo(left,y);ctx.lineTo(right,y);ctx.strokeStyle="#ddddD7";ctx.lineWidth=1;ctx.stroke();
  }
  ctx.textAlign="left";ctx.fillText("0",left,h-4);
  ctx.textAlign="right";ctx.fillText(`${Number(totalSteps || 0).toLocaleString()} steps`,right,h-4);
  const end=Math.max(totalSteps || 0,points.at(-1).step,1);
  ctx.beginPath();
  points.forEach((p,i)=>{const x=left+p.step/end*(right-left),y=top+(high-p.reward)/(high-low)*(bottom-top); i ? ctx.lineTo(x,y) : ctx.moveTo(x,y);});
  ctx.strokeStyle="#476579";ctx.lineWidth=1.5;ctx.stroke();
};
