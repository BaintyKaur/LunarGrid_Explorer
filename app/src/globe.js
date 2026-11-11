// The 3D Moon: a sphere painted from the data, with picking, a selection marker and camera moves.
import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import { D, R, HALF, CELL, NC } from './model.js';

const TW = 4096, TH = 2048;          // texture size, 0.088 degrees per texel
const RAD = Math.PI / 180;

export function createGlobe(container, onPick, onHover) {
  const renderer = new THREE.WebGLRenderer({ antialias: true });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
  renderer.setClearColor(0x05070b, 1);
  container.prepend(renderer.domElement);
  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(38, 1, 0.05, 100);
  const controls = new OrbitControls(camera, renderer.domElement);
  controls.enablePan = false; controls.enableDamping = true; controls.dampingFactor = 0.08;
  controls.minDistance = 1.35; controls.maxDistance = 6; controls.rotateSpeed = 0.6; controls.zoomSpeed = 0.8;
  controls.autoRotateSpeed = 0.6;

  // texture canvas
  const canvas = document.createElement('canvas'); canvas.width = TW; canvas.height = TH;
  const ctx = canvas.getContext('2d');
  const img = ctx.createImageData(TW, TH), img32 = new Uint32Array(img.data.buffer);
  const tex = new THREE.CanvasTexture(canvas);
  tex.colorSpace = THREE.SRGBColorSpace; tex.magFilter = THREE.NearestFilter;
  tex.anisotropy = renderer.capabilities.getMaxAnisotropy();

  const moon = new THREE.Mesh(new THREE.SphereGeometry(1, 192, 96), new THREE.MeshBasicMaterial({ map: tex }));
  moon.rotation.y = -Math.PI / 2;          // longitude 0 faces the camera
  scene.add(moon);

  // grid lines
  const gpts = [];
  const sph = (lat, lon, r) => { const a = lat * RAD, b = lon * RAD; return [r * Math.cos(a) * Math.sin(b), r * Math.sin(a), r * Math.cos(a) * Math.cos(b)]; };
  for (let lat = -60; lat <= 60; lat += 30) for (let lon = -180; lon < 180; lon += 3) gpts.push(...sph(lat, lon, 1.0015), ...sph(lat, lon + 3, 1.0015));
  for (let lon = -180; lon < 180; lon += 30) for (let lat = -90; lat < 90; lat += 3) gpts.push(...sph(lat, lon, 1.0015), ...sph(lat + 3, lon, 1.0015));
  const gg = new THREE.BufferGeometry(); gg.setAttribute('position', new THREE.Float32BufferAttribute(gpts, 3));
  const grid = new THREE.LineSegments(gg, new THREE.LineBasicMaterial({ color: 0xffffff, transparent: true, opacity: 0.22 }));
  scene.add(grid);

  // stars
  const sp = []; for (let i = 0; i < 1400; i++) { const u = Math.random() * 2 - 1, t = Math.random() * 6.2832, s = Math.sqrt(1 - u * u); sp.push(30 * s * Math.cos(t), 30 * u, 30 * s * Math.sin(t)); }
  const sg = new THREE.BufferGeometry(); sg.setAttribute('position', new THREE.Float32BufferAttribute(sp, 3));
  scene.add(new THREE.Points(sg, new THREE.PointsMaterial({ color: 0xb8c2d6, size: 0.06, sizeAttenuation: true, transparent: true, opacity: 0.7 })));

  // selection marker: a black ring with a white ring inside, lying on the surface
  const marker = new THREE.Group(); marker.visible = false;
  const ringMat = (c) => new THREE.MeshBasicMaterial({ color: c, side: THREE.DoubleSide, depthTest: false, transparent: true });
  const ringA = new THREE.Mesh(new THREE.RingGeometry(0.86, 1, 48), ringMat(0x000000));
  const ringB = new THREE.Mesh(new THREE.RingGeometry(0.9, 0.97, 48), ringMat(0xffffff));
  ringA.renderOrder = 10; ringB.renderOrder = 11; ringB.position.z = 0.0005;
  marker.add(ringA, ringB); moon.add(marker);
  let markerHalf = 0.02;

  // lookup tables: which tile and which polar cell lies under each texel
  const texTile = new Int16Array(TW * TH).fill(-1), texCell = new Int16Array(TW * TH).fill(-1);
  function buildLookups() {
    for (let j = 0; j < TH; j++) {
      const lat = 90 - (j + 0.5) / TH * 180, row = Math.max(0, Math.min(35, Math.floor((lat + 90) / 5)));
      const polar = Math.abs(lat) >= 65, north = lat > 0;
      const rho = polar ? 2 * R * Math.tan((90 - Math.abs(lat)) / 2 * RAD) : 0, grid = north ? D.cellGrid.N : D.cellGrid.S;
      for (let i = 0; i < TW; i++) {
        const lon = -180 + (i + 0.5) / TW * 360, p = j * TW + i;
        texTile[p] = D.tileGrid[row * 72 + Math.max(0, Math.min(71, Math.floor((lon + 180) / 5)))];
        if (polar) {
          const a = lon * RAD, x = rho * Math.sin(a), y = (north ? -1 : 1) * rho * Math.cos(a);
          const ix = Math.floor((x + HALF) / CELL), iy = Math.floor((y + HALF) / CELL);
          if (ix >= 0 && iy >= 0 && ix < NC && iy < NC) texCell[p] = grid[iy * NC + ix];
        }
      }
    }
  }

  function repaint(colT, colC, usePolar) {
    const n = TW * TH;
    for (let p = 0; p < n; p++) {
      const ci = usePolar ? texCell[p] : -1;
      img32[p] = ci >= 0 ? colC[ci] : colT[texTile[p]];
    }
    ctx.putImageData(img, 0, 0);
    tex.needsUpdate = true;
  }

  // marker placement uses the sphere's own texture mapping so it always matches a click
  function localPoint(lat, lon) {
    const phi = 2 * Math.PI * (lon + 180) / 360, theta = Math.PI * (1 - (lat + 90) / 180);
    return new THREE.Vector3(-Math.cos(phi) * Math.sin(theta), Math.cos(theta), Math.sin(phi) * Math.sin(theta));
  }
  function setSelected(lat, lon, halfRad) {
    if (lat == null) { marker.visible = false; return; }
    const p = localPoint(lat, lon);
    marker.position.copy(p).multiplyScalar(1.004);
    marker.quaternion.setFromUnitVectors(new THREE.Vector3(0, 0, 1), p);
    markerHalf = halfRad; marker.visible = true;
  }

  // camera
  function look(lat, lon, dist) {
    // stop just short of the poles so the orbit controls never flip
    lat = Math.max(-84, Math.min(84, lat));
    const d = dist || camera.position.length(), a = lat * RAD, b = lon * RAD;
    camera.position.set(d * Math.cos(a) * Math.sin(b), d * Math.sin(a), d * Math.cos(a) * Math.cos(b));
    controls.update();
  }
  camera.position.set(0, 0, 3.1);

  // picking
  const ray = new THREE.Raycaster(), ndc = new THREE.Vector2();
  function pick(clientX, clientY) {
    const r = renderer.domElement.getBoundingClientRect();
    ndc.set(((clientX - r.left) / r.width) * 2 - 1, -(((clientY - r.top) / r.height) * 2 - 1));
    ray.setFromCamera(ndc, camera);
    const hit = ray.intersectObject(moon, false)[0];
    if (!hit || !hit.uv) return null;
    return { lat: (hit.uv.y - 0.5) * 180, lon: (hit.uv.x - 0.5) * 360 };
  }

  let down = null;
  const el = renderer.domElement;
  el.addEventListener('pointerdown', (e) => { down = { x: e.clientX, y: e.clientY }; });
  el.addEventListener('pointerup', (e) => {
    if (down && Math.hypot(e.clientX - down.x, e.clientY - down.y) < 5) { const p = pick(e.clientX, e.clientY); if (p) onPick(p.lat, p.lon); }
    down = null;
  });
  let hoverQueued = false, lastEv = null;
  el.addEventListener('pointermove', (e) => {
    lastEv = e;
    if (hoverQueued || e.buttons) { if (e.buttons) onHover(null, e); return; }
    hoverQueued = true;
    requestAnimationFrame(() => { hoverQueued = false; const p = pick(lastEv.clientX, lastEv.clientY); onHover(p, lastEv); });
  });
  el.addEventListener('pointerleave', () => onHover(null, null));

  function resize() {
    const w = Math.max(200, container.clientWidth), h = Math.max(200, container.clientHeight);
    renderer.setSize(w, h, false); camera.aspect = w / h; camera.updateProjectionMatrix();
  }
  new ResizeObserver(resize).observe(container);
  resize();

  renderer.setAnimationLoop(() => {
    controls.update();
    const k = Math.max(markerHalf, 0.016 * camera.position.length() / 3.1);
    marker.scale.setScalar(k);
    renderer.render(scene, camera);
  });

  return {
    ready() { buildLookups(); },
    repaint, setSelected, look, pick,
    setAutoRotate(v) { controls.autoRotate = v; },
    setGrid(v) { grid.visible = v; },
    resize
  };
}
