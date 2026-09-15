import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

const clamp = (value, minimum, maximum) => Math.max(minimum, Math.min(maximum, value));

export function coveredTriangleIndices(covered, columns, rows) {
  const indices = [];
  for (let row = 0; row < rows - 1; row++) {
    for (let col = 0; col < columns - 1; col++) {
      const topLeft = row * columns + col;
      const topRight = topLeft + 1;
      const bottomLeft = topLeft + columns;
      const bottomRight = bottomLeft + 1;
      if (covered[topLeft] && covered[bottomLeft] && covered[topRight]) indices.push(topLeft, bottomLeft, topRight);
      if (covered[topRight] && covered[bottomLeft] && covered[bottomRight]) indices.push(topRight, bottomLeft, bottomRight);
    }
  }
  return indices;
}

export class Field3DView {
  constructor(canvas) {
    this.canvas = canvas;
    this.renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: false });
    this.renderer.setPixelRatio(Math.min(devicePixelRatio || 1, 2));
    this.renderer.setClearColor(0xe2e7e4, 1);
    this.scene = new THREE.Scene();
    this.camera = new THREE.PerspectiveCamera(38, 1, 0.01, 30);
    this.camera.position.set(2.4, 1.9, 2.7);
    this.controls = new OrbitControls(this.camera, canvas);
    this.controls.enableDamping = true;
    this.controls.dampingFactor = 0.08;
    this.controls.target.set(0, 0.15, 0);
    this.controls.minDistance = 1.2;
    this.controls.maxDistance = 8;
    this.controls.maxPolarAngle = Math.PI * 0.49;
    this.controls.addEventListener('change', () => this.drawScene());
    this.scene.add(new THREE.HemisphereLight(0xffffff, 0x64706b, 2.2));
    const light = new THREE.DirectionalLight(0xffffff, 2.4);
    light.position.set(-2, 4, 3);
    this.scene.add(light);
    this.grid = new THREE.GridHelper(2.4, 12, 0x718079, 0xb3bdb8);
    this.grid.position.y = -0.015;
    this.scene.add(this.grid);
    this.mesh = null;
    this.wire = null;
    this.frame = null;
  }

  update(calibration, range, heightScale, rigid = null) {
    const { field, maps } = calibration;
    const columns = Math.min(81, Math.max(25, Math.ceil(field.width / 18)));
    const rows = Math.min(81, Math.max(25, Math.ceil(field.height / 18)));
    const positions = new Float32Array(columns * rows * 3);
    const colors = new Float32Array(columns * rows * 3);
    const covered = new Uint8Array(columns * rows);
    const aspect = field.width / field.height;
    const spanX = aspect >= 1 ? 2.2 : 2.2 * aspect;
    const spanZ = aspect >= 1 ? 2.2 / aspect : 2.2;
    for (let row = 0; row < rows; row++) {
      const py = Math.round(row / (rows - 1) * (field.height - 1));
      for (let col = 0; col < columns; col++) {
        const px = Math.round(col / (columns - 1) * (field.width - 1));
        const sourceIndex = py * field.width + px;
        let ux = maps.forward[sourceIndex * 2];
        let uy = maps.forward[sourceIndex * 2 + 1];
        if (rigid) {
          const shiftedX = ux - rigid.tx;
          const shiftedY = uy - rigid.ty;
          ux = Math.cos(rigid.theta) * shiftedX + Math.sin(rigid.theta) * shiftedY;
          uy = -Math.sin(rigid.theta) * shiftedX + Math.cos(rigid.theta) * shiftedY;
        }
        const dx = ux - px;
        const dy = uy - py;
        const magnitude = Math.hypot(dx, dy);
        const vertex = (row * columns + col) * 3;
        positions[vertex] = col / (columns - 1) * spanX - spanX / 2;
        positions[vertex + 1] = Math.min(magnitude / range, 2) * 0.65 * heightScale;
        positions[vertex + 2] = row / (rows - 1) * spanZ - spanZ / 2;
        covered[row * columns + col] = maps.sourceCoverage[sourceIndex] >= 3;
        colors[vertex] = 0.5 + 0.5 * clamp(dx / range, -1, 1);
        colors[vertex + 1] = 0;
        colors[vertex + 2] = 0.5 + 0.5 * clamp(dy / range, -1, 1);
      }
    }
    const indices = coveredTriangleIndices(covered, columns, rows);
    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));
    geometry.setIndex(indices);
    geometry.computeVertexNormals();
    this.mesh?.geometry.dispose();
    this.wire?.geometry.dispose();
    if (!this.mesh) {
      this.mesh = new THREE.Mesh(geometry, new THREE.MeshStandardMaterial({ vertexColors: true, roughness: 0.72, metalness: 0, side: THREE.DoubleSide }));
      this.wire = new THREE.Mesh(geometry, new THREE.MeshBasicMaterial({ color: 0x17342b, wireframe: true, transparent: true, opacity: 0.13 }));
      this.scene.add(this.mesh, this.wire);
    } else {
      this.mesh.geometry = geometry;
      this.wire.geometry = geometry.clone();
    }
    this.render();
  }

  resize() {
    const bounds = this.canvas.getBoundingClientRect();
    const width = Math.max(1, Math.round(bounds.width));
    const height = Math.max(1, Math.round(bounds.height));
    this.renderer.setSize(width, height, false);
    this.camera.aspect = width / height;
    this.camera.updateProjectionMatrix();
    this.render();
  }

  render() {
    if (this.canvas.hidden) return;
    this.controls.update();
    this.drawScene();
  }

  drawScene() {
    if (this.canvas.hidden) return;
    this.renderer.render(this.scene, this.camera);
  }
}