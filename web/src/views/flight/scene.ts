import * as THREE from "three";
import { Line2 } from "three/addons/lines/Line2.js";
import { LineGeometry } from "three/addons/lines/LineGeometry.js";
import { LineMaterial } from "three/addons/lines/LineMaterial.js";

import type { Flight } from "../../data/flight";
import type { Course } from "../../data/types";
import { token } from "../../theme";

export type CameraMode = "chase" | "onboard";

const GATE_INNER = 0.6;
const GATE_OUTER = 0.8;
const GATE_DEPTH = 0.05;
const POST_WIDTH = 0.04;
const VOLUME_HALF = 15;
const ONBOARD_VFOV_DEG = 73.74;
const ONBOARD_PITCH_UP = THREE.MathUtils.degToRad(15);
const DRONE_SCALE = 3;
const CHASE_BACK_M = 1.5;
const CHASE_UP_M = 0.5;
const CHASE_LOOK_AHEAD_M = 1.4;

export interface Pose {
  position: THREE.Vector3;
  quaternion: THREE.Quaternion;
}

/** Interpolated drone pose at a time, in simulator coordinates (z up). */
export function poseAt(flight: Flight, time: number, out: Pose): Pose {
  const f = Math.max(0, Math.min(flight.frames - 1, time * flight.hz));
  const i = Math.floor(f);
  const j = Math.min(i + 1, flight.frames - 1);
  const a = f - i;
  const p = flight.position;
  out.position.set(
    p[i * 3] + (p[j * 3] - p[i * 3]) * a,
    p[i * 3 + 1] + (p[j * 3 + 1] - p[i * 3 + 1]) * a,
    p[i * 3 + 2] + (p[j * 3 + 2] - p[i * 3 + 2]) * a,
  );
  const q = flight.quaternion;
  const qi = new THREE.Quaternion(q[i * 4], q[i * 4 + 1], q[i * 4 + 2], q[i * 4 + 3]).normalize();
  const qj = new THREE.Quaternion(q[j * 4], q[j * 4 + 1], q[j * 4 + 2], q[j * 4 + 3]).normalize();
  out.quaternion.copy(qi).slerp(qj, a);
  return out;
}

function pathPoints(flight: Flight, upTo = flight.frames): number[] {
  const points: number[] = [];
  const stride = 2;
  for (let i = 0; i < upTo; i += stride) points.push(flight.position[i * 3], flight.position[i * 3 + 1], flight.position[i * 3 + 2]);
  const last = upTo - 1;
  points.push(flight.position[last * 3], flight.position[last * 3 + 1], flight.position[last * 3 + 2]);
  return points;
}

function gateMesh(material: THREE.Material, centerHeight: number): THREE.Group {
  const group = new THREE.Group();
  const bar = (GATE_OUTER - GATE_INNER) / 2;
  const barCenter = (GATE_INNER + bar) / 2;
  const parts: [number, number, number, number, number, number][] = [
    [GATE_DEPTH, GATE_OUTER, bar, 0, 0, barCenter],
    [GATE_DEPTH, GATE_OUTER, bar, 0, 0, -barCenter],
    [GATE_DEPTH, bar, GATE_INNER, 0, barCenter, 0],
    [GATE_DEPTH, bar, GATE_INNER, 0, -barCenter, 0],
  ];
  for (const [sx, sy, sz, x, y, z] of parts) {
    const mesh = new THREE.Mesh(new THREE.BoxGeometry(sx, sy, sz), material);
    mesh.position.set(x, y, z);
    group.add(mesh);
  }
  const postHeight = centerHeight - GATE_OUTER / 2;
  const post = new THREE.Mesh(new THREE.BoxGeometry(POST_WIDTH, POST_WIDTH, postHeight), material);
  post.position.set(0, 0, -GATE_OUTER / 2 - postHeight / 2);
  group.add(post);
  return group;
}

function droneMesh(body: THREE.Material, rotor: THREE.Material): THREE.Group {
  // CF2X geometry: motors 0.0397 m from the center along the diagonals, nose along +x.
  const group = new THREE.Group();
  const arm = 0.0397;
  const motor = arm / Math.SQRT2;
  for (const angle of [Math.PI / 4, -Math.PI / 4]) {
    const mesh = new THREE.Mesh(new THREE.BoxGeometry(arm * 2, 0.007, 0.005), body);
    mesh.rotation.z = angle;
    group.add(mesh);
  }
  group.add(new THREE.Mesh(new THREE.BoxGeometry(0.032, 0.022, 0.012), body));
  const nose = new THREE.Mesh(new THREE.ConeGeometry(0.006, 0.016, 10), body);
  nose.rotation.z = -Math.PI / 2;
  nose.position.x = 0.024;
  group.add(nose);
  for (const [x, y] of [
    [1, 1],
    [1, -1],
    [-1, 1],
    [-1, -1],
  ]) {
    const disc = new THREE.Mesh(new THREE.RingGeometry(0.017, 0.0235, 28), rotor);
    disc.position.set(x * motor, y * motor, 0.007);
    group.add(disc);
  }
  group.scale.setScalar(DRONE_SCALE);
  return group;
}

function crossMesh(material: THREE.Material): THREE.Group {
  const group = new THREE.Group();
  for (const angle of [Math.PI / 4, -Math.PI / 4]) {
    const mesh = new THREE.Mesh(new THREE.BoxGeometry(0.5, 0.05, 0.05), material);
    mesh.rotation.z = angle;
    group.add(mesh);
  }
  return group;
}

export interface GateLabel {
  index: number;
  x: number;
  y: number;
  visible: boolean;
}

export class FlightScene {
  readonly renderer: THREE.WebGLRenderer;
  private readonly scene = new THREE.Scene();
  /** Simulator frame: z up. Rotated so three.js sees y up. */
  private readonly world = new THREE.Group();
  private readonly courseGroup = new THREE.Group();
  private readonly chaseCamera = new THREE.PerspectiveCamera(58, 1, 0.05, 200);
  private readonly onboardCamera = new THREE.PerspectiveCamera(ONBOARD_VFOV_DEG, 4 / 3, 0.05, 200);
  private readonly drone: THREE.Group;
  private readonly target: THREE.Mesh;
  private readonly crash: THREE.Group;
  private route: Line2 | null = null;
  private ghost: Line2 | null = null;
  private readonly materials = {
    ground: new THREE.MeshLambertMaterial(),
    gate: new THREE.MeshLambertMaterial(),
    drone: new THREE.MeshLambertMaterial(),
    rotor: new THREE.MeshBasicMaterial({ side: THREE.DoubleSide }),
    target: new THREE.MeshBasicMaterial({ side: THREE.DoubleSide, transparent: true, opacity: 0.9 }),
    alert: new THREE.MeshBasicMaterial(),
    gridMinor: new THREE.LineBasicMaterial({ transparent: true }),
    gridMajor: new THREE.LineBasicMaterial({ transparent: true }),
    boundary: new THREE.LineDashedMaterial({ dashSize: 0.6, gapSize: 0.4, transparent: true }),
    route: new LineMaterial({ linewidth: 2.5, worldUnits: false }),
    ghost: new LineMaterial({ linewidth: 1.5, worldUnits: false, dashed: true, dashSize: 0.25, gapSize: 0.18, transparent: true, opacity: 0.85 }),
  };
  private readonly hemisphere = new THREE.HemisphereLight(0xffffff, 0xffffff, 1.6);
  private readonly sun = new THREE.DirectionalLight(0xffffff, 1.2);
  private gates: { center: THREE.Vector3 }[] = [];
  private readonly smoothedEye = new THREE.Vector3();
  private readonly smoothedLook = new THREE.Vector3();
  private hasSmoothed = false;
  private readonly pose: Pose = { position: new THREE.Vector3(), quaternion: new THREE.Quaternion() };
  private width = 1;
  private height = 1;

  constructor(canvas: HTMLCanvasElement) {
    this.renderer = new THREE.WebGLRenderer({ canvas, antialias: true, powerPreference: "high-performance" });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    this.renderer.outputColorSpace = THREE.SRGBColorSpace;
    this.world.rotation.x = -Math.PI / 2;
    this.scene.add(this.world, this.hemisphere, this.sun);
    this.sun.position.set(-6, 14, 8);
    this.world.add(this.courseGroup);
    this.world.add(this.buildGround());
    this.drone = droneMesh(this.materials.drone, this.materials.rotor);
    this.target = new THREE.Mesh(new THREE.RingGeometry(0.1, 0.13, 36), this.materials.target);
    this.crash = crossMesh(this.materials.alert);
    this.crash.visible = false;
    this.world.add(this.drone, this.target, this.crash);
    this.chaseCamera.up.set(0, 1, 0);
    this.applyTheme();
  }

  private buildGround(): THREE.Group {
    const group = new THREE.Group();
    const extent = VOLUME_HALF * 2.4;
    const ground = new THREE.Mesh(new THREE.PlaneGeometry(extent * 2, extent * 2), this.materials.ground);
    ground.position.z = -0.002;
    group.add(ground);
    const minor: number[] = [];
    const major: number[] = [];
    for (let v = -extent; v <= extent; v += 1) {
      const target = v % 5 === 0 ? major : minor;
      target.push(v, -extent, 0, v, extent, 0, -extent, v, 0, extent, v, 0);
    }
    const lines = (points: number[], material: THREE.LineBasicMaterial) => {
      const geometry = new THREE.BufferGeometry();
      geometry.setAttribute("position", new THREE.Float32BufferAttribute(points, 3));
      return new THREE.LineSegments(geometry, material);
    };
    group.add(lines(minor, this.materials.gridMinor), lines(major, this.materials.gridMajor));
    const h = VOLUME_HALF;
    const boundaryGeometry = new THREE.BufferGeometry().setFromPoints(
      [
        [-h, -h],
        [h, -h],
        [h, h],
        [-h, h],
        [-h, -h],
      ].map(([x, y]) => new THREE.Vector3(x, y, 0.003)),
    );
    const boundary = new THREE.Line(boundaryGeometry, this.materials.boundary);
    boundary.computeLineDistances();
    group.add(boundary);
    return group;
  }

  applyTheme(): void {
    const color = (name: string) => new THREE.Color(token(name));
    const paper = color("--paper");
    const ink = color("--ink");
    const night = paper.getHSL({ h: 0, s: 0, l: 0 }).l < 0.4;
    this.scene.background = paper;
    this.scene.fog = new THREE.Fog(paper, 22, 70);
    this.materials.ground.color = color("--terrain").lerp(paper, night ? 0.62 : 0.55);
    this.materials.gate.color = ink;
    this.materials.drone.color = ink;
    this.materials.rotor.color = color("--ink-soft");
    this.materials.target.color = color("--airway");
    this.materials.alert.color = color("--alert");
    this.materials.gridMinor.color = ink;
    this.materials.gridMinor.opacity = night ? 0.07 : 0.06;
    this.materials.gridMajor.color = ink;
    this.materials.gridMajor.opacity = night ? 0.16 : 0.13;
    this.materials.boundary.color = color("--ink-soft");
    this.materials.boundary.opacity = 0.7;
    this.materials.route.color = color("--route");
    this.materials.ghost.color = color("--ink-soft");
    this.hemisphere.intensity = night ? 1.1 : 1.6;
    this.sun.intensity = night ? 0.8 : 1.2;
  }

  setCourse(course: Course): void {
    for (const child of [...this.courseGroup.children]) this.courseGroup.remove(child);
    this.gates = course.gates.map((gate) => {
      const mesh = gateMesh(this.materials.gate, gate.center[2]);
      mesh.position.set(...gate.center);
      mesh.rotation.z = gate.yaw;
      this.courseGroup.add(mesh);
      return { center: new THREE.Vector3(...gate.center) };
    });
  }

  private makeLine(points: number[], material: LineMaterial): Line2 {
    const geometry = new LineGeometry();
    geometry.setPositions(points);
    const line = new Line2(geometry, material);
    line.computeLineDistances();
    return line;
  }

  setFlight(flight: Flight): void {
    if (this.route) {
      this.world.remove(this.route);
      this.route.geometry.dispose();
    }
    this.route = this.makeLine(pathPoints(flight), this.materials.route);
    this.world.add(this.route);
    const last = flight.frames - 1;
    this.crash.visible = flight.header.end === "crash";
    this.crash.position.set(flight.position[last * 3], flight.position[last * 3 + 1], flight.position[last * 3 + 2]);
    this.hasSmoothed = false;
  }

  setGhost(flight: Flight | null): void {
    if (this.ghost) {
      this.world.remove(this.ghost);
      this.ghost.geometry.dispose();
      this.ghost = null;
    }
    if (!flight) return;
    this.ghost = this.makeLine(pathPoints(flight), this.materials.ghost);
    this.world.add(this.ghost);
  }

  /** Reveal the route from the start up to a share of its length (0 to 1). */
  setRouteReveal(share: number): void {
    if (!this.route) return;
    const geometry = this.route.geometry as LineGeometry;
    const segments = (geometry.attributes.instanceStart as THREE.InterleavedBufferAttribute).count;
    geometry.instanceCount = Math.max(0, Math.round(segments * share));
  }

  resize(width: number, height: number): void {
    this.width = Math.max(1, width);
    this.height = Math.max(1, height);
    this.renderer.setSize(this.width, this.height, false);
    this.chaseCamera.aspect = this.width / this.height;
    this.chaseCamera.updateProjectionMatrix();
    const resolution = new THREE.Vector2(this.width, this.height);
    this.materials.route.resolution = resolution;
    this.materials.ghost.resolution = resolution;
  }

  private placeOnboard(camera: THREE.PerspectiveCamera, pose: Pose): void {
    const worldPos = pose.position.clone().applyMatrix4(this.world.matrixWorld);
    const simForward = new THREE.Vector3(Math.cos(ONBOARD_PITCH_UP), 0, Math.sin(ONBOARD_PITCH_UP)).applyQuaternion(pose.quaternion);
    const simUp = new THREE.Vector3(-Math.sin(ONBOARD_PITCH_UP), 0, Math.cos(ONBOARD_PITCH_UP)).applyQuaternion(pose.quaternion);
    const forward = simForward.applyQuaternion(this.world.quaternion);
    camera.up.copy(simUp.applyQuaternion(this.world.quaternion));
    camera.position.copy(worldPos);
    camera.lookAt(worldPos.clone().add(forward));
  }

  private placeChase(pose: Pose, smooth: boolean): void {
    const simForward = new THREE.Vector3(1, 0, 0).applyQuaternion(pose.quaternion);
    simForward.z = 0;
    if (simForward.lengthSq() < 1e-6) simForward.set(1, 0, 0);
    simForward.normalize();
    const eyeSim = pose.position.clone().addScaledVector(simForward, -CHASE_BACK_M);
    eyeSim.z = Math.max(0.35, eyeSim.z + CHASE_UP_M);
    const lookSim = pose.position.clone().addScaledVector(simForward, CHASE_LOOK_AHEAD_M);
    const eye = eyeSim.applyMatrix4(this.world.matrixWorld);
    const look = lookSim.applyMatrix4(this.world.matrixWorld);
    if (smooth && this.hasSmoothed) {
      this.smoothedEye.lerp(eye, 0.12);
      this.smoothedLook.lerp(look, 0.18);
    } else {
      this.smoothedEye.copy(eye);
      this.smoothedLook.copy(look);
      this.hasSmoothed = true;
    }
    this.chaseCamera.position.copy(this.smoothedEye);
    this.chaseCamera.lookAt(this.smoothedLook);
  }

  /** Draw the frame. Returns the gate label positions in CSS pixels for the main view. */
  render(
    flight: Flight,
    time: number,
    mode: CameraMode,
    inset: { x: number; y: number; width: number; height: number } | null,
    smooth: boolean,
  ): GateLabel[] {
    this.world.updateMatrixWorld();
    const pose = poseAt(flight, time, this.pose);
    this.drone.position.copy(pose.position);
    this.drone.quaternion.copy(pose.quaternion);
    const frame = Math.max(0, Math.min(flight.frames - 1, Math.round(time * flight.hz)));
    this.target.position.set(flight.target[frame * 3], flight.target[frame * 3 + 1], flight.target[frame * 3 + 2]);
    this.target.lookAt(pose.position.clone().applyMatrix4(this.world.matrixWorld));

    let main: THREE.PerspectiveCamera;
    if (mode === "onboard") {
      this.onboardCamera.aspect = this.width / this.height;
      this.onboardCamera.updateProjectionMatrix();
      this.placeOnboard(this.onboardCamera, pose);
      main = this.onboardCamera;
    } else {
      this.placeChase(pose, smooth);
      main = this.chaseCamera;
    }

    this.renderer.setScissorTest(false);
    this.renderer.setViewport(0, 0, this.width, this.height);
    this.drone.visible = mode !== "onboard";
    this.renderer.render(this.scene, main);

    if (inset && mode === "chase") {
      const insetCamera = this.onboardCamera;
      insetCamera.aspect = 4 / 3;
      insetCamera.updateProjectionMatrix();
      this.placeOnboard(insetCamera, pose);
      const y = this.height - inset.y - inset.height;
      this.renderer.setScissorTest(true);
      this.renderer.setScissor(inset.x, y, inset.width, inset.height);
      this.renderer.setViewport(inset.x, y, inset.width, inset.height);
      this.drone.visible = false;
      this.target.visible = false;
      this.renderer.render(this.scene, insetCamera);
      this.target.visible = true;
      this.drone.visible = true;
      this.renderer.setScissorTest(false);
    }

    return this.gates.map((gate, index) => {
      const above = gate.center.clone();
      above.z += GATE_OUTER / 2 + 0.22;
      const projected = above.applyMatrix4(this.world.matrixWorld).project(main);
      return {
        index,
        x: ((projected.x + 1) / 2) * this.width,
        y: ((1 - projected.y) / 2) * this.height,
        visible: projected.z > -1 && projected.z < 1 && Math.abs(projected.x) < 1.1 && Math.abs(projected.y) < 1.1,
      };
    });
  }

  dispose(): void {
    this.renderer.dispose();
    this.scene.traverse((object) => {
      if (object instanceof THREE.Mesh || object instanceof THREE.Line || object instanceof THREE.LineSegments) object.geometry.dispose();
    });
    for (const material of Object.values(this.materials)) material.dispose();
  }
}
