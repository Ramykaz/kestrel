// Top-down sectional chart of a course and a flight, in SVG. Simulator x points right (east),
// y points up (north); SVG y is flipped.

import type { Flight } from "../../data/flight";
import type { Course } from "../../data/types";

const SVG_NS = "http://www.w3.org/2000/svg";
const MARGIN_M = 2.5;
const GATE_SYMBOL_M = 0.8;
const TICK_EVERY_S = 1;
const LABEL_TICK_EVERY_S = 10;

function el<K extends keyof SVGElementTagNameMap>(name: K, attrs: Record<string, string | number> = {}): SVGElementTagNameMap[K] {
  const node = document.createElementNS(SVG_NS, name);
  for (const [key, value] of Object.entries(attrs)) node.setAttribute(key, String(value));
  return node;
}

function polylinePoints(flight: Flight, upTo = flight.frames): string {
  const parts: string[] = [];
  for (let i = 0; i < upTo; i += 2) parts.push(`${flight.position[i * 3].toFixed(2)},${(-flight.position[i * 3 + 1]).toFixed(2)}`);
  const last = upTo - 1;
  parts.push(`${flight.position[last * 3].toFixed(2)},${(-flight.position[last * 3 + 1]).toFixed(2)}`);
  return parts.join(" ");
}

interface Bounds {
  minX: number;
  maxX: number;
  minY: number;
  maxY: number;
}

function courseBounds(course: Course, flights: (Flight | null)[]): Bounds {
  const xs = [course.start.position[0], ...course.gates.map((g) => g.center[0])];
  const ys = [course.start.position[1], ...course.gates.map((g) => g.center[1])];
  for (const flight of flights) {
    if (!flight) continue;
    for (let i = 0; i < flight.frames; i += 5) {
      xs.push(flight.position[i * 3]);
      ys.push(flight.position[i * 3 + 1]);
    }
  }
  return {
    minX: Math.min(...xs) - MARGIN_M,
    maxX: Math.max(...xs) + MARGIN_M,
    minY: Math.min(...ys) - MARGIN_M,
    maxY: Math.max(...ys) + MARGIN_M,
  };
}

export class FlightChart {
  readonly svg: SVGSVGElement;
  private readonly routeLayer = el("g", { class: "chart-route" });
  private readonly droneMarker = el("g", { class: "chart-drone" });
  private readonly targetMarker = el("circle", { class: "chart-target", r: 0.16 });
  private route: SVGPolylineElement | null = null;
  private routeLength = 0;
  private flight: Flight | null = null;

  constructor() {
    this.svg = el("svg", { class: "chart", role: "img", "aria-label": "Top-down chart of the course and flight path" });
    this.droneMarker.append(el("path", { d: "M0.34 0 L-0.2 0.2 L-0.1 0 L-0.2 -0.2 Z" }));
  }

  draw(course: Course, flight: Flight | null, ghost: Flight | null): void {
    this.flight = flight;
    const b = courseBounds(course, [flight, ghost]);
    const width = b.maxX - b.minX;
    const height = b.maxY - b.minY;
    this.svg.setAttribute("viewBox", `${b.minX} ${-b.maxY} ${width} ${height}`);
    this.svg.setAttribute("preserveAspectRatio", "xMidYMid meet");
    this.svg.replaceChildren();

    const grid = el("g", { class: "chart-grid" });
    for (let x = Math.ceil(b.minX); x <= b.maxX; x++) {
      grid.append(el("line", { x1: x, x2: x, y1: -b.maxY, y2: -b.minY, class: x % 5 === 0 ? "major" : "minor" }));
    }
    for (let y = Math.ceil(b.minY); y <= b.maxY; y++) {
      grid.append(el("line", { y1: -y, y2: -y, x1: b.minX, x2: b.maxX, class: y % 5 === 0 ? "major" : "minor" }));
    }
    const edgeLabels = el("g", { class: "chart-edge-labels" });
    for (let x = Math.ceil(b.minX / 5) * 5; x <= b.maxX - 1; x += 5) {
      const label = el("text", { x: x + 0.15, y: -b.minY - 0.25 });
      label.textContent = `${x} m`;
      edgeLabels.append(label);
    }
    for (let y = Math.ceil(b.minY / 5) * 5; y <= b.maxY - 1; y += 5) {
      const label = el("text", { x: b.minX + 0.2, y: -y - 0.15 });
      label.textContent = `${y} m`;
      edgeLabels.append(label);
    }

    const ghostLayer = el("g", { class: "chart-ghost" });
    if (ghost) ghostLayer.append(el("polyline", { points: polylinePoints(ghost) }));

    this.routeLayer.replaceChildren();
    this.route = null;
    this.routeLength = 0;
    if (flight) this.drawRoute(flight);

    const gates = el("g", { class: "chart-gates" });
    course.gates.forEach((gate, index) => {
      const [x, y] = gate.center;
      const deg = (-gate.yaw * 180) / Math.PI;
      const symbol = el("g", { transform: `translate(${x} ${-y}) rotate(${deg})` });
      const half = GATE_SYMBOL_M / 2;
      symbol.append(el("rect", { x: -half, y: -half, width: GATE_SYMBOL_M, height: GATE_SYMBOL_M, class: "chart-gate-box" }));
      // The tick sticks out of the front face, toward the side a drone flies through from.
      symbol.append(el("line", { x1: -half, y1: 0, x2: -half - 0.45, y2: 0, class: "chart-gate-tick" }));
      gates.append(symbol);
      const label = el("text", { x: x + 0.65, y: -y - 0.55, class: "chart-gate-label" });
      label.textContent = String(index + 1);
      gates.append(label);
    });

    const start = el("g", { class: "chart-start", transform: `translate(${course.start.position[0]} ${-course.start.position[1]})` });
    start.append(el("circle", { r: 0.22 }));
    const startLabel = el("text", { x: 0.4, y: 0.55 });
    startLabel.textContent = "Start";
    start.append(startLabel);

    const marks = el("g");
    if (flight && flight.header.end === "crash") {
      const last = flight.frames - 1;
      const cx = flight.position[last * 3];
      const cy = -flight.position[last * 3 + 1];
      const cross = el("g", { class: "chart-crash", transform: `translate(${cx} ${cy})` });
      cross.append(el("path", { d: "M-0.32 -0.32 L0.32 0.32 M-0.32 0.32 L0.32 -0.32" }));
      const label = el("text", { x: 0.45, y: 0.15 });
      label.textContent = "Crash";
      cross.append(label);
      marks.append(cross);
    }

    this.svg.append(grid, edgeLabels, this.compass(b), this.scaleBar(b), ghostLayer, this.routeLayer, gates, start, marks);
    if (flight) {
      this.svg.append(this.targetMarker, this.droneMarker);
      // Measured after insertion: detached SVG elements report no length in some browsers.
      this.routeLength = (this.route as SVGPolylineElement | null)?.getTotalLength() ?? 0;
    }
  }

  private drawRoute(flight: Flight): void {
    this.route = el("polyline", { points: polylinePoints(flight), class: "chart-route-line" });
    this.routeLayer.append(this.route);
    const ticks = el("g", { class: "chart-route-ticks" });
    const stepFrames = Math.round(TICK_EVERY_S * flight.hz);
    for (let f = stepFrames; f < flight.frames - 1; f += stepFrames) {
      const x = flight.position[f * 3];
      const y = flight.position[f * 3 + 1];
      const nx = flight.position[(f + 1) * 3] - flight.position[(f - 1) * 3];
      const ny = flight.position[(f + 1) * 3 + 1] - flight.position[(f - 1) * 3 + 1];
      const length = Math.hypot(nx, ny) || 1;
      const px = (-ny / length) * 0.14;
      const py = (nx / length) * 0.14;
      ticks.append(el("line", { x1: x - px, y1: -(y - py), x2: x + px, y2: -(y + py) }));
      const seconds = Math.round(f / flight.hz);
      if (seconds % LABEL_TICK_EVERY_S === 0) {
        const label = el("text", { x: x + px * 2.6, y: -(y + py * 2.6), class: "chart-tick-label" });
        label.textContent = `${seconds} s`;
        ticks.append(label);
      }
    }
    this.routeLayer.append(ticks);
  }

  private compass(b: Bounds): SVGGElement {
    const g = el("g", { class: "chart-compass", transform: `translate(${b.maxX - 1.3} ${-b.maxY + 1.6})` });
    g.append(el("circle", { r: 0.9 }));
    g.append(el("path", { d: "M0 -0.85 L0.22 0 L0 -0.2 L-0.22 0 Z", class: "chart-compass-needle" }));
    const n = el("text", { x: 0, y: -1.05 });
    n.textContent = "N";
    g.append(n);
    return g;
  }

  private scaleBar(b: Bounds): SVGGElement {
    const g = el("g", { class: "chart-scale", transform: `translate(${b.maxX - 6} ${-b.minY - 1.8})` });
    for (let i = 0; i < 5; i++) g.append(el("rect", { x: i, y: -0.12, width: 1, height: 0.12, class: i % 2 === 0 ? "dark" : "light" }));
    for (const [x, text] of [
      [0, "0"],
      [5, "5 m"],
    ] as const) {
      const label = el("text", { x, y: 0.5 });
      label.textContent = text;
      g.append(label);
    }
    return g;
  }

  /** Draw the route from the start up to a share of its length (0 to 1). */
  setRouteReveal(share: number): void {
    if (!this.route || this.routeLength === 0) return;
    this.route.style.strokeDasharray = `${this.routeLength}`;
    this.route.style.strokeDashoffset = `${this.routeLength * (1 - share)}`;
    if (share >= 1) {
      this.route.style.strokeDasharray = "";
      this.route.style.strokeDashoffset = "";
    }
  }

  update(time: number): void {
    const flight = this.flight;
    if (!flight) return;
    const f = Math.max(0, Math.min(flight.frames - 1, Math.round(time * flight.hz)));
    const x = flight.position[f * 3];
    const y = flight.position[f * 3 + 1];
    const q = flight.quaternion;
    const [qx, qy, qz, qw] = [q[f * 4], q[f * 4 + 1], q[f * 4 + 2], q[f * 4 + 3]];
    const yaw = Math.atan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz));
    this.droneMarker.setAttribute("transform", `translate(${x} ${-y}) rotate(${(-yaw * 180) / Math.PI})`);
    this.targetMarker.setAttribute("cx", String(flight.target[f * 3]));
    this.targetMarker.setAttribute("cy", String(-flight.target[f * 3 + 1]));
  }
}
