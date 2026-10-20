// Decoder for the flight files described in kestrel/export/web_bundle.py.

import type { FlightEnd } from "./types";

export interface FlightHeader {
  schema: number;
  course_seed: number;
  course_preset: string;
  sample_hz: number;
  frames: number;
  image_size: [number, number];
  detections_source: "detector" | "ground_truth";
  gate_pass_times_s: number[];
  end: FlightEnd;
  end_time_s: number;
  gates_passed: number;
  gate_count: number;
  lap_time_s: number | null;
  crash_body_kind: string | null;
  arrays: { name: string; dtype: string; shape: number[] }[];
}

export interface FrameDetection {
  classId: number;
  confidence: number;
  /** Normalized x0, y0, x1, y1. */
  box: [number, number, number, number];
  /** Normalized keypoints: top left, top right, bottom right, bottom left. */
  keypoints: [number, number][];
  visibility: number[];
}

export interface Flight {
  header: FlightHeader;
  frames: number;
  hz: number;
  duration: number;
  position: Float32Array;
  quaternion: Float32Array;
  target: Float32Array;
  state: Uint8Array;
  detectionStart: Uint32Array;
  detectionCount: Uint8Array;
  detections: Float32Array;
}

const MAGIC = "KFL1";
const DETECTION_WIDTH = 15;

function halfToFloat(bits: number): number {
  const sign = bits & 0x8000 ? -1 : 1;
  const exponent = (bits >> 10) & 0x1f;
  const fraction = bits & 0x3ff;
  if (exponent === 0) return sign * 2 ** -14 * (fraction / 1024);
  if (exponent === 0x1f) return fraction ? Number.NaN : sign * Number.POSITIVE_INFINITY;
  return sign * 2 ** (exponent - 15) * (1 + fraction / 1024);
}

function readHalfs(view: DataView, offset: number, count: number): Float32Array {
  const out = new Float32Array(count);
  for (let i = 0; i < count; i++) out[i] = halfToFloat(view.getUint16(offset + i * 2, true));
  return out;
}

export function decodeFlight(buffer: ArrayBuffer): Flight {
  const view = new DataView(buffer);
  const magic = String.fromCharCode(...new Uint8Array(buffer, 0, 4));
  if (magic !== MAGIC) throw new Error(`not a Kestrel flight file (magic ${JSON.stringify(magic)})`);
  const headerLength = view.getUint32(4, true);
  const header = JSON.parse(new TextDecoder().decode(new Uint8Array(buffer, 8, headerLength))) as FlightHeader;

  const arrays: Record<string, Float32Array | Uint8Array> = {};
  let offset = 8 + headerLength;
  for (const spec of header.arrays) {
    const count = spec.shape.reduce((a, b) => a * b, 1);
    if (spec.dtype === "float16") {
      arrays[spec.name] = readHalfs(view, offset, count);
      offset += count * 2;
    } else if (spec.dtype === "uint8") {
      arrays[spec.name] = new Uint8Array(buffer.slice(offset, offset + count));
      offset += count;
    } else {
      throw new Error(`unsupported dtype ${spec.dtype} for ${spec.name}`);
    }
  }
  if (offset !== buffer.byteLength) {
    throw new Error(`flight file has ${buffer.byteLength - offset} unexpected trailing bytes`);
  }

  const detectionCount = arrays.det_count as Uint8Array;
  const detectionStart = new Uint32Array(header.frames);
  let running = 0;
  for (let i = 0; i < header.frames; i++) {
    detectionStart[i] = running;
    running += detectionCount[i];
  }

  return {
    header,
    frames: header.frames,
    hz: header.sample_hz,
    duration: header.frames > 0 ? (header.frames - 1) / header.sample_hz : 0,
    position: arrays.position as Float32Array,
    quaternion: arrays.quaternion as Float32Array,
    target: arrays.target as Float32Array,
    state: arrays.state as Uint8Array,
    detectionStart,
    detectionCount,
    detections: arrays.detections as Float32Array,
  };
}

export function frameDetections(flight: Flight, frame: number): FrameDetection[] {
  const out: FrameDetection[] = [];
  const start = flight.detectionStart[frame];
  for (let k = 0; k < flight.detectionCount[frame]; k++) {
    const row = flight.detections.subarray((start + k) * DETECTION_WIDTH, (start + k + 1) * DETECTION_WIDTH);
    const packed = Math.round(row[14]);
    out.push({
      classId: Math.round(row[0]),
      confidence: row[1],
      box: [row[2], row[3], row[4], row[5]],
      keypoints: [
        [row[6], row[7]],
        [row[8], row[9]],
        [row[10], row[11]],
        [row[12], row[13]],
      ],
      visibility: [0, 1, 2, 3].map((i) => (packed >> (2 * i)) & 3),
    });
  }
  return out;
}

export function frameAt(flight: Flight, time: number): number {
  return Math.max(0, Math.min(flight.frames - 1, Math.round(time * flight.hz)));
}
