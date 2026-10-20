import { frameDetections, type Flight } from "../../data/flight";
import { token } from "../../theme";

const KEYPOINT_RADIUS = 2.6;

/** Draw one frame's logged detections, normalized to the inset size. */
export function drawDetections(canvas: HTMLCanvasElement, flight: Flight, frame: number): number {
  const ratio = Math.min(window.devicePixelRatio, 2);
  const width = canvas.clientWidth;
  const height = canvas.clientHeight;
  if (canvas.width !== Math.round(width * ratio) || canvas.height !== Math.round(height * ratio)) {
    canvas.width = Math.round(width * ratio);
    canvas.height = Math.round(height * ratio);
  }
  const ctx = canvas.getContext("2d");
  if (!ctx) return 0;
  ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
  ctx.clearRect(0, 0, width, height);
  const airway = token("--airway");
  const paper = token("--paper");
  const detections = frameDetections(flight, frame);
  ctx.lineWidth = 1.5;
  ctx.font = `600 11px ${token("--font-display")}`;
  ctx.textBaseline = "bottom";
  for (const det of detections) {
    const [x0, y0, x1, y1] = det.box;
    const bx = x0 * width;
    const by = y0 * height;
    const bw = (x1 - x0) * width;
    const bh = (y1 - y0) * height;
    if (bw * bh < 9) continue;
    ctx.strokeStyle = airway;
    ctx.setLineDash(det.classId === 1 ? [3, 2] : []);
    ctx.strokeRect(bx, by, bw, bh);
    ctx.setLineDash([]);
    const label = `${det.classId === 1 ? "back " : ""}${det.confidence.toFixed(2)}`;
    const textWidth = ctx.measureText(label).width;
    const labelY = by > 13 ? by : by + bh + 13;
    ctx.fillStyle = airway;
    ctx.fillRect(bx - 0.75, labelY - 12, textWidth + 6, 12);
    ctx.fillStyle = paper;
    ctx.fillText(label, bx + 2.25, labelY - 0.5);
    det.keypoints.forEach(([u, v], k) => {
      const visibility = det.visibility[k];
      if (visibility === 0) return;
      ctx.beginPath();
      ctx.arc(u * width, v * height, KEYPOINT_RADIUS, 0, Math.PI * 2);
      if (visibility === 2) {
        ctx.fillStyle = airway;
        ctx.fill();
      } else {
        ctx.strokeStyle = airway;
        ctx.stroke();
      }
    });
  }
  return detections.length;
}
