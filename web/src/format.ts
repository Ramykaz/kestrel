import type { FlightSummary } from "./data/types";

export function clock(seconds: number): string {
  const safe = Math.max(0, seconds);
  const minutes = Math.floor(safe / 60);
  const rest = safe - minutes * 60;
  return `${minutes}:${rest.toFixed(1).padStart(4, "0")}`;
}

export function percent(share: number): string {
  return `${Math.round(share * 100)}%`;
}

const CRASH_TARGETS: Record<string, string> = {
  gate_frame: "a gate frame",
  gate_post: "a gate post",
  ground: "the ground",
  object: "an object",
};

/** How a flight ended, in a few words: "finished, 21.4 s" or "crashed at gate 2". */
export function outcome(summary: Pick<FlightSummary, "end" | "gates_passed" | "gate_count" | "lap_time_s" | "end_time_s">): string {
  if (summary.end === "finished" && summary.lap_time_s !== null) return `finished, ${summary.lap_time_s.toFixed(1)} s`;
  const gate = Math.min(summary.gates_passed + 1, summary.gate_count);
  if (summary.end === "crash") return `crashed at gate ${gate}`;
  return `timed out before gate ${gate}`;
}

export function crashTarget(kind: string | null): string {
  return kind ? (CRASH_TARGETS[kind] ?? kind) : "an unknown object";
}

export function escapeHtml(text: string): string {
  return text.replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]!);
}
