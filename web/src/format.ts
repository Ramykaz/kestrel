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

const CRASH_CAUSES: Record<string, string> = {
  missed_gate: "the detector missed the gate",
  bad_pose: "the gate position estimate was off",
  late_turn: "turned too late",
  lost_track: "lost track of the gate",
  wrong_side: "aimed at a gate from behind",
  decoy_followed: "followed a decoy",
  unclassified: "cause not classified",
};

export function crashCause(cause: string | null | undefined): string | null {
  return cause ? (CRASH_CAUSES[cause] ?? cause) : null;
}

/** How a flight ended, in a few words: "finished, 21.4 s" or "crashed at gate 2, turned too late". */
export function outcome(summary: Pick<FlightSummary, "end" | "gates_passed" | "gate_count" | "lap_time_s" | "end_time_s" | "crash_cause">): string {
  if (summary.end === "finished" && summary.lap_time_s !== null) return `finished, ${summary.lap_time_s.toFixed(1)} s`;
  const gate = Math.min(summary.gates_passed + 1, summary.gate_count);
  const cause = crashCause(summary.crash_cause);
  if (summary.end === "crash") return cause ? `crashed at gate ${gate}, ${cause}` : `crashed at gate ${gate}`;
  return `timed out before gate ${gate}`;
}

export function crashTarget(kind: string | null): string {
  return kind ? (CRASH_TARGETS[kind] ?? kind) : "an unknown object";
}

export function escapeHtml(text: string): string {
  return text.replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]!);
}
