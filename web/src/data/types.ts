// Mirrors kestrel/export/web_bundle.py. Keep both in step.

export type FlightEnd = "finished" | "crash" | "timeout";
export type RoundStatus = "baseline" | "promoted" | "rejected";
export type BundleSource = "rounds" | "reference_fixtures";

export interface FlightSummary {
  file: string;
  end: FlightEnd;
  end_time_s: number;
  gates_passed: number;
  gate_count: number;
  lap_time_s: number | null;
  crash_body_kind: string | null;
  /** Ground-truth crash cause (SPEC 9.3); absent in fixture bundles. */
  crash_cause?: string | null;
}

export interface FlightMetrics {
  completion_rate: number;
  gates_per_course: number;
  lap_time_s: number | null;
  crash_rate: number;
  night_completion: number | null;
}

export interface ExperimentCard {
  round: number;
  hypothesis: string;
  evidence: string[];
  changes: Record<string, string>;
  expected_effect: string;
  stop_if: string;
}

export interface ReportClaim {
  text: string;
  passed: boolean;
  correct_value: string | null;
}

export interface VerifiedReport {
  summary: string;
  claims: ReportClaim[];
}

export interface BudgetUse {
  role: string;
  tokens_used: number;
  tokens_budget: number;
  gpu_minutes_used: number;
  gpu_minutes_budget: number;
}

export interface Round {
  round: number;
  status: RoundStatus;
  label: string;
  card: ExperimentCard | null;
  report: VerifiedReport | null;
  metrics: FlightMetrics;
  ablation: FlightMetrics | null;
  budgets: BudgetUse[];
  flights: Record<string, FlightSummary>;
}

export interface Bundle {
  schema: number;
  generated_at: string;
  source: BundleSource;
  eval_suite: { tech: number[]; night: number[] };
  rounds: Round[];
}

export interface CourseGate {
  center: [number, number, number];
  yaw: number;
}

export interface Course {
  seed: number;
  preset: string;
  lighting: "day" | "night";
  gates: CourseGate[];
  decoys: unknown[];
  start: { position: [number, number, number]; yaw: number };
}

export enum PilotState {
  Search = 0,
  Approach = 1,
  Commit = 2,
  Exit = 3,
}

export const PILOT_STATE_NAMES: Record<PilotState, string> = {
  [PilotState.Search]: "search",
  [PilotState.Approach]: "approach",
  [PilotState.Commit]: "commit",
  [PilotState.Exit]: "exit",
};
