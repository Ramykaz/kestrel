import "./rounds.css";

import { DataError, loadBundle } from "../data/load";
import type { Bundle, FlightMetrics, Round } from "../data/types";
import { escapeHtml, percent } from "../format";
import type { MountedView } from "../router";

let CHART_W = 720;
const CHART_H = 220;
const PAD = { left: 44, right: 16, top: 16, bottom: 30 };

function x(index: number, count: number): number {
  const inner = CHART_W - PAD.left - PAD.right;
  return PAD.left + (count === 1 ? inner / 2 : (index / (count - 1)) * inner);
}

function y(share: number): number {
  return PAD.top + (1 - share) * (CHART_H - PAD.top - PAD.bottom);
}

function lineChart(bundle: Bundle): string {
  const rounds = bundle.rounds;
  const series: { key: "completion_rate" | "night_completion"; name: string; className: string }[] = [
    { key: "completion_rate", name: "Completion, tech courses", className: "series-tech" },
    { key: "night_completion", name: "Completion, night courses", className: "series-night" },
  ];
  const gridLines = [0, 0.25, 0.5, 0.75, 1]
    .map((v) => `<line class="grid" x1="${PAD.left}" x2="${CHART_W - PAD.right}" y1="${y(v)}" y2="${y(v)}" /><text class="axis" x="${PAD.left - 8}" y="${y(v) + 4}" text-anchor="end">${percent(v)}</text>`)
    .join("");
  const xLabels = rounds.map((r, i) => `<text class="axis" x="${x(i, rounds.length)}" y="${CHART_H - 8}" text-anchor="middle">${r.round}</text>`).join("");
  const lines = series
    .map(({ key, className }) => {
      const accepted = rounds.map((r, i) => ({ r, i })).filter(({ r }) => r.status !== "rejected" && r.metrics[key] !== null);
      const path = accepted.map(({ r, i }, n) => `${n === 0 ? "M" : "L"}${x(i, rounds.length)},${y(r.metrics[key] as number)}`).join(" ");
      const points = rounds
        .map((r, i) => {
          const value = r.metrics[key];
          if (value === null) return "";
          const hollow = r.status === "rejected" ? " hollow" : "";
          return `<circle class="point${hollow}" cx="${x(i, rounds.length)}" cy="${y(value)}" r="4"><title>Round ${r.round}: ${percent(value)}. ${escapeHtml(r.card?.hypothesis ?? r.label)}</title></circle>`;
        })
        .join("");
      return `<g class="${className}"><path d="${path}" />${points}</g>`;
    })
    .join("");
  const legend = series.map(({ name, className }) => `<span class="${className}"><i></i>${name}</span>`).join("");
  return `<figure class="rounds-chart panel">
    <figcaption><h2>Completion across rounds</h2><div class="chart-legend">${legend}<span class="legend-rejected"><i></i>Rejected round</span></div></figcaption>
    <svg viewBox="0 0 ${CHART_W} ${CHART_H}" role="img" aria-label="Completion rate per round on tech and night courses">${gridLines}${xLabels}${lines}</svg>
  </figure>`;
}

function sparkline(rounds: Round[], upTo: number): string {
  const values = rounds.slice(0, upTo + 1).map((r) => r.metrics.completion_rate);
  if (values.length < 2) return `<svg class="sparkline" viewBox="0 0 64 18" aria-hidden="true"></svg>`;
  const path = values.map((v, i) => `${i === 0 ? "M" : "L"}${(i / (values.length - 1)) * 62 + 1},${17 - v * 16}`).join(" ");
  return `<svg class="sparkline" viewBox="0 0 64 18" aria-hidden="true"><path d="${path}" /></svg>`;
}

function previousAccepted(rounds: Round[], index: number): Round | undefined {
  for (let i = index - 1; i >= 0; i--) if (rounds[i].status !== "rejected") return rounds[i];
  return undefined;
}

function metricsTable(round: Round, previous: Round | undefined): string {
  const rows: [string, (m: FlightMetrics) => string][] = [
    ["Completion", (m) => percent(m.completion_rate)],
    ["Gates per course", (m) => m.gates_per_course.toFixed(2)],
    ["Median lap", (m) => (m.lap_time_s === null ? "none" : `${m.lap_time_s.toFixed(1)} s`)],
    ["Crash rate", (m) => percent(m.crash_rate)],
    ["Night completion", (m) => (m.night_completion === null ? "not flown" : percent(m.night_completion))],
  ];
  const head = `<tr><th scope="col">Metric</th><th scope="col">Round ${round.round}</th>${previous ? `<th scope="col">Round ${previous.round}</th>` : ""}${round.ablation ? `<th scope="col">New detector, old pilot</th>` : ""}</tr>`;
  const body = rows
    .map(([name, format]) => `<tr><th scope="row">${name}</th><td class="num">${format(round.metrics)}</td>${previous ? `<td class="num">${format(previous.metrics)}</td>` : ""}${round.ablation ? `<td class="num">${format(round.ablation)}</td>` : ""}</tr>`)
    .join("");
  return `<table class="metrics-table"><thead>${head}</thead><tbody>${body}</tbody></table>`;
}

function roundDetail(bundle: Bundle, index: number): string {
  const round = bundle.rounds[index];
  const fixture = bundle.source === "reference_fixtures";
  const missing = (what: string) => `<p class="detail-missing">${fixture ? `Fixture rounds have no ${what}.` : `No ${what} was recorded for this round.`}</p>`;
  const card = round.card
    ? `<dl class="card"><div><dt>Hypothesis</dt><dd>${escapeHtml(round.card.hypothesis)}</dd></div>
       <div><dt>Evidence</dt><dd>${round.card.evidence.map(escapeHtml).join("<br />")}</dd></div>
       ${Object.entries(round.card.changes).map(([k, v]) => `<div><dt>Change: ${escapeHtml(k)}</dt><dd>${escapeHtml(v)}</dd></div>`).join("")}
       <div><dt>Expected effect</dt><dd>${escapeHtml(round.card.expected_effect)}</dd></div>
       <div><dt>Stop if</dt><dd>${escapeHtml(round.card.stop_if)}</dd></div></dl>
       <details class="raw-json"><summary>Show JSON</summary><pre>${escapeHtml(JSON.stringify(round.card, null, 2))}</pre></details>`
    : missing("experiment card");
  const report = round.report
    ? `<p>${escapeHtml(round.report.summary)}</p><ul class="claims">${round.report.claims
        .map((c) => `<li data-passed="${c.passed}">${c.passed ? escapeHtml(c.text) : `<s>${escapeHtml(c.text)}</s> <span class="claim-correct">Measured: ${escapeHtml(c.correct_value ?? "unknown")}</span>`}</li>`)
        .join("")}</ul>`
    : missing("verified report");
  return `<div class="round-detail">
    <section><h3>Experiment card</h3>${card}</section>
    <section><h3>Metrics</h3>${metricsTable(round, previousAccepted(bundle.rounds, index))}</section>
    <section><h3>Verified report</h3>${report}</section>
    <section><h3>Task board and budgets</h3>${round.budgets.length ? "" : missing("task board")}</section>
  </div>`;
}

export async function mount(root: HTMLElement): Promise<MountedView> {
  root.innerHTML = `<div class="page rounds"><header class="page-header"><h1>Rounds</h1>
    <p>Each round, the planner proposes one change, the agents carry it out, and the evaluator flies the 30 held-out courses. A round is promoted only if it beats the best round so far: more courses finished, or as many finished with more gates passed.</p></header>
    <div class="rounds-body"><p class="label">Loading the round history</p></div></div>`;
  const body = root.querySelector<HTMLElement>(".rounds-body")!;
  let bundle: Bundle;
  try {
    bundle = await loadBundle();
  } catch (error) {
    body.innerHTML = `<div class="state state-error"><h2>The round history did not load</h2><p>${error instanceof DataError ? "Check your connection and reload the page." : escapeHtml(String(error))}</p></div>`;
    return { unmount: () => undefined };
  }
  if (bundle.rounds.length === 0) {
    body.innerHTML = `<div class="state"><h2>No rounds have run yet</h2><p>Round 0 is the baseline: the first detector and pilot, flown on the held-out courses before any agent changes anything. It appears here once it has been flown.</p></div>`;
    return { unmount: () => undefined };
  }
  const rows = bundle.rounds
    .map((r, i) => {
      const m = r.metrics;
      return `<li class="round-row" data-status="${r.status}">
        <details>
          <summary>
            <span class="row-number num">${r.round}</span>
            <span class="row-text">${escapeHtml(r.card?.hypothesis ?? r.label)}</span>
            <span class="row-result num">${percent(m.completion_rate)} finished</span>
            <span class="row-status">${r.status === "rejected" ? "Rejected" : r.status === "promoted" ? "Promoted" : "Baseline"}</span>
            ${sparkline(bundle.rounds, i)}
          </summary>
          ${roundDetail(bundle, i)}
        </details>
      </li>`;
    })
    .reverse()
    .join("");
  // Draw at the real width so axis text stays 12 px on narrow screens.
  CHART_W = Math.max(320, Math.min(900, body.clientWidth - 40));
  body.innerHTML = `${lineChart(bundle)}<ol class="round-list">${rows}</ol>`;
  return { unmount: () => undefined };
}
