import "./flight.css";

import { frameAt, type Flight } from "../../data/flight";
import { DataError, bestRound, loadBundle, loadCourse, loadFlight, suiteSeeds } from "../../data/load";
import { PILOT_STATE_NAMES, type Bundle, type Course, type FlightSummary, type PilotState, type Round } from "../../data/types";
import { clock, crashCause, crashTarget, escapeHtml, outcome, percent } from "../../format";
import { updateQuery, type MountedView } from "../../router";
import { onThemeChange } from "../../theme";
import { FlightChart } from "./chart";
import { drawDetections } from "./inset";
import type { FlightScene, GateLabel } from "./scene";

type Camera = "chase" | "onboard" | "top";
const CAMERAS: Camera[] = ["chase", "onboard", "top"];
const INTRO_SECONDS = 2.5;
const INTRO_KEY = "kestrel-intro-seen";
// Below this width the onboard inset would cover most of the stage; the Onboard camera shows it instead.
const narrowScreen = window.matchMedia("(max-width: 520px)");

interface State {
  roundIndex: number;
  courseIndex: number;
  camera: Camera;
  time: number;
  playing: boolean;
  ghost: boolean;
}

function webglAvailable(): boolean {
  try {
    const canvas = document.createElement("canvas");
    return Boolean(canvas.getContext("webgl2") ?? canvas.getContext("webgl"));
  } catch {
    return false;
  }
}

function reducedMotion(): boolean {
  return window.matchMedia("(prefers-reduced-motion: reduce)").matches;
}

function introAlreadySeen(): boolean {
  try {
    return sessionStorage.getItem(INTRO_KEY) === "1";
  } catch {
    return false;
  }
}

function markIntroSeen(): void {
  try {
    sessionStorage.setItem(INTRO_KEY, "1");
  } catch {
    // Without session storage the intro plays on every visit to this view.
  }
}

const STATUS_TEXT: Record<Round["status"], string> = {
  baseline: "Baseline",
  promoted: "Promoted",
  rejected: "Rejected",
};

const TEMPLATE = `
<section class="flight" aria-label="Flight replay">
  <div class="stage" data-camera="chase">
    <canvas class="stage-canvas" aria-hidden="true"></canvas>
    <div class="stage-chart"></div>
    <div class="gate-labels" aria-hidden="true"></div>
    <figure class="inset" hidden>
      <canvas class="inset-overlay" aria-hidden="true"></canvas>
      <figcaption class="inset-caption"></figcaption>
    </figure>
    <div class="stage-legend" aria-hidden="true">
      <span><svg viewBox="0 0 22 14"><rect x="7" y="3" width="8" height="8" /><line x1="7" y1="7" x2="2" y2="7" /></svg>Gate, tick on the side to fly in from</span>
      <span><svg viewBox="0 0 22 14"><line class="legend-route" x1="1" y1="7" x2="21" y2="7" /></svg>Flight path</span>
      <span class="legend-ghost-item" hidden><svg viewBox="0 0 22 14"><line class="legend-ghost" x1="1" y1="7" x2="21" y2="7" /></svg>Round 0</span>
      <span><svg viewBox="0 0 22 14"><circle class="legend-target" cx="11" cy="7" r="4" /></svg>Pilot target</span>
    </div>
    <p class="stage-status" role="status"></p>
    <p class="visually-hidden flight-summary" aria-live="polite"></p>
  </div>

  <aside class="flight-panel" aria-label="Round details">
    <div class="panel-round">
      <p class="label">Training round</p>
      <div class="panel-round-line">
        <h1 class="round-title num">Round</h1>
        <span class="status-chip"></span>
      </div>
      <p class="round-headline"></p>
    </div>
    <dl class="round-stats"></dl>
    <div class="panel-controls">
      <label class="control">
        <span class="label">Course</span>
        <select class="select course-select"></select>
      </label>
      <div class="control">
        <span class="label" id="camera-label">Camera</span>
        <div class="segmented camera-switch" role="group" aria-labelledby="camera-label">
          <button type="button" data-camera="chase">Chase</button>
          <button type="button" data-camera="onboard">Onboard</button>
          <button type="button" data-camera="top">Top</button>
        </div>
      </div>
      <label class="switch ghost-switch">
        <input type="checkbox" class="ghost-toggle" />
        <span>Compare with round 0</span>
      </label>
    </div>
    <div class="panel-why">
      <h2 class="label">Why it changed</h2>
      <p class="why-text"></p>
    </div>
  </aside>

  <div class="dock">
  <div class="scrubber">
    <div class="scrubber-track" role="slider" tabindex="0" aria-label="Training round" aria-valuemin="0">
      <div class="scrubber-rail"></div>
      <div class="scrubber-ticks"></div>
    </div>
    <div class="scrubber-ends">
      <span class="scrubber-end scrubber-start"></span>
      <span class="scrubber-end scrubber-current"></span>
    </div>
  </div>

  <div class="player">
    <button type="button" class="button button-primary play-button">
      <svg class="icon-play" viewBox="0 0 12 12" aria-hidden="true"><path d="M2.5 1.5v9l8-4.5z" /></svg>
      <svg class="icon-pause" viewBox="0 0 12 12" aria-hidden="true"><path d="M2.5 1.5h2.6v9H2.5zM6.9 1.5h2.6v9H6.9z" /></svg>
      <span class="play-text">Play flight</span>
    </button>
    <span class="player-clock num"><span class="clock-now">0:00.0</span><span class="clock-sep"> / </span><span class="clock-total">0:00.0</span></span>
    <div class="timeline">
      <input type="range" class="timeline-input" min="0" max="1" step="0.0333" value="0" aria-label="Flight time in seconds" />
      <div class="timeline-marks" aria-hidden="true"></div>
    </div>
    <dl class="readouts">
      <div><dt>Speed</dt><dd class="num readout-speed">0.0 m/s</dd></div>
      <div><dt>State</dt><dd class="readout-state">search</dd></div>
      <div><dt>Gate</dt><dd class="num readout-gate">0/0</dd></div>
    </dl>
  </div>
  </div>
</section>
`;

export async function mount(root: HTMLElement, params: URLSearchParams): Promise<MountedView> {
  root.innerHTML = TEMPLATE;
  const q = <T extends Element>(selector: string) => root.querySelector<T>(selector)!;
  const stage = q<HTMLDivElement>(".stage");
  const canvas = q<HTMLCanvasElement>(".stage-canvas");
  const chartHost = q<HTMLDivElement>(".stage-chart");
  const gateLabelHost = q<HTMLDivElement>(".gate-labels");
  const inset = q<HTMLElement>(".inset");
  const insetOverlay = q<HTMLCanvasElement>(".inset-overlay");
  const insetCaption = q<HTMLElement>(".inset-caption");
  const status = q<HTMLParagraphElement>(".stage-status");
  const summaryRegion = q<HTMLParagraphElement>(".flight-summary");
  const courseSelect = q<HTMLSelectElement>(".course-select");
  const ghostToggle = q<HTMLInputElement>(".ghost-toggle");
  const playButton = q<HTMLButtonElement>(".play-button");
  const timeline = q<HTMLInputElement>(".timeline-input");
  const timelineMarks = q<HTMLDivElement>(".timeline-marks");
  const scrubberTrack = q<HTMLDivElement>(".scrubber-track");
  const scrubberTicks = q<HTMLDivElement>(".scrubber-ticks");

  let disposed = false;
  const cleanups: (() => void)[] = [];
  const chart = new FlightChart();
  chartHost.append(chart.svg);
  const hasWebgl = webglAvailable();
  let scene: FlightScene | null = null;

  const showStatus = (text: string, kind: "loading" | "error" | "" = "") => {
    status.textContent = text;
    status.dataset.kind = kind;
    status.hidden = text === "";
  };
  showStatus("Loading flight data", "loading");

  let bundle: Bundle;
  try {
    bundle = await loadBundle();
  } catch (error) {
    showStatus(error instanceof DataError ? "The round history did not load. Check your connection and reload the page." : String(error), "error");
    return { unmount: () => undefined };
  }
  if (disposed) return { unmount: () => undefined };

  const seeds = suiteSeeds(bundle);
  if (bundle.rounds.length === 0) {
    root.querySelector(".flight")!.classList.add("flight-empty");
    showStatus("");
    stage.innerHTML = `<div class="stage-chart flight-empty-chart"></div><div class="state flight-empty-state"><h2>Round 0 has not flown yet</h2>
      <p>The pilot and the detector are still being built. The chart shows held-out course 1 of ${seeds.length}:
      the drone will meet it for the first time in round 0, knowing only what its camera shows.</p></div>`;
    try {
      const firstCourse = await loadCourse(seeds[0]);
      const preview = new FlightChart();
      preview.draw(firstCourse, null, null);
      stage.querySelector(".flight-empty-chart")!.append(preview.svg);
    } catch (error) {
      console.error(error);
    }
    return { unmount: () => undefined };
  }

  const roundByNumber = new Map(bundle.rounds.map((r, i) => [r.round, i]));
  const best = bestRound(bundle) ?? bundle.rounds[bundle.rounds.length - 1];
  const defaultCourse = (round: Round): number => {
    let pick = 0;
    let pickScore = -Infinity;
    seeds.forEach((seed, index) => {
      const flight = round.flights[String(seed)];
      if (!flight) return;
      const score = flight.end === "finished" && flight.lap_time_s !== null ? 1000 - flight.lap_time_s : flight.gates_passed;
      if (score > pickScore && index < bundle.eval_suite.tech.length) {
        pick = index;
        pickScore = score;
      }
    });
    return pick;
  };

  const requestedRound = Number(params.get("round"));
  const roundIndex = params.has("round") && roundByNumber.has(requestedRound) ? roundByNumber.get(requestedRound)! : bundle.rounds.indexOf(best);
  const requestedCourse = Number(params.get("course")) - 1;
  const requestedCamera = params.get("cam") as Camera | null;
  const state: State = {
    roundIndex,
    courseIndex: Number.isInteger(requestedCourse) && requestedCourse >= 0 && requestedCourse < seeds.length ? requestedCourse : defaultCourse(bundle.rounds[roundIndex]),
    camera: requestedCamera && CAMERAS.includes(requestedCamera) ? requestedCamera : "chase",
    time: Math.max(0, Number(params.get("t")) || 0),
    playing: false,
    ghost: params.get("ghost") === "1",
  };
  if (!hasWebgl) state.camera = "top";
  const linkedMoment = params.has("t");

  let course: Course | null = null;
  let flight: Flight | null = null;
  let ghostFlight: Flight | null = null;
  let loadToken = 0;
  let introStart: number | null = null;
  let introShare = 1;
  let needsRender = true;
  let lastFrameTime = performance.now();

  const round = () => bundle.rounds[state.roundIndex];
  const seed = () => seeds[state.courseIndex];
  const summaryFor = (r: Round): FlightSummary | undefined => r.flights[String(seed())];
  const courseName = (index: number) => `Course ${index + 1} (${index < bundle.eval_suite.tech.length ? "tech" : "night"})`;

  courseSelect.innerHTML = seeds.map((_, i) => `<option value="${i}">${courseName(i)}</option>`).join("");

  if (!hasWebgl) {
    root.querySelectorAll<HTMLButtonElement>(".camera-switch button").forEach((button) => {
      if (button.dataset.camera !== "top") {
        button.disabled = true;
        button.title = "The 3D views need WebGL, which this browser does not provide";
      }
    });
  }

  const writeUrl = () => {
    updateQuery({
      round: round().round,
      course: state.courseIndex + 1,
      cam: state.camera,
      t: state.time > 0 ? state.time.toFixed(1) : null,
      ghost: state.ghost ? 1 : null,
    });
  };

  const renderPanel = () => {
    const r = round();
    const m = r.metrics;
    const finished = Math.round(m.completion_rate * bundle.eval_suite.tech.length);
    q(".round-title").textContent = `Round ${r.round}`;
    const chip = q<HTMLElement>(".status-chip");
    chip.textContent = STATUS_TEXT[r.status];
    chip.dataset.status = r.status;
    q(".round-headline").textContent = `${finished} of ${bundle.eval_suite.tech.length} courses finished`;
    const stats: [string, string][] = [
      ["Gates per course", m.gates_per_course.toFixed(1)],
      ["Median lap", m.lap_time_s === null ? "none finished" : `${m.lap_time_s.toFixed(1)} s`],
      ["Crash rate", percent(m.crash_rate)],
      ["Night courses", m.night_completion === null ? "not flown" : `${percent(m.night_completion)} finished`],
    ];
    q(".round-stats").innerHTML = stats.map(([k, v]) => `<div><dt>${k}</dt><dd class="num">${escapeHtml(v)}</dd></div>`).join("");
    const why = q(".why-text");
    if (r.report) why.textContent = r.report.summary;
    else if (bundle.source === "reference_fixtures") why.textContent = `${r.label}. Fixture rounds have no experiment card or verified report.`;
    else why.textContent = `${r.label}. No verified agent report for this round.`;
    courseSelect.value = String(state.courseIndex);
    ghostToggle.checked = state.ghost;
    q<HTMLElement>(".ghost-switch").hidden = state.roundIndex === 0;
    root.querySelectorAll<HTMLButtonElement>(".camera-switch button").forEach((b) => b.setAttribute("aria-pressed", String(b.dataset.camera === state.camera)));
  };

  const renderScrubber = () => {
    const rounds = bundle.rounds;
    scrubberTrack.setAttribute("aria-valuemax", String(rounds.length - 1));
    scrubberTrack.setAttribute("aria-valuenow", String(state.roundIndex));
    const current = round();
    const currentSummary = summaryFor(current);
    scrubberTrack.setAttribute(
      "aria-valuetext",
      `Round ${current.round}, ${STATUS_TEXT[current.status].toLowerCase()}${currentSummary ? `, ${outcome(currentSummary)}` : ""}`,
    );
    scrubberTicks.innerHTML = rounds
      .map((r, i) => {
        const s = summaryFor(r);
        const mark = s ? `<span class="tick-outcome" data-end="${s.end}"></span>` : "";
        const position = rounds.length === 1 ? 50 : (i / (rounds.length - 1)) * 100;
        return `<span class="tick" data-index="${i}" data-status="${r.status}" ${i === state.roundIndex ? 'data-selected="true"' : ""} style="left:${position}%">
          <span class="tick-mark"></span>${mark}<span class="tick-number num">${r.round}</span></span>`;
      })
      .join("");
    const first = rounds[0];
    const firstSummary = summaryFor(first);
    q(".scrubber-start").innerHTML = `<span class="num">Round ${first.round}</span> ${firstSummary ? escapeHtml(outcome(firstSummary)) : "did not fly this course"}`;
    q(".scrubber-current").innerHTML = `<span class="num">Round ${current.round}</span> ${currentSummary ? escapeHtml(outcome(currentSummary)) : "did not fly this course"}`;
  };

  const renderTimelineMarks = () => {
    if (!flight) return;
    const duration = Math.max(flight.duration, 0.001);
    const marks = flight.header.gate_pass_times_s.map((t, i) => `<span class="mark-gate" style="left:${(t / duration) * 100}%" title="Gate ${i + 1}"></span>`);
    if (flight.header.end === "crash") marks.push(`<span class="mark-crash" style="left:100%"></span>`);
    timelineMarks.innerHTML = marks.join("");
  };

  const updateReadouts = () => {
    if (!flight) return;
    const frame = frameAt(flight, state.time);
    const prev = Math.max(0, frame - 3);
    const next = Math.min(flight.frames - 1, frame + 3);
    const dt = (next - prev) / flight.hz;
    const p = flight.position;
    const speed = dt > 0 ? Math.hypot(p[next * 3] - p[prev * 3], p[next * 3 + 1] - p[prev * 3 + 1], p[next * 3 + 2] - p[prev * 3 + 2]) / dt : 0;
    const passed = flight.header.gate_pass_times_s.filter((t) => t <= state.time + 1e-6).length;
    q(".readout-speed").textContent = `${speed.toFixed(1)} m/s`;
    q(".readout-state").textContent = PILOT_STATE_NAMES[flight.state[frame] as PilotState] ?? "unknown";
    q(".readout-gate").textContent = `${passed}/${flight.header.gate_count}`;
    q(".clock-now").textContent = clock(state.time);
    timeline.value = String(state.time);
  };

  const updatePlayButton = () => {
    playButton.dataset.playing = String(state.playing);
    const atEnd = flight !== null && state.time >= flight.duration - 1e-3;
    q(".play-text").textContent = state.playing ? "Pause" : atEnd ? "Replay flight" : "Play flight";
  };

  const insetRect = () => {
    const width = stage.clientWidth;
    const insetWidth = Math.round(Math.min(320, Math.max(150, width * 0.27)));
    const insetHeight = Math.round((insetWidth * 3) / 4);
    const margin = width < 600 ? 10 : 16;
    return { x: width - insetWidth - margin, y: stage.clientHeight - insetHeight - margin, width: insetWidth, height: insetHeight };
  };

  const placeGateLabels = (labels: GateLabel[]) => {
    if (gateLabelHost.childElementCount !== labels.length) {
      gateLabelHost.innerHTML = labels.map((l) => `<span class="gate-label num">${l.index + 1}</span>`).join("");
    }
    labels.forEach((label, i) => {
      const node = gateLabelHost.children[i] as HTMLElement;
      node.style.transform = `translate(${label.x.toFixed(1)}px, ${label.y.toFixed(1)}px)`;
      node.hidden = !label.visible;
    });
  };

  const draw = (smooth: boolean) => {
    if (!flight || !course) return;
    stage.dataset.camera = state.camera;
    const showInset = state.camera === "chase" && scene !== null && !narrowScreen.matches;
    inset.hidden = !showInset;
    if (state.camera === "top" || !scene) {
      chart.setRouteReveal(introShare);
      chart.update(state.time);
      gateLabelHost.hidden = true;
    } else {
      scene.setRouteReveal(introShare);
      const rect = showInset ? insetRect() : null;
      if (rect) {
        inset.style.setProperty("--inset-x", `${rect.x}px`);
        inset.style.setProperty("--inset-y", `${rect.y}px`);
        inset.style.setProperty("--inset-w", `${rect.width}px`);
        inset.style.setProperty("--inset-h", `${rect.height}px`);
      }
      gateLabelHost.hidden = false;
      placeGateLabels(scene.render(flight, state.time, state.camera === "onboard" ? "onboard" : "chase", rect, smooth));
      if (rect) drawDetections(insetOverlay, flight, frameAt(flight, state.time));
    }
    updateReadouts();
  };

  const loop = (now: number) => {
    if (disposed) return;
    // Real elapsed time keeps playback at real speed on slow devices; long gaps (a hidden tab) are dropped.
    const elapsed = (now - lastFrameTime) / 1000;
    const dt = elapsed > 0.5 ? 0 : elapsed;
    lastFrameTime = now;
    if (introStart !== null) {
      const share = Math.min(1, (now - introStart) / (INTRO_SECONDS * 1000));
      introShare = 1 - (1 - share) ** 3;
      needsRender = true;
      if (share >= 1) {
        introStart = null;
        introShare = 1;
        markIntroSeen();
        setPlaying(true);
      }
    }
    if (state.playing && flight) {
      state.time = Math.min(flight.duration, state.time + dt);
      if (state.time >= flight.duration) {
        setPlaying(false);
        writeUrl();
      }
      needsRender = true;
    }
    if (needsRender) {
      draw(state.playing);
      needsRender = false;
    }
    requestAnimationFrame(loop);
  };

  const setPlaying = (playing: boolean) => {
    if (playing && flight && state.time >= flight.duration - 1e-3) state.time = 0;
    state.playing = playing;
    updatePlayButton();
    if (!playing) writeUrl();
  };

  const describe = () => {
    const s = summaryFor(round());
    if (!s || !flight) return;
    const gates = `${s.gates_passed} of ${s.gate_count} gates`;
    const cause = crashCause(s.crash_cause);
    const crash = `then crashed into ${crashTarget(s.crash_body_kind)} after ${s.end_time_s.toFixed(1)} s${cause ? ` (${cause})` : ""}`;
    const end = s.end === "crash" ? crash : s.end === "timeout" ? "then ran out of time" : `in ${s.lap_time_s?.toFixed(1)} s`;
    summaryRegion.textContent = `Round ${round().round}, ${courseName(state.courseIndex)}: passed ${gates} ${end}.`;
    stage.setAttribute("aria-label", summaryRegion.textContent);
    insetCaption.textContent =
      flight.header.detections_source === "ground_truth" ? "Onboard view rebuilt from the log, ground-truth boxes" : "Onboard view rebuilt from the log, logged detections";
  };

  const loadSelection = async () => {
    const token = ++loadToken;
    const r = round();
    const summary = summaryFor(r);
    renderPanel();
    renderScrubber();
    if (!summary) {
      showStatus(`Round ${r.round} did not fly ${courseName(state.courseIndex).toLowerCase()}.`, "error");
      return;
    }
    showStatus(flight ? "" : "Loading flight data", flight ? "" : "loading");
    try {
      const first = bundle.rounds[0];
      const ghostSummary = state.ghost && state.roundIndex > 0 ? summaryFor(first) : undefined;
      const [nextCourse, nextFlight, nextGhost] = await Promise.all([
        loadCourse(seed()),
        loadFlight(summary.file, `Flight data for round ${r.round}`),
        ghostSummary ? loadFlight(ghostSummary.file, `Flight data for round ${first.round}`) : Promise.resolve(null),
      ]);
      if (token !== loadToken || disposed) return;
      const courseChanged = course?.seed !== nextCourse.seed;
      course = nextCourse;
      flight = nextFlight;
      ghostFlight = nextGhost;
      state.time = Math.min(state.time, flight.duration);
      timeline.max = String(flight.duration);
      q(".clock-total").textContent = clock(flight.duration);
      q<HTMLElement>(".legend-ghost-item").hidden = !ghostFlight;
      q<HTMLElement>(".legend-ghost-item").lastChild!.textContent = `Round ${first.round}`;
      chart.draw(course, flight, ghostFlight);
      if (scene) {
        if (courseChanged) scene.setCourse(course);
        scene.setFlight(flight);
        scene.setGhost(ghostFlight);
      }
      renderTimelineMarks();
      updatePlayButton();
      describe();
      showStatus("");
      needsRender = true;
      for (const neighbor of [state.roundIndex - 1, state.roundIndex + 1]) {
        const s = bundle.rounds[neighbor] ? summaryFor(bundle.rounds[neighbor]) : undefined;
        if (s) void loadFlight(s.file, `Flight data for round ${bundle.rounds[neighbor].round}`).catch(() => undefined);
      }
    } catch (error) {
      if (token !== loadToken || disposed) return;
      console.error(error);
      showStatus(`Flight data for round ${r.round} did not load. Check your connection and reload the page.`, "error");
    }
  };

  const selectRound = (index: number) => {
    const clamped = Math.max(0, Math.min(bundle.rounds.length - 1, index));
    if (clamped === state.roundIndex) return;
    state.roundIndex = clamped;
    introStart = null;
    introShare = 1;
    void loadSelection().then(writeUrl);
  };

  const seek = (time: number) => {
    if (!flight) return;
    state.time = Math.max(0, Math.min(flight.duration, time));
    introStart = null;
    introShare = 1;
    needsRender = true;
    updatePlayButton();
  };

  const setCamera = (camera: Camera) => {
    if (camera !== "top" && !scene) return;
    state.camera = camera;
    renderPanel();
    needsRender = true;
    writeUrl();
  };

  playButton.addEventListener("click", () => {
    introStart = null;
    introShare = 1;
    setPlaying(!state.playing);
  });
  timeline.addEventListener("input", () => {
    seek(Number(timeline.value));
    if (state.playing) setPlaying(false);
  });
  timeline.addEventListener("change", writeUrl);
  courseSelect.addEventListener("change", () => {
    state.courseIndex = Number(courseSelect.value);
    state.time = 0;
    setPlaying(false);
    void loadSelection().then(writeUrl);
  });
  ghostToggle.addEventListener("change", () => {
    state.ghost = ghostToggle.checked;
    void loadSelection().then(writeUrl);
  });
  root.querySelectorAll<HTMLButtonElement>(".camera-switch button").forEach((button) =>
    button.addEventListener("click", () => setCamera(button.dataset.camera as Camera)),
  );

  const tickIndexAt = (clientX: number) => {
    const rect = scrubberTicks.getBoundingClientRect();
    const share = Math.max(0, Math.min(1, (clientX - rect.left) / rect.width));
    return Math.round(share * (bundle.rounds.length - 1));
  };
  scrubberTrack.addEventListener("pointerdown", (event) => {
    scrubberTrack.setPointerCapture(event.pointerId);
    scrubberTrack.focus({ preventScroll: true });
    selectRound(tickIndexAt(event.clientX));
  });
  scrubberTrack.addEventListener("pointermove", (event) => {
    if (scrubberTrack.hasPointerCapture(event.pointerId)) selectRound(tickIndexAt(event.clientX));
  });
  scrubberTrack.addEventListener("keydown", (event) => {
    const moves: Record<string, number> = { ArrowLeft: -1, ArrowDown: -1, ArrowRight: 1, ArrowUp: 1 };
    if (event.key in moves) selectRound(state.roundIndex + moves[event.key]);
    else if (event.key === "Home") selectRound(0);
    else if (event.key === "End") selectRound(bundle.rounds.length - 1);
    else return;
    event.preventDefault();
    event.stopPropagation();
  });

  const onKey = (event: KeyboardEvent) => {
    if (event.metaKey || event.ctrlKey || event.altKey) return;
    const target = event.target as HTMLElement;
    if (target.closest("select, textarea, input:not([type=range]):not([type=checkbox])")) return;
    if (event.key === " " && !target.closest("button, a")) {
      event.preventDefault();
      introStart = null;
      introShare = 1;
      setPlaying(!state.playing);
    } else if ((event.key === "ArrowLeft" || event.key === "ArrowRight") && !target.closest("input[type=range]")) {
      event.preventDefault();
      seek(state.time + (event.key === "ArrowLeft" ? -1 : 1));
      writeUrl();
    } else if (event.key === "[") {
      selectRound(state.roundIndex - 1);
    } else if (event.key === "]") {
      selectRound(state.roundIndex + 1);
    } else if (event.key === "c" || event.key === "C") {
      const available = scene ? CAMERAS : (["top"] as Camera[]);
      setCamera(available[(available.indexOf(state.camera) + 1) % available.length]);
    }
  };
  document.addEventListener("keydown", onKey);
  cleanups.push(() => document.removeEventListener("keydown", onKey));

  const resizeObserver = new ResizeObserver(() => {
    scene?.resize(stage.clientWidth, stage.clientHeight);
    needsRender = true;
  });
  resizeObserver.observe(stage);
  cleanups.push(() => resizeObserver.disconnect());
  cleanups.push(
    onThemeChange(() => {
      scene?.applyTheme();
      needsRender = true;
    }),
  );

  if (hasWebgl) {
    try {
      const { FlightScene } = await import("./scene");
      if (disposed) return { unmount: () => undefined };
      scene = new FlightScene(canvas);
      scene.resize(stage.clientWidth, stage.clientHeight);
    } catch (error) {
      console.error(error);
      scene = null;
      state.camera = "top";
      root.querySelectorAll<HTMLButtonElement>(".camera-switch button").forEach((button) => {
        if (button.dataset.camera !== "top") button.disabled = true;
      });
    }
  }

  await loadSelection();
  if (scene && course && flight) {
    scene.setCourse(course);
    scene.setFlight(flight);
    scene.setGhost(ghostFlight);
  }
  writeUrl();
  if (!linkedMoment && !introAlreadySeen() && flight) {
    if (reducedMotion()) {
      introShare = 1;
      markIntroSeen();
    } else {
      introShare = 0;
      introStart = performance.now();
    }
  }
  requestAnimationFrame((now) => {
    lastFrameTime = now;
    loop(now);
  });

  return {
    unmount() {
      disposed = true;
      for (const cleanup of cleanups) cleanup();
      scene?.dispose();
    },
  };
}
