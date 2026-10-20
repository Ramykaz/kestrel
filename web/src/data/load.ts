import { decodeFlight, type Flight } from "./flight";
import type { Bundle, Course, Round } from "./types";

export const DATA_ROOT: string = import.meta.env.VITE_DATA_ROOT ?? "/data";

let bundleRequest: Promise<Bundle> | null = null;
const courseRequests = new Map<number, Promise<Course>>();
const flightRequests = new Map<string, Promise<Flight>>();

export class DataError extends Error {
  constructor(
    message: string,
    readonly what: string,
  ) {
    super(message);
  }
}

async function fetchOk(path: string, what: string): Promise<Response> {
  let response: Response;
  try {
    response = await fetch(`${DATA_ROOT}/${path}`);
  } catch (error) {
    throw new DataError(`network error loading ${path}: ${String(error)}`, what);
  }
  if (!response.ok) throw new DataError(`${path} returned HTTP ${response.status}`, what);
  return response;
}

export function loadBundle(): Promise<Bundle> {
  bundleRequest ??= fetchOk("rounds.json", "The round history").then((r) => r.json() as Promise<Bundle>);
  bundleRequest.catch(() => (bundleRequest = null));
  return bundleRequest;
}

export function loadCourse(seed: number): Promise<Course> {
  let request = courseRequests.get(seed);
  if (!request) {
    request = fetchOk(`courses/s${seed}.json`, `Course ${seed}`).then((r) => r.json() as Promise<Course>);
    request.catch(() => courseRequests.delete(seed));
    courseRequests.set(seed, request);
  }
  return request;
}

export function loadFlight(file: string, what: string): Promise<Flight> {
  let request = flightRequests.get(file);
  if (!request) {
    request = fetchOk(file, what)
      .then((r) => r.arrayBuffer())
      .then(decodeFlight);
    request.catch(() => flightRequests.delete(file));
    flightRequests.set(file, request);
  }
  return request;
}

/** Eval suite courses in display order: tech first, then night. Course numbers start at 1. */
export function suiteSeeds(bundle: Bundle): number[] {
  return [...bundle.eval_suite.tech, ...bundle.eval_suite.night];
}

export function bestRound(bundle: Bundle): Round | undefined {
  const accepted = bundle.rounds.filter((r) => r.status !== "rejected");
  return accepted[accepted.length - 1];
}
