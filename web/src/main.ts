import "./styles/tokens.css";
import "./styles/base.css";

import { loadBundle } from "./data/load";
import { Router } from "./router";
import { initThemeToggle } from "./theme";

initThemeToggle(document.querySelector<HTMLElement>(".theme-toggle")!);

const router = new Router(document.querySelector<HTMLElement>("#main")!, {
  flight: () => import("./views/flight/view"),
  rounds: () => import("./views/rounds"),
  lab: () => import("./views/lab"),
  crashes: () => import("./views/crashes"),
  notes: () => import("./views/notes"),
});
void router.render();

loadBundle()
  .then((bundle) => {
    if (bundle.source !== "reference_fixtures") return;
    const banner = document.querySelector<HTMLElement>(".fixture-banner")!;
    banner.innerHTML =
      "<strong>Development data.</strong> These flights come from a reference autopilot that steers by the true gate positions. They test the replay, they are not results.";
    banner.hidden = false;
  })
  .catch(() => {
    // Each view reports its own loading errors.
  });
