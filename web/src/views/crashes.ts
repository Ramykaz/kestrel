import type { MountedView } from "../router";

export function mount(root: HTMLElement): MountedView {
  root.innerHTML = `<div class="page"><header class="page-header"><h1>Crash review</h1>
    <p>Every crash on the held-out courses gets a 2 by 2 strip of its last 2 seconds of onboard frames. A fine-tuned vision-language model diagnoses each one, and its answer is checked against the cause computed from the flight log.</p></header>
    <div class="state"><h2>No crashes have been reviewed yet</h2>
    <p>The crash analyst is fine-tuned once early rounds have produced enough crashes to learn from.</p></div></div>`;
  return { unmount: () => undefined };
}
