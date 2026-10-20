import type { MountedView } from "../router";

export function mount(root: HTMLElement): MountedView {
  root.innerHTML = `<div class="page"><header class="page-header"><h1>Detector lab</h1>
    <p>This view will run the trained gate detector in your browser on held-out frames, in fp32 and int8, so you can see what quantization costs in accuracy and speed.</p></header>
    <div class="state"><h2>The detector has not been trained yet</h2>
    <p>Training starts with the first 8,000 synthetic frames in phase 2. The model downloads only when this view opens, so it never slows down the rest of the site.</p></div></div>`;
  return { unmount: () => undefined };
}
