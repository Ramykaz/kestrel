import type { MountedView } from "../router";

export function mount(root: HTMLElement): MountedView {
  root.innerHTML = `<div class="page"><header class="page-header"><h1>Notes</h1>
    <p>The engineering write-up: the problem, the rules of the world, how the agent harness works, what broke and what fixed it.</p></header>
    <div class="state"><h2>The write-up comes after the first rounds</h2>
    <p>It will link straight into specific moments of the flight replay and the round history.</p></div></div>`;
  return { unmount: () => undefined };
}
