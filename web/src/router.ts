export type ViewName = "flight" | "rounds" | "lab" | "crashes" | "notes";

export interface MountedView {
  unmount(): void;
}

export type ViewLoader = () => Promise<{
  mount(root: HTMLElement, params: URLSearchParams): MountedView | Promise<MountedView>;
}>;

const VIEW_TITLES: Record<ViewName, string> = {
  flight: "Flight",
  rounds: "Rounds",
  lab: "Detector lab",
  crashes: "Crash review",
  notes: "Notes",
};

export function viewFromPath(pathname: string): ViewName | null {
  const segment = pathname.replace(/^\/+|\/+$/g, "");
  if (segment === "" || segment === "flight") return "flight";
  if (segment in VIEW_TITLES) return segment as ViewName;
  return null;
}

/** Replace the query string without adding a history entry. */
export function updateQuery(values: Record<string, string | number | null>): void {
  const url = new URL(window.location.href);
  for (const [key, value] of Object.entries(values)) {
    if (value === null) url.searchParams.delete(key);
    else url.searchParams.set(key, String(value));
  }
  window.history.replaceState(window.history.state, "", url);
}

export class Router {
  private current: MountedView | null = null;
  private generation = 0;

  constructor(
    private readonly root: HTMLElement,
    private readonly loaders: Record<ViewName, ViewLoader>,
  ) {
    document.addEventListener("click", (event) => this.onClick(event));
    window.addEventListener("popstate", () => void this.render());
  }

  navigate(href: string): void {
    const url = new URL(href, window.location.href);
    if (url.pathname === window.location.pathname && url.search === window.location.search) return;
    window.history.pushState(null, "", url);
    void this.render();
    this.root.focus({ preventScroll: true });
    window.scrollTo(0, 0);
  }

  async render(): Promise<void> {
    const generation = ++this.generation;
    const view = viewFromPath(window.location.pathname);
    this.current?.unmount();
    this.current = null;
    this.root.replaceChildren();
    document.querySelectorAll<HTMLAnchorElement>("[data-view]").forEach((link) => {
      if (link.dataset.view === view) link.setAttribute("aria-current", "page");
      else link.removeAttribute("aria-current");
    });
    if (view === null) {
      document.title = "Page not found | Kestrel";
      this.root.innerHTML = `<div class="page"><div class="state"><h2>There is no page at this address</h2>
        <p>Use the views in the top bar, or <a href="/flight" data-link>open the flight replay</a>.</p></div></div>`;
      return;
    }
    document.title = `${VIEW_TITLES[view]} | Kestrel`;
    try {
      const module = await this.loaders[view]();
      if (generation !== this.generation) return;
      const mounted = await module.mount(this.root, new URLSearchParams(window.location.search));
      if (generation !== this.generation) {
        mounted.unmount();
        return;
      }
      this.current = mounted;
    } catch (error) {
      if (generation !== this.generation) return;
      console.error(error);
      this.root.innerHTML = `<div class="page"><div class="state state-error"><h2>This view did not load</h2>
        <p>Check your connection and reload the page.</p></div></div>`;
    }
  }

  private onClick(event: MouseEvent): void {
    if (event.defaultPrevented || event.button !== 0 || event.metaKey || event.ctrlKey || event.shiftKey || event.altKey) return;
    const link = (event.target as Element).closest<HTMLAnchorElement>("a[data-link]");
    if (!link || link.origin !== window.location.origin) return;
    event.preventDefault();
    this.navigate(link.href);
  }
}
