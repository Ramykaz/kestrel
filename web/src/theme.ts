export type Theme = "day" | "night";

const STORAGE_KEY = "kestrel-theme";
const listeners = new Set<(theme: Theme) => void>();
const darkQuery = window.matchMedia("(prefers-color-scheme: dark)");

export function currentTheme(): Theme {
  const explicit = document.documentElement.dataset.theme;
  if (explicit === "day" || explicit === "night") return explicit;
  return darkQuery.matches ? "night" : "day";
}

function notify(): void {
  const theme = currentTheme();
  for (const listener of listeners) listener(theme);
}

export function setTheme(theme: Theme): void {
  document.documentElement.dataset.theme = theme;
  try {
    localStorage.setItem(STORAGE_KEY, theme);
  } catch {
    // Storage can be unavailable in private windows; the choice then lasts for this page only.
  }
  notify();
}

export function onThemeChange(listener: (theme: Theme) => void): () => void {
  listeners.add(listener);
  return () => listeners.delete(listener);
}

/** Resolved value of a CSS custom property such as --route. */
export function token(name: string): string {
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
}

export function initThemeToggle(group: HTMLElement): void {
  const buttons = [...group.querySelectorAll<HTMLButtonElement>("[data-theme-choice]")];
  const sync = (theme: Theme) => {
    for (const button of buttons) button.setAttribute("aria-pressed", String(button.dataset.themeChoice === theme));
  };
  for (const button of buttons) button.addEventListener("click", () => setTheme(button.dataset.themeChoice as Theme));
  darkQuery.addEventListener("change", notify);
  onThemeChange(sync);
  sync(currentTheme());
}
