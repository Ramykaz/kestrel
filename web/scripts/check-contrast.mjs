// WCAG AA contrast check for every text color the site puts on a background, in both themes.
// Colors are read from src/styles/tokens.css so the check cannot drift from the real tokens.
import { readFileSync } from "node:fs";

const css = readFileSync(new URL("../src/styles/tokens.css", import.meta.url), "utf8");

function tokensFrom(block) {
  const tokens = {};
  for (const [, name, value] of block.matchAll(/--([a-z-]+):\s*(#[0-9a-f]{6})/gi)) tokens[name] = value;
  return tokens;
}

const day = tokensFrom(css.slice(css.indexOf(":root {"), css.indexOf("@media")));
const night = tokensFrom(css.slice(css.indexOf(':root[data-theme="night"]')));

function luminance(hex) {
  const channels = [1, 3, 5].map((i) => parseInt(hex.slice(i, i + 2), 16) / 255);
  const [r, g, b] = channels.map((c) => (c <= 0.03928 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4));
  return 0.2126 * r + 0.7152 * g + 0.0722 * b;
}

function ratio(a, b) {
  const [hi, lo] = [luminance(a), luminance(b)].sort((x, y) => y - x);
  return (hi + 0.05) / (lo + 0.05);
}

// [foreground, background, minimum ratio, where it is used]
const pairs = [
  ["ink", "paper", 4.5, "body text"],
  ["ink", "paper-raised", 4.5, "panel text"],
  ["ink-soft", "paper", 4.5, "secondary text"],
  ["ink-soft", "paper-raised", 4.5, "secondary text on panels"],
  ["airway", "paper", 4.5, "links, selected round number"],
  ["airway", "paper-raised", 4.5, "selected round number on the scrubber"],
  ["paper", "route", 4.5, "primary button label"],
  ["paper", "ink", 4.5, "inset caption, active theme label"],
  ["paper", "airway", 4.5, "detection confidence labels"],
  ["alert", "paper", 4.5, "crash labels"],
  ["alert", "paper-raised", 4.5, "failed claim values"],
  ["route", "paper", 3, "route time labels (large display text)"],
];

let failures = 0;
for (const [theme, tokens] of [["day", day], ["night", night]]) {
  for (const [fg, bg, minimum, use] of pairs) {
    if (!tokens[fg] || !tokens[bg]) throw new Error(`token --${fg} or --${bg} missing from the ${theme} theme`);
    const value = ratio(tokens[fg], tokens[bg]);
    const ok = value >= minimum;
    if (!ok) failures++;
    console.log(`${ok ? "pass" : "FAIL"}  ${theme.padEnd(5)} ${`--${fg} on --${bg}`.padEnd(30)} ${value.toFixed(2)} (needs ${minimum})  ${use}`);
  }
}
if (failures) {
  console.error(`${failures} pairs below WCAG AA`);
  process.exit(1);
}
