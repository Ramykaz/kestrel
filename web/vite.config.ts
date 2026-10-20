import { createReadStream, existsSync, statSync } from "node:fs";
import { resolve, sep } from "node:path";
import { fileURLToPath } from "node:url";
import { defineConfig, type Plugin } from "vite";

// Serves web/fixtures at /fixtures during development only. Fixtures are reference autopilot
// flights written by `kestrel export-web --reference-fixtures --out web/fixtures`; they are
// never copied into the build.
function serveFixtures(): Plugin {
  const root = fileURLToPath(new URL("./fixtures", import.meta.url));
  return {
    name: "kestrel-fixtures",
    apply: "serve",
    configureServer(server) {
      server.middlewares.use("/fixtures", (req, res, next) => {
        const path = resolve(root, `.${decodeURIComponent((req.url ?? "/").split("?")[0])}`);
        if (!path.startsWith(root + sep) || !existsSync(path) || !statSync(path).isFile()) return next();
        res.setHeader("Content-Type", path.endsWith(".json") ? "application/json" : "application/octet-stream");
        createReadStream(path).pipe(res);
      });
    },
  };
}

export default defineConfig({
  plugins: [serveFixtures()],
  build: {
    outDir: "dist",
    target: "es2022",
  },
});
