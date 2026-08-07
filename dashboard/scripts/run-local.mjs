import { spawn } from "child_process";
import path from "path";
import { fileURLToPath } from "url";

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
process.chdir(root);
process.env.NODE_ENV = "production";
process.env.UAII_DASH_BIND = process.env.UAII_DASH_BIND || "127.0.0.1";
process.env.UAII_DASH_PORT = process.env.UAII_DASH_PORT || "8787";

const build = spawn("npm", ["run", "build"], { stdio: "inherit", shell: true });
build.on("close", (code) => {
  if (code !== 0) process.exit(code || 1);
  const start = spawn("npm", ["start"], { stdio: "inherit", shell: true });
  start.on("close", (c) => process.exit(c || 0));
});
