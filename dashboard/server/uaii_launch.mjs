import { spawnSync } from "child_process";
import fs from "fs";
import os from "os";
import path from "path";

/**
 * Resolve how to invoke `uaii` — native binary, or WSL when Windows App Control blocks .exe.
 */
export function createUaiiLauncher({ repoRoot, configuredBin = "" }) {
  let cached = null;

  function winToWsl(p) {
    if (!p) return p;
    const norm = path.resolve(p);
    const m = /^([A-Za-z]):[\\/](.*)$/.exec(norm);
    if (!m) return norm.replace(/\\/g, "/");
    return `/mnt/${m[1].toLowerCase()}/${m[2].replace(/\\/g, "/")}`;
  }

  function wslExists(linuxPath) {
    const distro = process.env.UAII_WSL_DISTRO || "Ubuntu";
    const r = spawnSync("wsl", ["-d", distro, "--", "test", "-x", linuxPath], {
      windowsHide: true,
      encoding: "utf8",
    });
    return r.status === 0;
  }

  function findWslUaii() {
    const candidates = [
      process.env.UAII_WSL_BIN,
      `${process.env.UAII_WSL_HOME || ""}/uaii-wsl-build-dash/libs/uaii-cli/uaii`.replace(/^\/+/, "/"),
      "/home/arjun/uaii-wsl-build-dash/libs/uaii-cli/uaii",
      "/home/arjun/uaii-wsl-build-vendor/libs/uaii-cli/uaii",
      "/home/arjun/uaii-wsl-build/libs/uaii-cli/uaii",
    ].filter((p) => p && !p.includes("undefined"));
    for (const p of candidates) {
      if (wslExists(p)) return p;
    }
    return "";
  }

  function resolveNative(configBin) {
    if (configBin && fs.existsSync(configBin)) return path.resolve(configBin);
    const exe = process.platform === "win32" ? "uaii.exe" : "uaii";
    const list = [
      path.join(repoRoot, "build", "libs", "uaii-cli", exe),
      path.join(repoRoot, "build", "libs", "uaii-cli", "Release", exe),
      path.join(repoRoot, "build", "Release", exe),
      path.join(repoRoot, "build", exe),
    ];
    for (const p of list) if (fs.existsSync(p)) return path.resolve(p);
    return configBin || "uaii";
  }

  function nativeRunnable(bin) {
    if (process.platform !== "win32") return fs.existsSync(bin);
    if (!fs.existsSync(bin)) return false;
    // App Control often blocks newly linked unsigned EXEs — probe with --help quickly.
    const r = spawnSync(bin, ["version"], { windowsHide: true, encoding: "utf8", timeout: 5000 });
    if (r.error) return false;
    // exit 0 or printable version → ok; blocked often yields status null + error
    if (r.status === 0) return true;
    const err = `${r.stderr || ""}${r.stdout || ""}${r.error || ""}`;
    if (/Application Control|blocked this file/i.test(err)) return false;
    // Some hosts return non-zero oddly; if stdout has a version-like string accept it
    if ((r.stdout || "").trim().length > 0) return true;
    return false;
  }

  function resolve() {
    if (cached) return cached;
    const forceWsl = process.env.UAII_USE_WSL === "1";
    const native = resolveNative(configuredBin);
    const distro = process.env.UAII_WSL_DISTRO || "Ubuntu";

    if (forceWsl || (process.platform === "win32" && !nativeRunnable(native))) {
      const linux = findWslUaii();
      if (linux) {
        cached = {
          mode: "wsl",
          cmd: "wsl",
          prefix: ["-d", distro, "--", linux],
          display: `wsl:${linux}`,
          toModelPath: winToWsl,
        };
        return cached;
      }
    }

    cached = {
      mode: "native",
      cmd: native,
      prefix: [],
      display: native,
      toModelPath: (p) => p,
    };
    return cached;
  }

  function spawnArgs(uaiiArgs) {
    const L = resolve();
    return { cmd: L.cmd, args: [...L.prefix, ...uaiiArgs], display: L.display, toModelPath: L.toModelPath };
  }

  return { resolve, spawnArgs, winToWsl };
}

export function createBenchLauncher({ repoRoot, configuredBin = "" }) {
  if (configuredBin && fs.existsSync(configuredBin)) {
    return { cmd: path.resolve(configuredBin), prefix: [], display: configuredBin };
  }
  const names = [
    path.join(repoRoot, "build", "benchmarks", process.platform === "win32" ? "uaii_bench.exe" : "uaii_bench"),
    path.join(repoRoot, "build", "benchmarks", "Release", "uaii_bench.exe"),
    path.join(os.homedir(), "uaii-wsl-build-vendor", "benchmarks", "uaii_bench"),
  ];
  for (const p of names) {
    if (fs.existsSync(p)) return { cmd: path.resolve(p), prefix: [], display: p };
  }
  // WSL bench fallback
  if (process.platform === "win32") {
    const linux = "/home/arjun/uaii-wsl-build-vendor/benchmarks/uaii_bench";
    const distro = process.env.UAII_WSL_DISTRO || "Ubuntu";
    const r = spawnSync("wsl", ["-d", distro, "--", "test", "-x", linux], { windowsHide: true });
    if (r.status === 0) {
      return { cmd: "wsl", prefix: ["-d", distro, "--", linux], display: `wsl:${linux}` };
    }
  }
  return { cmd: "", prefix: [], display: "" };
}
