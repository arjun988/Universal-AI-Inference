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
      timeout: 10000,
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
    const forceNative = process.env.UAII_USE_WSL === "0";
    const native = resolveNative(configuredBin);
    const distro = process.env.UAII_WSL_DISTRO || "Ubuntu";

    // On Windows, try native first unless UAII_USE_WSL=1 forces WSL.
    // Fall back to WSL only when the native binary is missing or blocked by App Control.
    if (!forceWsl && (forceNative || nativeRunnable(native))) {
      cached = {
        mode: "native",
        cmd: native,
        prefix: [],
        display: native,
        toModelPath: (p) => p,
      };
      return cached;
    }

    // Try WSL when forced or native is unavailable/blocked.
    if (process.platform === "win32" || forceWsl) {
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

    // Last resort: use whatever native resolved to (may not be runnable).
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

  /**
   * For WSL launches, copy large GGUFs onto the Linux filesystem once.
   * Reading weights via /mnt/c is often too slow for chat ready timeouts.
   */
  function wslHome(distro) {
    const r = spawnSync("wsl", ["-d", distro, "--", "printenv", "HOME"], {
      windowsHide: true,
      encoding: "utf8",
      timeout: 15000,
    });
    const h = (r.stdout || "").trim();
    return r.status === 0 && h ? h : "/home/arjun";
  }

  function stageModelPath(hostPath) {
    const L = resolve();
    if (L.mode !== "wsl" || !hostPath) return L.toModelPath(hostPath);
    const src = L.toModelPath(hostPath);
    if (!src.startsWith("/mnt/")) return src;
    const base = path.posix.basename(src.replace(/\\/g, "/"));
    const distro = process.env.UAII_WSL_DISTRO || "Ubuntu";
    const cacheDir = (process.env.UAII_WSL_MODEL_DIR || `${wslHome(distro)}/uaii-models`).replace(
      /\/$/,
      "",
    );
    const dst = `${cacheDir}/${base}`;
    // Avoid $VAR in the bash string — Windows/WSL arg passing can strip them.
    const bash = [
      `mkdir -p '${cacheDir}'`,
      `if [ ! -f '${dst}' ] || [ "$(stat -c%s '${dst}' 2>/dev/null || echo 0)" != "$(stat -c%s '${src}' 2>/dev/null || echo 1)" ]; then cp -f '${src}' '${dst}'; fi`,
      `test -f '${dst}' && printf '%s' '${dst}'`,
    ].join(" && ");
    const ready = spawnSync("wsl", ["-d", distro, "--", "bash", "-lc", bash], {
      windowsHide: true,
      encoding: "utf8",
      timeout: 600000,
    });
    const staged = (ready.stdout || "").trim();
    if (ready.status === 0 && staged) return staged;
    console.warn("[uaii-launch] WSL model stage failed; using /mnt path:", ready.stderr || ready.stdout);
    return src;
  }

  return { resolve, spawnArgs, winToWsl, stageModelPath };
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
