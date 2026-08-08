import cors from "cors";
import express from "express";
import fs from "fs";
import multer from "multer";
import os from "os";
import path from "path";
import { spawn, spawnSync } from "child_process";
import { fileURLToPath } from "url";
import { randomUUID } from "crypto";
import { ChatWorker, getChatWorker, stopAllChatWorkers } from "./chat_worker.mjs";
import { createBenchLauncher, createUaiiLauncher } from "./uaii_launch.mjs";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const DASH_ROOT = path.resolve(__dirname, "..");
const REPO_ROOT = path.resolve(DASH_ROOT, "..");
const VERSION = "0.2.0";

const LOG_LIMIT = 300;
const logs = [];
const jobs = new Map(); // id -> { child, status, ... }

const DEMOS = [
  "toy_mlp",
  "tiny_block",
  "gguf",
  "safetensors",
  "moe",
  "parity",
  "optimize",
  "streaming",
  "profile",
  "quant",
];

// ---------------------------------------------------------------------------
// GPU auto-detection
// ---------------------------------------------------------------------------

/**
 * Probe which GPU backends are natively available by running `uaii doctor`
 * and parsing its output. Returns the best backend name to use.
 * Priority: cuda > rocm > metal > vulkan > cpu
 *
 * This runs once at server startup (synchronously, with a short timeout).
 * Result is cached in `detectedBackend`.
 */
let detectedBackend = null;        // null = not yet probed
let detectedBackendDetails = "";   // human-readable details for /api/health

function probeGpuBackend(launcherDisplay) {
  if (detectedBackend !== null) return detectedBackend;

  const L = uaiiLauncher.resolve();
  const cmd = L.cmd;
  const args = [...(L.prefix || []), "doctor"];

  try {
    const r = spawnSync(cmd, args, {
      encoding: "utf8",
      timeout: 15000,
      windowsHide: true,
      env: { ...process.env },
    });
    const out = (r.stdout || "") + (r.stderr || "");

    // Parse `uaii doctor` backend lines:
    //   cuda: Cuda create=ok native_compiled=yes — ... host_fallback=no ...
    // The backends in priority order we want to try:
    const priority = ["cuda", "rocm", "metal", "vulkan"];
    for (const name of priority) {
      // Match lines like:  cuda: ... create=ok ... host_fallback=no
      const re = new RegExp(`${name}:.*create=ok.*host_fallback=no`, "i");
      if (re.test(out)) {
        // Extract details snippet
        const lineMatch = out.match(new RegExp(`${name}:.*`, "i"));
        detectedBackendDetails = lineMatch ? lineMatch[0].trim().slice(0, 120) : name;
        detectedBackend = name;
        console.log(`[uaii] GPU detected: ${name} — ${detectedBackendDetails}`);
        return detectedBackend;
      }
    }
    // All gpu backends have host_fallback=yes or failed — use cpu
    detectedBackendDetails = "no GPU backend with native device found; using CPU";
    detectedBackend = "cpu";
  } catch (e) {
    detectedBackendDetails = `probe failed: ${e.message}`;
    detectedBackend = "cpu";
  }
  console.log(`[uaii] Backend: ${detectedBackend} (${detectedBackendDetails})`);
  return detectedBackend;
}

/**
 * Return the effective backend to use for inference.
 * If user explicitly set a backend in config/env (not "auto"), honour it.
 * Otherwise fall back to auto-detected GPU or CPU.
 */
function effectiveBackend() {
  if (config.backend && config.backend !== "auto") return config.backend;
  return probeGpuBackend();
}


function pushLog(entry) {
  logs.unshift({ ts: new Date().toISOString(), ...entry });
  if (logs.length > LOG_LIMIT) logs.length = LOG_LIMIT;
}

function loadConfig() {
  const cfgPath = process.env.UAII_DASH_CONFIG
    ? path.resolve(process.env.UAII_DASH_CONFIG)
    : path.join(DASH_ROOT, "uaii-dash.json");
  let file = {};
  if (fs.existsSync(cfgPath)) {
    try {
      file = JSON.parse(fs.readFileSync(cfgPath, "utf8"));
    } catch (e) {
      console.warn("Failed to parse config:", e.message);
    }
  }
  const modelDir = process.env.UAII_MODEL_DIR || file.modelDir || path.join(DASH_ROOT, "models");
  const bind = process.env.UAII_DASH_BIND || file.bind || "127.0.0.1";
  const token = process.env.UAII_DASH_TOKEN || file.token || "";
  const loopback = bind === "127.0.0.1" || bind === "localhost" || bind === "::1";
  return {
    bind,
    port: Number(process.env.UAII_DASH_PORT || file.port || 8787),
    uaiiBin: process.env.UAII_BIN || file.uaiiBin || "",
    benchBin: process.env.UAII_BENCH_BIN || file.benchBin || "",
    modelDir: path.resolve(modelDir),
    token,
    threads: Number(process.env.UAII_NUM_THREADS || file.threads || 0),
    gemm: process.env.UAII_GEMM || file.gemm || "",
    backend: process.env.UAII_BACKEND || file.backend || "auto",
    maxContext: Number(process.env.UAII_MAX_CONTEXT || file.maxContext || 0),
    configPath: cfgPath,
    loopback,
    // Auth required on non-loopback, or whenever token is set
    authRequired: Boolean(token) && (!loopback || process.env.UAII_DASH_FORCE_AUTH === "1"),
  };
}

let config = loadConfig();
let uaiiLauncher = createUaiiLauncher({ repoRoot: REPO_ROOT, configuredBin: config.uaiiBin });

function resolveUaiiBin() {
  return uaiiLauncher.resolve().display;
}

function resolveBenchBin() {
  const b = createBenchLauncher({ repoRoot: REPO_ROOT, configuredBin: config.benchBin });
  return b.display || "";
}

function reloadLauncher() {
  uaiiLauncher = createUaiiLauncher({ repoRoot: REPO_ROOT, configuredBin: config.uaiiBin });
}

function childEnv() {
  const env = { ...process.env };
  if (config.threads > 0) env.UAII_NUM_THREADS = String(config.threads);
  if (config.gemm) env.UAII_GEMM = config.gemm;
  return env;
}

function runProcess(bin, args, opts = {}) {
  return new Promise((resolve) => {
    const jobId = opts.jobId || randomUUID();
    const spawnOpts = {
      cwd: opts.cwd || REPO_ROOT,
      env: childEnv(),
      windowsHide: true,
    };
    // Allow {cmd, args} launch objects (e.g. wsl + linux binary)
    let cmd = bin;
    let argv = args;
    if (bin && typeof bin === "object" && bin.cmd) {
      cmd = bin.cmd;
      argv = [...(bin.prefix || []), ...args];
    }
    let child;
    try {
      child = spawn(cmd, argv, spawnOpts);
    } catch (err) {
      const result = {
        ok: false,
        code: -1,
        jobId,
        bin: cmd,
        args: argv,
        stdout: "",
        stderr: err.message,
        ms: 0,
      };
      pushLog({ kind: "error", cmd: [cmd, ...argv].join(" "), ...result });
      return resolve(result);
    }
    jobs.set(jobId, { child, status: "running", bin: cmd, args: argv, started: Date.now() });
    let stdout = "";
    let stderr = "";
    const t0 = Date.now();
    const onChunk = opts.onChunk;
    child.stdout.on("data", (d) => {
      const s = d.toString();
      stdout += s;
      if (onChunk) onChunk("stdout", s);
    });
    child.stderr.on("data", (d) => {
      const s = d.toString();
      stderr += s;
      if (onChunk) onChunk("stderr", s);
    });
    child.on("error", (err) => {
      jobs.set(jobId, { status: "error", error: err.message });
      const result = {
        ok: false,
        code: -1,
        jobId,
        bin,
        args,
        stdout,
        stderr: stderr + "\n" + err.message,
        ms: Date.now() - t0,
      };
      pushLog({ kind: "error", cmd: [bin, ...args].join(" "), ...result });
      resolve(result);
    });
    child.on("close", (code) => {
      jobs.set(jobId, { status: code === 0 ? "ok" : "fail", code, finished: Date.now() });
      const result = {
        ok: code === 0,
        code,
        jobId,
        bin,
        args,
        stdout,
        stderr,
        ms: Date.now() - t0,
      };
      pushLog({
        kind: code === 0 ? "ok" : "fail",
        cmd: [bin, ...args].join(" "),
        code,
        ms: result.ms,
        preview: (stdout || stderr).slice(0, 500),
      });
      resolve(result);
    });
  });
}

function runUaii(args, opts = {}) {
  const launch = uaiiLauncher.spawnArgs(args);
  return runProcess(launch.cmd, launch.args, opts);
}

function ensureModelDir() {
  fs.mkdirSync(config.modelDir, { recursive: true });
}

function listModels() {
  ensureModelDir();
  const items = [];
  for (const name of fs.readdirSync(config.modelDir)) {
    const full = path.join(config.modelDir, name);
    let st;
    try {
      st = fs.statSync(full);
    } catch {
      continue;
    }
    if (!st.isFile()) continue;
    const lower = name.toLowerCase();
    let kind = null;
    if (lower.endsWith(".gguf")) kind = "gguf";
    else if (lower.endsWith(".uaii.json")) kind = "uaii-ir";
    else if (lower.endsWith(".onnx")) kind = "onnx";
    else if (lower.endsWith(".safetensors")) kind = "safetensors";
    if (!kind) continue;
    items.push({
      name,
      path: full,
      bytes: st.size,
      mib: +(st.size / (1024 * 1024)).toFixed(2),
      mtime: st.mtime.toISOString(),
      kind,
    });
  }
  items.sort((a, b) => (a.mtime < b.mtime ? 1 : -1));
  return items;
}

function extractBearer(req) {
  const hdr = req.headers.authorization || "";
  if (hdr.startsWith("Bearer ")) return hdr.slice(7);
  if (req.query.token) return String(req.query.token);
  if (req.headers["x-uaii-token"]) return String(req.headers["x-uaii-token"]);
  return "";
}

function authMiddleware(req, res, next) {
  // Bootstrap endpoints (login must work before the client has a Bearer token)
  if (
    req.path === "/api/health" ||
    req.path === "/api/auth/status" ||
    req.path === "/api/auth/verify"
  ) {
    return next();
  }
  if (!config.authRequired) return next();
  const token = extractBearer(req);
  if (!config.token || token !== config.token) {
    return res.status(401).json({ error: "unauthorized", hint: "Send Authorization: Bearer <token>" });
  }
  return next();
}

const app = express();
app.use(cors({ origin: true, credentials: true }));
app.use(express.json({ limit: "4mb" }));
app.use(authMiddleware);

const upload = multer({
  storage: multer.diskStorage({
    destination: (_req, _file, cb) => {
      ensureModelDir();
      cb(null, config.modelDir);
    },
    filename: (_req, file, cb) => cb(null, path.basename(file.originalname)),
  }),
  limits: { fileSize: 64 * 1024 * 1024 * 1024 },
});

app.get("/api/health", (_req, res) => {
  const L = uaiiLauncher.resolve();
  const bench = createBenchLauncher({ repoRoot: REPO_ROOT, configuredBin: config.benchBin });
  const nativePath = path.join(
    REPO_ROOT,
    "build",
    "libs",
    "uaii-cli",
    process.platform === "win32" ? "uaii.exe" : "uaii",
  );

  // For WSL mode, cmd is "wsl" which always exists — check the actual linux binary instead.
  let uaiiBinExists = false;
  if (L.mode === "wsl") {
    // prefix is ["-d", distro, "--", linuxPath]; last element is the binary
    const linuxBin = L.prefix && L.prefix[L.prefix.length - 1];
    uaiiBinExists = Boolean(linuxBin && linuxBin.startsWith("/"));
  } else {
    uaiiBinExists = Boolean(L.cmd && fs.existsSync(L.cmd));
  }

  let hint = null;
  if (L.mode === "wsl" && !uaiiBinExists) {
    hint = "WSL binary not found. Build uaii inside WSL or set UAII_WSL_BIN.";
  } else if (L.mode === "native" && !uaiiBinExists) {
    hint = "Set UAII_BIN or build uaii (cmake --build build --target uaii).";
  } else if (L.mode === "native" && fs.existsSync(nativePath) && !uaiiBinExists) {
    hint = "Set UAII_BIN / UAII_USE_WSL=1, or build uaii (see scripts/test_generate_wsl.sh)";
  }

  res.json({
    ok: true,
    service: "uaii-dashboard",
    version: VERSION,
    bind: config.bind,
    port: config.port,
    uaiiBin: L.display,
    uaiiLaunchMode: L.mode,
    uaiiBinExists,
    benchBin: bench.display || null,
    benchBinExists: Boolean(bench.display),
    modelDir: config.modelDir,
    hostname: os.hostname(),
    platform: process.platform,
    loopbackOnly: config.loopback,
    authRequired: config.authRequired,
    demos: DEMOS,
    mode: config.loopback ? "local" : "self-host",
    backend: config.backend,
    effectiveBackend: effectiveBackend(),
    backendDetails: detectedBackendDetails || null,
    hint,
  });
});

app.get("/api/auth/status", (req, res) => {
  const provided = extractBearer(req);
  res.json({
    authRequired: config.authRequired,
    tokenConfigured: Boolean(config.token),
    authorized: !config.authRequired || (config.token && provided === config.token),
  });
});

app.post("/api/auth/verify", (req, res) => {
  const token = (req.body && req.body.token) || "";
  if (!config.token) {
    return res.json({ ok: true, authRequired: false });
  }
  if (token === config.token) return res.json({ ok: true, authRequired: config.authRequired });
  return res.status(401).json({ ok: false, error: "invalid token" });
});

app.get("/api/settings", (_req, res) => {
  res.json({
    bind: config.bind,
    port: config.port,
    uaiiBin: resolveUaiiBin(),
    benchBin: resolveBenchBin() || "",
    modelDir: config.modelDir,
    threads: config.threads,
    gemm: config.gemm,
    backend: config.backend,  // stored value ("auto" or explicit)
    effectiveBackend: effectiveBackend(),
    tokenSet: Boolean(config.token),
    authRequired: config.authRequired,
    configPath: config.configPath,
    mode: config.loopback ? "local" : "self-host",
  });
});

app.put("/api/settings", (req, res) => {
  const body = req.body || {};
  if (typeof body.uaiiBin === "string") config.uaiiBin = body.uaiiBin;
  if (typeof body.benchBin === "string") config.benchBin = body.benchBin;
  if (typeof body.modelDir === "string" && body.modelDir) {
    config.modelDir = path.resolve(body.modelDir);
  }
  if (body.threads !== undefined) config.threads = Number(body.threads) || 0;
  if (typeof body.gemm === "string") config.gemm = body.gemm;
  if (typeof body.backend === "string") {
    config.backend = body.backend;
    // Reset detection cache so next effectiveBackend() call re-probes.
    detectedBackend = null;
    detectedBackendDetails = "";
  }
  if (typeof body.token === "string" && body.token.length > 0) config.token = body.token;
  if (typeof body.bind === "string" && body.bind) config.bind = body.bind;
  if (body.port !== undefined) config.port = Number(body.port) || config.port;
  config.loopback =
    config.bind === "127.0.0.1" || config.bind === "localhost" || config.bind === "::1";
  config.authRequired =
    Boolean(config.token) && (!config.loopback || process.env.UAII_DASH_FORCE_AUTH === "1");

  const out = {
    bind: config.bind,
    port: config.port,
    uaiiBin: config.uaiiBin,
    benchBin: config.benchBin,
    modelDir: config.modelDir,
    token: config.token,
    threads: config.threads,
    gemm: config.gemm,
    backend: config.backend,
    effectiveBackend: effectiveBackend(),
  };
  try {
    fs.writeFileSync(config.configPath, JSON.stringify(out, null, 2));
  } catch (e) {
    return res.status(500).json({ error: e.message });
  }
  reloadLauncher();
  stopAllChatWorkers();
  pushLog({ kind: "ok", cmd: "settings.save", preview: config.configPath });
  res.json({
    ok: true,
    settings: { ...out, token: undefined, tokenSet: Boolean(config.token) },
    restartRequired: true,
    note: "bind/port changes apply on next process start",
  });
});

app.get("/api/doctor", async (_req, res) => {
  const result = await runUaii(["doctor", "--load-plugins"]);
  res.status(result.ok ? 200 : 502).json(result);
});

// Returns detected GPU info + what backend is currently active.
// The frontend uses this to show the backend pill and Settings hint.
app.get("/api/backends", (_req, res) => {
  const active = effectiveBackend();
  res.json({
    configured: config.backend,        // "auto" or explicit name
    effective: active,                  // what will actually be used
    details: detectedBackendDetails,
    available: [
      { name: "auto",   label: "Auto-detect GPU" },
      { name: "cpu",    label: "CPU" },
      { name: "cuda",   label: "CUDA (NVIDIA)" },
      { name: "rocm",   label: "ROCm (AMD)" },
      { name: "metal",  label: "Metal (Apple)" },
      { name: "vulkan", label: "Vulkan (cross-platform)" },
    ],
  });
});

app.get("/api/models", (_req, res) => {
  try {
    res.json({ modelDir: config.modelDir, models: listModels() });
  } catch (e) {
    res.status(500).json({ error: e.message });
  }
});

app.post("/api/models/upload", upload.single("file"), (req, res) => {
  if (!req.file) return res.status(400).json({ error: "file required" });
  pushLog({ kind: "ok", cmd: "models.upload", preview: req.file.filename });
  res.json({ ok: true, name: req.file.filename, path: req.file.path, bytes: req.file.size });
});

app.delete("/api/models/:name", (req, res) => {
  const full = path.join(config.modelDir, path.basename(req.params.name));
  if (!fs.existsSync(full)) return res.status(404).json({ error: "not found" });
  fs.unlinkSync(full);
  pushLog({ kind: "ok", cmd: "models.delete", preview: req.params.name });
  res.json({ ok: true });
});

app.post("/api/run/demo", async (req, res) => {
  const demo = (req.body && req.body.demo) || "toy_mlp";
  if (!DEMOS.includes(demo)) {
    return res.status(400).json({ error: "unknown demo", allowed: DEMOS });
  }
  const result = await runUaii(["run", "--demo", demo, "--backend", effectiveBackend()]);
  res.status(result.ok ? 200 : 502).json(result);
});

app.post("/api/run/model", async (req, res) => {
  const name = req.body && req.body.name;
  if (!name) return res.status(400).json({ error: "name required" });
  const full = path.join(config.modelDir, path.basename(name));
  if (!fs.existsSync(full)) return res.status(404).json({ error: "model not found" });
  const lower = full.toLowerCase();

  if (lower.endsWith(".uaii.json")) {
    const inputs = (req.body && req.body.input) || "x=1,2,3,4";
    const args = [
      "run",
      full,
      "--backend",
      effectiveBackend(),
      "--weight-init",
      req.body?.weightInit || "ones",
    ];
    if (inputs) {
      args.push("--input", inputs);
    }
    const result = await runUaii(args);
    return res.status(result.ok ? 200 : 502).json(result);
  }
  if (lower.endsWith(".gguf")) {
    const outIr = full.replace(/\.gguf$/i, ".uaii.json");
    const conv = await runUaii(["convert", full, "-o", outIr]);
    if (!conv.ok) return res.status(502).json({ step: "convert", ...conv });
    return res.json({
      ok: true,
      step: "convert",
      message: "GGUF converted to UAII IR. Run the .uaii.json from Models, or use Chat → demo gguf for synthetic generate.",
      convert: conv,
      irPath: outIr,
      irName: path.basename(outIr),
    });
  }
  return res.status(400).json({ error: "unsupported type; use .uaii.json or .gguf" });
});

function formatChatPrompt(messages, fallbackPrompt, system) {
  const parts = [];
  if (system) parts.push(`System: ${system}`);
  const msgs = Array.isArray(messages) ? messages : [];
  if (msgs.length === 0 && fallbackPrompt) {
    parts.push(`User: ${fallbackPrompt}`);
    parts.push("Assistant:");
    return parts.join("\n\n");
  }
  for (const m of msgs) {
    const role = (m.role || "user").toLowerCase();
    const content = String(m.content || "");
    if (role === "system") parts.push(`System: ${content}`);
    else if (role === "assistant") parts.push(`Assistant: ${content}`);
    else parts.push(`User: ${content}`);
  }
  if (!parts.some((p) => p.startsWith("Assistant:") && p === parts[parts.length - 1])) {
    // Ensure the model continues as assistant
    if (parts.length && !parts[parts.length - 1].startsWith("Assistant:")) {
      parts.push("Assistant:");
    }
  }
  return parts.join("\n\n");
}

function resolveChatModel(body) {
  const mode = body.mode || "gguf";
  // Explicit demo only (API); Operator UI Chat no longer offers tiny demo.
  if (mode === "demo" || body.demoModel === true || body.model === "__demo__") {
    return { kind: "demo", path: "", label: "uaii-tiny-demo" };
  }
  const name = body.model || body.gguf || "";
  if (!name || name === "__demo__") {
    return {
      error: "No model selected. Upload a .gguf in Models, then choose it in Chat.",
    };
  }
  const full = path.isAbsolute(name) ? name : path.join(config.modelDir, path.basename(name));
  if (!fs.existsSync(full)) {
    return { error: "model not found — upload a .gguf in Models", path: full };
  }
  const lower = full.toLowerCase();
  if (lower.endsWith(".gguf")) {
    return { kind: "gguf", path: full, label: path.basename(full) };
  }
  if (lower.endsWith(".uaii.json")) {
    // Prefer sibling .gguf for tokenizer + prefer generate via convert path
    const sibling = full.replace(/\.uaii\.json$/i, ".gguf");
    if (fs.existsSync(sibling)) {
      return { kind: "gguf", path: sibling, label: path.basename(sibling), ir: full };
    }
    return { kind: "ir", path: full, label: path.basename(full) };
  }
  return { error: "unsupported model type (use .gguf)", path: full };
}

function workerForModel(resolved) {
  const key = resolved.kind === "demo" ? "__demo__" : resolved.path;
  const L = uaiiLauncher.resolve();
  let modelPath;
  if (resolved.kind === "demo") {
    modelPath = "";
  } else if (uaiiLauncher.stageModelPath) {
    // NOTE: stageModelPath may copy the GGUF into WSL (~/.uaii-models/) synchronously.
    // This blocks the Node event loop for large models; the chat request will appear
    // to hang until the copy finishes. This only runs once per model — subsequent
    // requests use the cached worker and skip staging.
    if (L.mode === "wsl") {
      console.log(`[uaii-launch] Staging model into WSL filesystem (one-time copy): ${resolved.path}`);
    }
    modelPath = uaiiLauncher.stageModelPath(resolved.path);
  } else if (L.toModelPath) {
    modelPath = L.toModelPath(resolved.path);
  } else {
    modelPath = resolved.path;
  }
  return getChatWorker(key, () =>
    new ChatWorker({
      launch: L,
      modelPath,
      demo: resolved.kind === "demo",
      backend: effectiveBackend(),
      maxContext: Number(config.maxContext || 0),
      env: childEnv(),
    }),
  );
}

async function runLlmChat(body, { onToken = null } = {}) {
  const mode = body.mode || "gguf";
  const prompt = String(body.prompt || "").trim();
  const system = String(body.system || "").trim();
  const maxNew = Math.max(1, Number(body.max_new_tokens || body.maxTokens || 64));
  const temperature =
    body.temperature != null && body.temperature !== "" ? Number(body.temperature) : undefined;
  const topP = body.top_p != null && body.top_p !== "" ? Number(body.top_p) : undefined;
  const topK = body.top_k != null && body.top_k !== "" ? Number(body.top_k) : undefined;
  const repetitionPenalty =
    body.repetition_penalty != null && body.repetition_penalty !== ""
      ? Number(body.repetition_penalty)
      : undefined;
  const seed = body.seed != null && body.seed !== "" ? Number(body.seed) : undefined;

  // Legacy utility modes
  if (mode === "tokenize") {
    if (!prompt) return { ok: false, status: 400, error: "prompt required" };
    const args = ["tokenize", "encode", prompt];
    if (body.gguf || body.model) {
      const ggufPath = path.join(config.modelDir, path.basename(body.gguf || body.model));
      if (fs.existsSync(ggufPath)) args.push("--gguf", ggufPath);
    }
    const result = await runUaii(args);
    return {
      ok: result.ok,
      status: result.ok ? 200 : 502,
      mode,
      role: "assistant",
      content: result.stdout || result.stderr,
      ...result,
    };
  }

  if (mode === "ir") {
    const name = body.model;
    if (!name) return { ok: false, status: 400, error: "model required for ir mode" };
    const full = path.join(config.modelDir, path.basename(name));
    if (!fs.existsSync(full)) return { ok: false, status: 404, error: "model not found" };
    const args = ["run", full, "--backend", effectiveBackend(), "--weight-init", "ones"];
    if (body.input) args.push("--input", body.input);
    else if (prompt) args.push("--input", `x=${prompt.split(/\s+/).slice(0, 8).join(",") || "1,2,3,4"}`);
    const result = await runUaii(args);
    return {
      ok: result.ok,
      status: result.ok ? 200 : 502,
      mode,
      role: "assistant",
      content: result.stdout || result.stderr,
      ...result,
    };
  }

  if (mode === "cli-demo") {
    const demo = body.demo || "gguf";
    if (!DEMOS.includes(demo)) return { ok: false, status: 400, error: "bad demo" };
    const result = await runUaii(["run", "--demo", demo, "--backend", effectiveBackend()]);
    return {
      ok: result.ok,
      status: result.ok ? 200 : 502,
      mode: "cli-demo",
      role: "assistant",
      content: result.stdout || result.stderr || "(no output)",
      demo,
      ...result,
    };
  }

  // Real LM path: gguf | demo (tiny) via uaii chat/generate
  const resolved = resolveChatModel(body);
  if (resolved.error) return { ok: false, status: 404, error: resolved.error, path: resolved.path };

  const fullPrompt = formatChatPrompt(body.messages, prompt, system);
  if (!fullPrompt.trim()) return { ok: false, status: 400, error: "prompt or messages required" };

  try {
    const worker = workerForModel(resolved);
    const ev = await worker.generate({
      prompt: fullPrompt,
      system: "", // already folded into fullPrompt
      maxNewTokens: maxNew,
      temperature,
      topP,
      topK,
      repetitionPenalty,
      seed,
      stream: Boolean(onToken || body.stream),
      onToken,
    });
    pushLog({
      kind: "ok",
      cmd: `chat.generate ${resolved.label}`,
      ms: ev.ms,
      preview: (ev.text || "").slice(0, 500),
    });
    return {
      ok: true,
      status: 200,
      mode: resolved.kind === "demo" ? "demo" : "gguf",
      role: "assistant",
      content: ev.text || "",
      model: resolved.label,
      prompt_tokens: ev.prompt_tokens,
      new_tokens: ev.new_tokens,
      ms: ev.ms,
    };
  } catch (e) {
    pushLog({ kind: "fail", cmd: "chat.generate", preview: e.message });
    return { ok: false, status: 502, error: e.message, content: e.message };
  }
}

app.post("/api/chat", async (req, res) => {
  const body = req.body || {};
  const result = await runLlmChat(body);
  if (!result.ok) return res.status(result.status || 502).json(result);
  res.json(result);
});

app.post("/api/chat/stream", async (req, res) => {
  const body = req.body || {};
  res.setHeader("Content-Type", "text/event-stream");
  res.setHeader("Cache-Control", "no-cache");
  res.setHeader("Connection", "keep-alive");
  res.flushHeaders?.();

  const writeEv = (obj) => {
    res.write(`data: ${JSON.stringify(obj)}\n\n`);
  };

  try {
    const result = await runLlmChat(
      { ...body, stream: true },
      {
        onToken: (ev) => writeEv({ type: "token", text: ev.text || "", token: ev.token }),
      },
    );
    if (!result.ok) {
      writeEv({ type: "error", error: result.error || "failed" });
      return res.end();
    }
    writeEv({
      type: "done",
      content: result.content,
      model: result.model,
      prompt_tokens: result.prompt_tokens,
      new_tokens: result.new_tokens,
      ms: result.ms,
    });
    res.end();
  } catch (e) {
    writeEv({ type: "error", error: e.message });
    res.end();
  }
});

app.post("/api/chat/reset", async (req, res) => {
  const resolved = resolveChatModel(req.body || {});
  if (resolved.error) return res.status(404).json(resolved);
  try {
    const worker = workerForModel(resolved);
    await worker.reset();
    res.json({ ok: true, model: resolved.label });
  } catch (e) {
    res.status(502).json({ error: e.message });
  }
});

app.post("/api/bench", async (req, res) => {
  const bench = createBenchLauncher({ repoRoot: REPO_ROOT, configuredBin: config.benchBin });
  if (!bench.display) {
    return res.status(404).json({
      error: "uaii_bench not found",
      hint: "Build with -DUAII_BUILD_BENCHMARKS=ON or set UAII_BENCH_BIN / settings.benchBin",
    });
  }
  const suite = (req.body && req.body.suite) || "gemm,session,quant";
  const providers = (req.body && req.body.providers) || "all";
  const trials = Number((req.body && req.body.trials) || 7);
  const args = [
    "--suite",
    suite,
    "--providers",
    providers,
    "--trials",
    String(Math.max(3, trials)),
    "--warmup",
    "2",
    "--json",
  ];
  const cwd = bench.cmd === "wsl" ? REPO_ROOT : path.dirname(bench.cmd);
  const result = await runProcess(bench.cmd, [...bench.prefix, ...args], { cwd });
  let parsed = null;
  try {
    parsed = JSON.parse(result.stdout);
  } catch {
    /* keep raw */
  }
  const outPath = path.join(DASH_ROOT, "models", "last_bench.json");
  try {
    if (parsed) fs.writeFileSync(outPath, JSON.stringify(parsed, null, 2));
  } catch {
    /* ignore */
  }
  res.status(result.ok ? 200 : 502).json({ ...result, parsed, saved: parsed ? outPath : null });
});

app.get("/api/logs", (_req, res) => res.json({ logs }));
app.delete("/api/logs", (_req, res) => {
  logs.length = 0;
  res.json({ ok: true });
});

app.post("/api/jobs/:id/cancel", (req, res) => {
  const job = jobs.get(req.params.id);
  if (!job || !job.child) return res.status(404).json({ error: "job not found or finished" });
  try {
    job.child.kill("SIGTERM");
    job.status = "cancelled";
    res.json({ ok: true });
  } catch (e) {
    res.status(500).json({ error: e.message });
  }
});

app.post("/api/jobs/cancel-running", (_req, res) => {
  let cancelled = 0;
  for (const [id, job] of jobs.entries()) {
    if (job.status === "running" && job.child) {
      try {
        job.child.kill("SIGTERM");
        job.status = "cancelled";
        cancelled += 1;
      } catch {
        /* ignore */
      }
      jobs.set(id, job);
    }
  }
  // Also tear down warm chat workers (stops in-flight generate)
  stopAllChatWorkers();
  res.json({ ok: true, cancelled, chatWorkersStopped: true });
});

// --- OpenAI-compatible surface for internal apps (self-host) ---
app.get("/v1/models", (_req, res) => {
  const models = [
    ...listModels()
      .filter((m) => m.kind === "gguf" || m.kind === "uaii-ir")
      .map((m) => ({
        id: `uaii-file-${m.name}`,
        object: "model",
        owned_by: "local",
      })),
    ...DEMOS.map((d) => ({
      id: `uaii-demo-${d}`,
      object: "model",
      owned_by: "uaii-cli-demo",
    })),
  ];
  res.json({ object: "list", data: models });
});

app.post("/v1/chat/completions", async (req, res) => {
  const body = req.body || {};
  const model = String(body.model || "");
  const messages = body.messages || [];
  const maxNew = Number(body.max_tokens || body.max_new_tokens || 64);

  let chatBody = {
    mode: "gguf",
    messages,
    max_new_tokens: maxNew,
    temperature: body.temperature,
    top_p: body.top_p,
    top_k: body.top_k,
    repetition_penalty: body.repetition_penalty,
    seed: body.seed,
    stream: Boolean(body.stream),
  };
  if (!model) {
    return res.status(400).json({
      error: {
        message: "model is required — upload a .gguf and pass its filename (or uaii-file-<name>)",
        type: "invalid_request_error",
      },
    });
  }
  if (model === "uaii-tiny-demo" || model === "uaii-demo-gguf" || model.endsWith("-tiny-demo")) {
    chatBody = { ...chatBody, mode: "gguf", model: "__demo__", demoModel: true };
  } else if (model.startsWith("uaii-demo-")) {
    // Legacy CLI demos (non-LLM smoke)
    chatBody = { mode: "cli-demo", demo: model.replace("uaii-demo-", ""), prompt: "" };
  } else if (model.startsWith("uaii-file-")) {
    chatBody = { ...chatBody, model: model.replace("uaii-file-", "") };
  } else {
    chatBody = { ...chatBody, model };
  }

  const id = `chatcmpl-${randomUUID()}`;

  if (body.stream && chatBody.mode !== "cli-demo") {
    res.setHeader("Content-Type", "text/event-stream");
    res.setHeader("Cache-Control", "no-cache");
    res.flushHeaders?.();
    try {
      const result = await runLlmChat(
        { ...chatBody, stream: true },
        {
          onToken: (ev) => {
            res.write(
              `data: ${JSON.stringify({
                id,
                object: "chat.completion.chunk",
                choices: [{ index: 0, delta: { content: ev.text || "" }, finish_reason: null }],
              })}\n\n`,
            );
          },
        },
      );
      if (!result.ok) {
        res.write(
          `data: ${JSON.stringify({ error: { message: result.error || "failed", type: "server_error" } })}\n\n`,
        );
        return res.end();
      }
      res.write(
        `data: ${JSON.stringify({
          id,
          object: "chat.completion.chunk",
          choices: [{ index: 0, delta: {}, finish_reason: "stop" }],
        })}\n\n`,
      );
      res.write("data: [DONE]\n\n");
      return res.end();
    } catch (e) {
      res.write(
        `data: ${JSON.stringify({ error: { message: e.message, type: "server_error" } })}\n\n`,
      );
      return res.end();
    }
  }

  const result = await runLlmChat(chatBody);
  if (!result.ok) {
    return res.status(result.status || 502).json({
      error: { message: result.error || result.content || "failed", type: "server_error" },
    });
  }
  res.json({
    id,
    object: "chat.completion",
    created: Math.floor(Date.now() / 1000),
    model,
    choices: [
      {
        index: 0,
        message: { role: "assistant", content: result.content || "" },
        finish_reason: "stop",
      },
    ],
    usage: {
      prompt_tokens: result.prompt_tokens || 0,
      completion_tokens: result.new_tokens || 0,
      total_tokens: (result.prompt_tokens || 0) + (result.new_tokens || 0),
    },
  });
});

// Production static UI
const dist = path.join(DASH_ROOT, "web", "dist");
if (fs.existsSync(dist)) {
  app.use(express.static(dist));
  app.get(/^(?!\/api\/)(?!\/v1\/).*/, (_req, res) => {
    res.sendFile(path.join(dist, "index.html"));
  });
} else if (process.env.NODE_ENV === "production") {
  console.warn("web/dist missing — run: npm run build");
}

ensureModelDir();

if (!config.loopback && !config.token) {
  console.error("FATAL: non-loopback bind requires UAII_DASH_TOKEN (or token in uaii-dash.json)");
  process.exit(1);
}

const server = app.listen(config.port, config.bind, () => {
  const url = `http://${config.bind}:${config.port}`;
  console.log(`UAII Dashboard v${VERSION}`);
  console.log(`  URL:    ${url}`);
  console.log(`  Mode:   ${config.loopback ? "local" : "self-host"}`);
  console.log(`  Auth:   ${config.authRequired ? "required" : "open (loopback)"}`);
  console.log(`  uaii:   ${resolveUaiiBin()}`);
  console.log(`  bench:  ${resolveBenchBin() || "(not found)"}`);
  console.log(`  models: ${config.modelDir}`);

  // Run GPU probe immediately on startup (synchronous, 15s cap) so the first
  // chat request doesn't pay the detection cost.
  if (config.backend === "auto" || !config.backend) {
    setImmediate(() => {
      const be = probeGpuBackend();
      console.log(`  backend: ${be}${detectedBackendDetails ? " — " + detectedBackendDetails : ""}`);
    });
  } else {
    console.log(`  backend: ${config.backend} (explicit)`);
  }
});

server.on("error", (err) => {
  console.error(err);
  process.exit(1);
});

function shutdown() {
  stopAllChatWorkers();
  server.close(() => process.exit(0));
  setTimeout(() => process.exit(0), 2000).unref?.();
}
process.on("SIGINT", shutdown);
process.on("SIGTERM", shutdown);
