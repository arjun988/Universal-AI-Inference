import cors from "cors";
import express from "express";
import fs from "fs";
import multer from "multer";
import os from "os";
import path from "path";
import { spawn } from "child_process";
import { fileURLToPath } from "url";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const DASH_ROOT = path.resolve(__dirname, "..");
const REPO_ROOT = path.resolve(DASH_ROOT, "..");

const LOG_LIMIT = 200;
const logs = [];

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
  return {
    bind: process.env.UAII_DASH_BIND || file.bind || "127.0.0.1",
    port: Number(process.env.UAII_DASH_PORT || file.port || 8787),
    uaiiBin: process.env.UAII_BIN || file.uaiiBin || "",
    modelDir: path.resolve(modelDir),
    token: process.env.UAII_DASH_TOKEN || file.token || "",
    threads: Number(process.env.UAII_NUM_THREADS || file.threads || 0),
    gemm: process.env.UAII_GEMM || file.gemm || "",
    backend: process.env.UAII_BACKEND || file.backend || "cpu",
    configPath: cfgPath,
  };
}

let config = loadConfig();

function candidateBins() {
  const exe = process.platform === "win32" ? "uaii.exe" : "uaii";
  const list = [];
  if (config.uaiiBin) list.push(config.uaiiBin);
  list.push(
    path.join(REPO_ROOT, "build", "libs", "uaii-cli", exe),
    path.join(REPO_ROOT, "build", "libs", "uaii-cli", "Release", exe),
    path.join(REPO_ROOT, "build", "Release", exe),
    path.join(REPO_ROOT, "build", exe),
    path.join(REPO_ROOT, "build-wsl", "libs", "uaii-cli", "uaii"),
  );
  return list;
}

function resolveUaiiBin() {
  for (const p of candidateBins()) {
    if (p && fs.existsSync(p)) return path.resolve(p);
  }
  return config.uaiiBin || "uaii";
}

function runUaii(args, opts = {}) {
  const bin = resolveUaiiBin();
  const env = { ...process.env };
  if (config.threads > 0) env.UAII_NUM_THREADS = String(config.threads);
  if (config.gemm) env.UAII_GEMM = config.gemm;

  return new Promise((resolve) => {
    const child = spawn(bin, args, {
      cwd: opts.cwd || REPO_ROOT,
      env,
      shell: process.platform === "win32",
    });
    let stdout = "";
    let stderr = "";
    const t0 = Date.now();
    child.stdout.on("data", (d) => {
      stdout += d.toString();
    });
    child.stderr.on("data", (d) => {
      stderr += d.toString();
    });
    child.on("error", (err) => {
      const result = {
        ok: false,
        code: -1,
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
      const result = {
        ok: code === 0,
        code,
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
        preview: (stdout || stderr).slice(0, 400),
      });
      resolve(result);
    });
  });
}

function ensureModelDir() {
  fs.mkdirSync(config.modelDir, { recursive: true });
}

function listModels() {
  ensureModelDir();
  const exts = new Set([".gguf", ".uaii.json", ".onnx", ".safetensors"]);
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
    const ok =
      [...exts].some((e) => lower.endsWith(e)) || lower.endsWith(".json");
    if (!ok && !lower.endsWith(".gguf")) continue;
    if (!lower.endsWith(".gguf") && !lower.endsWith(".uaii.json") && !lower.endsWith(".onnx") && !lower.endsWith(".safetensors")) {
      continue;
    }
    items.push({
      name,
      path: full,
      bytes: st.size,
      mib: +(st.size / (1024 * 1024)).toFixed(2),
      mtime: st.mtime.toISOString(),
      kind: lower.endsWith(".gguf")
        ? "gguf"
        : lower.endsWith(".uaii.json")
          ? "uaii-ir"
          : lower.endsWith(".onnx")
            ? "onnx"
            : "safetensors",
    });
  }
  items.sort((a, b) => (a.mtime < b.mtime ? 1 : -1));
  return items;
}

function authMiddleware(req, res, next) {
  const bind = config.bind;
  const loopback = bind === "127.0.0.1" || bind === "localhost" || bind === "::1";
  if (loopback || !config.token) return next();
  const hdr = req.headers.authorization || "";
  const token = hdr.startsWith("Bearer ") ? hdr.slice(7) : req.query.token || "";
  if (token !== config.token) {
    return res.status(401).json({ error: "unauthorized" });
  }
  return next();
}

const app = express();
app.use(cors());
app.use(express.json({ limit: "2mb" }));
app.use(authMiddleware);

const upload = multer({
  storage: multer.diskStorage({
    destination: (_req, _file, cb) => {
      ensureModelDir();
      cb(null, config.modelDir);
    },
    filename: (_req, file, cb) => cb(null, path.basename(file.originalname)),
  }),
  limits: { fileSize: 32 * 1024 * 1024 * 1024 },
});

app.get("/api/health", async (_req, res) => {
  const bin = resolveUaiiBin();
  const exists = fs.existsSync(bin) || bin === "uaii";
  res.json({
    ok: true,
    service: "uaii-dashboard",
    version: "0.1.0",
    bind: config.bind,
    port: config.port,
    uaiiBin: bin,
    uaiiBinExists: fs.existsSync(bin),
    modelDir: config.modelDir,
    hostname: os.hostname(),
    platform: process.platform,
    loopbackOnly: config.bind === "127.0.0.1" || config.bind === "localhost",
    authRequired: Boolean(config.token) && config.bind !== "127.0.0.1",
    hint: exists
      ? null
      : "Set UAII_BIN or build the repo (cmake --build build --target uaii).",
  });
});

app.get("/api/settings", (_req, res) => {
  res.json({
    bind: config.bind,
    port: config.port,
    uaiiBin: config.uaiiBin || resolveUaiiBin(),
    modelDir: config.modelDir,
    threads: config.threads,
    gemm: config.gemm,
    backend: config.backend,
    tokenSet: Boolean(config.token),
    configPath: config.configPath,
  });
});

app.put("/api/settings", (req, res) => {
  const body = req.body || {};
  if (typeof body.uaiiBin === "string") config.uaiiBin = body.uaiiBin;
  if (typeof body.modelDir === "string" && body.modelDir) {
    config.modelDir = path.resolve(body.modelDir);
  }
  if (body.threads !== undefined) config.threads = Number(body.threads) || 0;
  if (typeof body.gemm === "string") config.gemm = body.gemm;
  if (typeof body.backend === "string") config.backend = body.backend;
  if (typeof body.token === "string" && body.token.length > 0) config.token = body.token;
  // Persist non-secret-ish settings (token included if provided — local tool)
  const out = {
    bind: config.bind,
    port: config.port,
    uaiiBin: config.uaiiBin,
    modelDir: config.modelDir,
    token: config.token,
    threads: config.threads,
    gemm: config.gemm,
    backend: config.backend,
  };
  try {
    fs.writeFileSync(config.configPath, JSON.stringify(out, null, 2));
  } catch (e) {
    return res.status(500).json({ error: e.message });
  }
  pushLog({ kind: "ok", cmd: "settings.save", preview: config.configPath });
  res.json({ ok: true, settings: out });
});

app.get("/api/doctor", async (_req, res) => {
  const result = await runUaii(["doctor", "--load-plugins"]);
  res.status(result.ok ? 200 : 502).json(result);
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
  res.json({
    ok: true,
    name: req.file.filename,
    path: req.file.path,
    bytes: req.file.size,
  });
});

app.post("/api/run/demo", async (req, res) => {
  const demo = (req.body && req.body.demo) || "toy_mlp";
  const allowed = new Set([
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
  ]);
  if (!allowed.has(demo)) {
    return res.status(400).json({ error: "unknown demo", allowed: [...allowed] });
  }
  const result = await runUaii(["run", "--demo", demo, "--backend", config.backend || "cpu"]);
  res.status(result.ok ? 200 : 502).json(result);
});

app.post("/api/run/model", async (req, res) => {
  const name = req.body && req.body.name;
  if (!name) return res.status(400).json({ error: "name required" });
  const full = path.join(config.modelDir, path.basename(name));
  if (!fs.existsSync(full)) return res.status(404).json({ error: "model not found" });

  const lower = full.toLowerCase();
  if (lower.endsWith(".uaii.json")) {
    const result = await runUaii([
      "run",
      full,
      "--backend",
      config.backend || "cpu",
      "--weight-init",
      "ones",
    ]);
    return res.status(result.ok ? 200 : 502).json(result);
  }
  if (lower.endsWith(".gguf")) {
    // Convert then note — full chat generate CLI may vary; convert is the safe path.
    const outIr = full.replace(/\.gguf$/i, ".uaii.json");
    const conv = await runUaii(["convert", full, "-o", outIr]);
    if (!conv.ok) return res.status(502).json({ step: "convert", ...conv });
    return res.json({
      ok: true,
      step: "convert",
      message:
        "GGUF converted to UAII IR. Use Chat → Run demo gguf for synthetic generate, or run the IR when inputs are wired.",
      convert: conv,
      irPath: outIr,
    });
  }
  return res.status(400).json({ error: "unsupported model kind for run; use .uaii.json or .gguf" });
});

app.get("/api/logs", (_req, res) => {
  res.json({ logs });
});

app.delete("/api/logs", (_req, res) => {
  logs.length = 0;
  res.json({ ok: true });
});

// Production: serve built web UI from dashboard/web/dist
const dist = path.join(DASH_ROOT, "web", "dist");
if (process.env.NODE_ENV === "production" && fs.existsSync(dist)) {
  app.use(express.static(dist));
  app.get("*", (_req, res) => {
    res.sendFile(path.join(dist, "index.html"));
  });
}

ensureModelDir();
const server = app.listen(config.port, config.bind, () => {
  console.log(`UAII Dashboard  http://${config.bind}:${config.port}`);
  console.log(`  uaii:  ${resolveUaiiBin()}`);
  console.log(`  models: ${config.modelDir}`);
  if (config.bind !== "127.0.0.1" && config.bind !== "localhost" && !config.token) {
    console.warn("WARNING: non-loopback bind without UAII_DASH_TOKEN — set a token.");
  }
});

server.on("error", (err) => {
  console.error(err);
  process.exit(1);
});
