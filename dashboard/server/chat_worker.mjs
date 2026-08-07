import { spawn } from "child_process";
import { randomUUID } from "crypto";
import readline from "readline";

/**
 * Warm `uaii chat --jsonl` process: loads the GGUF once, serves generate requests.
 * `launch` is { cmd, prefix, mode, toModelPath } from uaii_launch.mjs.
 */
export class ChatWorker {
  constructor({
    launch,
    uaiiBin,
    modelPath,
    demo = false,
    backend = "cpu",
    maxContext = 0,
    env = {},
  }) {
    this.launch = launch || { cmd: uaiiBin || "uaii", prefix: [], mode: "native" };
    this.modelPath = modelPath || "";
    this.demo = demo;
    this.backend = backend;
    this.maxContext = maxContext;
    this.env = env;
    this.child = null;
    this.ready = false;
    this.busy = false;
    this.pending = new Map();
  }

  key() {
    return this.demo ? "__demo__" : this.modelPath;
  }

  start() {
    if (this.child) return;
    const uaiiArgs = ["chat", "--jsonl", "--backend", this.backend || "cpu", "--log-level", "error"];
    if (this.demo) uaiiArgs.push("--demo");
    else uaiiArgs.push("--model", this.modelPath);
    if (this.maxContext > 0) uaiiArgs.push("--max-context", String(this.maxContext));

    const cmd = this.launch.cmd;
    const args = [...(this.launch.prefix || []), ...uaiiArgs];

    this.child = spawn(cmd, args, {
      env: this.env,
      stdio: ["pipe", "pipe", "pipe"],
      windowsHide: true,
    });

    const rl = readline.createInterface({ input: this.child.stdout });
    rl.on("line", (line) => this.#onLine(line));

    this.child.stderr.on("data", (d) => {
      const s = d.toString();
      if (s.trim()) console.error("[uaii-chat]", s.trim());
    });
    this.child.on("error", (err) => {
      this.#failAll(err.message);
      this.child = null;
      this.ready = false;
    });
    this.child.on("close", (code) => {
      this.#failAll(`chat worker exited (${code})`);
      this.child = null;
      this.ready = false;
      this.busy = false;
    });
  }

  stop() {
    if (!this.child) return;
    try {
      this.child.stdin.write(JSON.stringify({ cmd: "quit" }) + "\n");
    } catch {
      /* ignore */
    }
    try {
      this.child.kill("SIGTERM");
    } catch {
      /* ignore */
    }
    this.child = null;
    this.ready = false;
  }

  #failAll(msg) {
    for (const [, p] of this.pending) p.reject(new Error(msg));
    this.pending.clear();
  }

  #onLine(line) {
    let ev;
    try {
      ev = JSON.parse(line);
    } catch {
      return;
    }
    if (ev.event === "ready") {
      this.ready = true;
      return;
    }
    const id = ev.id;
    const p = id ? this.pending.get(id) : null;
    if (!p) return;
    if (ev.event === "token") {
      if (p.onToken) p.onToken(ev);
      return;
    }
    if (ev.event === "done") {
      this.pending.delete(id);
      this.busy = false;
      p.resolve(ev);
      return;
    }
    if (ev.event === "error") {
      this.pending.delete(id);
      this.busy = false;
      p.reject(new Error(ev.error || "generate failed"));
      return;
    }
    if (ev.event === "reset") {
      this.pending.delete(id);
      this.busy = false;
      p.resolve(ev);
    }
  }

  async waitReady(timeoutMs = 180000) {
    this.start();
    const t0 = Date.now();
    while (!this.ready) {
      if (!this.child) throw new Error("chat worker failed to start");
      if (Date.now() - t0 > timeoutMs) throw new Error("chat worker ready timeout");
      await new Promise((r) => setTimeout(r, 50));
    }
  }

  async generate({
    prompt,
    system = "",
    maxNewTokens = 64,
    temperature,
    topP,
    topK,
    repetitionPenalty,
    seed,
    stream = false,
    onToken = null,
  }) {
    await this.waitReady();
    while (this.busy) await new Promise((r) => setTimeout(r, 25));
    this.busy = true;
    const id = randomUUID();
    const req = {
      cmd: "generate",
      id,
      prompt,
      system,
      max_new_tokens: maxNewTokens,
      stream: Boolean(stream || onToken),
    };
    if (temperature != null && Number.isFinite(Number(temperature))) {
      req.temperature = Number(temperature);
    }
    if (topP != null && Number.isFinite(Number(topP))) req.top_p = Number(topP);
    if (topK != null && Number.isFinite(Number(topK))) req.top_k = Number(topK);
    if (repetitionPenalty != null && Number.isFinite(Number(repetitionPenalty))) {
      req.repetition_penalty = Number(repetitionPenalty);
    }
    if (seed != null && seed !== "" && Number.isFinite(Number(seed))) {
      req.seed = Number(seed);
    }
    return new Promise((resolve, reject) => {
      this.pending.set(id, { resolve, reject, onToken });
      try {
        this.child.stdin.write(JSON.stringify(req) + "\n");
      } catch (e) {
        this.pending.delete(id);
        this.busy = false;
        reject(e);
      }
    });
  }

  async reset() {
    await this.waitReady();
    const id = randomUUID();
    this.busy = true;
    return new Promise((resolve, reject) => {
      this.pending.set(id, { resolve, reject, onToken: null });
      try {
        this.child.stdin.write(JSON.stringify({ cmd: "reset", id }) + "\n");
      } catch (e) {
        this.pending.delete(id);
        this.busy = false;
        reject(e);
      }
    });
  }
}

const workers = new Map();

export function getChatWorker(key, factory) {
  let w = workers.get(key);
  if (!w) {
    w = factory();
    workers.set(key, w);
  }
  return w;
}

export function stopAllChatWorkers() {
  for (const w of workers.values()) w.stop();
  workers.clear();
}
