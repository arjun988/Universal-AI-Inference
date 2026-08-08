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

    this._stderr = "";
    this._lastError = "";
    this.child.stderr.on("data", (d) => {
      const s = d.toString();
      this._stderr += s;
      if (s.trim()) console.error("[uaii-chat]", s.trim());
    });
    this.child.on("error", (err) => {
      this._lastError = err.message;
      this.#failAll(err.message);
      this.child = null;
      this.ready = false;
    });
    this.child.on("close", (code) => {
      const detail =
        this._lastError ||
        (this._stderr && this._stderr.trim()) ||
        "";
      const msg = detail
        ? `chat worker exited (${code}): ${detail.slice(0, 400)}`
        : `chat worker exited (${code})`;
      this.#failAll(msg);
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
    if (ev.event === "error" && !ev.id) {
      // Startup / load failure before any request id — fail all pending.
      this._lastError = ev.error || "uaii chat error";
      console.error("[uaii-chat]", this._lastError);
      this.#failAll(this._lastError);
      return;
    }
    const id = ev.id;
    const p = id ? this.pending.get(id) : null;
    if (!p) {
      // Unknown id — log but don't crash; may be a late token from a cancelled request.
      if (ev.event === "error") {
        console.error("[uaii-chat] unmatched error event:", ev.error || ev);
      }
      return;
    }
    if (ev.event === "token") {
      if (p.onToken) p.onToken(ev);
      return;
    }
    if (ev.event === "done") {
      this.pending.delete(id);
      this.busy = false;
      if (p._timer) clearTimeout(p._timer);
      p.resolve(ev);
      return;
    }
    if (ev.event === "error") {
      this.pending.delete(id);
      this.busy = false;
      if (p._timer) clearTimeout(p._timer);
      p.reject(new Error(ev.error || "generate failed"));
      return;
    }
    if (ev.event === "reset") {
      this.pending.delete(id);
      this.busy = false;
      if (p._timer) clearTimeout(p._timer);
      p.resolve(ev);
    }
  }

  async waitReady(timeoutMs = 180000) {
    this.start();
    const t0 = Date.now();
    while (!this.ready) {
      if (!this.child) {
        const detail = this._lastError || (this._stderr && this._stderr.trim().slice(0, 400));
        throw new Error(
          detail
            ? `chat worker failed to start: ${detail}`
            : "chat worker failed to start (process exited before emitting ready)",
        );
      }
      const elapsed = Date.now() - t0;
      if (elapsed > timeoutMs) {
        const detail = this._lastError || (this._stderr && this._stderr.trim().slice(0, 400));
        throw new Error(
          detail
            ? `chat worker ready timeout (${timeoutMs}ms): ${detail}`
            : `chat worker ready timeout after ${timeoutMs}ms — model may still be loading; check Logs`,
        );
      }
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
    // Wait for any in-flight request, but cap at 5 minutes to avoid deadlock.
    const busyDeadline = Date.now() + 300000;
    while (this.busy) {
      if (Date.now() > busyDeadline) throw new Error("chat worker busy timeout — previous request stalled");
      await new Promise((r) => setTimeout(r, 25));
    }
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
      // Safety timeout: if the C++ side stalls and never emits done/error, reject
      // after 10 minutes so the UI unblocks.
      const timer = setTimeout(() => {
        if (this.pending.has(id)) {
          this.pending.delete(id);
          this.busy = false;
          reject(new Error("generate request timed out after 10 minutes — check Logs"));
        }
      }, 600000);
      try {
        this.child.stdin.write(JSON.stringify(req) + "\n");
      } catch (e) {
        clearTimeout(timer);
        this.pending.delete(id);
        this.busy = false;
        reject(e);
      }
      // Attach timer so we can clear it in #onLine when done/error arrives
      this.pending.get(id) && (this.pending.get(id)._timer = timer);
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
