import { useCallback, useEffect, useMemo, useRef, useState } from "react";

const NAV = [
  { id: "chat", label: "Chat / Run" },
  { id: "models", label: "Models" },
  { id: "runtime", label: "Runtime" },
  { id: "bench", label: "Benchmarks" },
  { id: "logs", label: "Logs" },
  { id: "settings", label: "Settings" },
];

const THEME_KEY = "uaii_dash_theme";

function readTheme() {
  try {
    const t = localStorage.getItem(THEME_KEY);
    if (t === "light" || t === "dark") return t;
  } catch {
    /* ignore */
  }
  if (typeof window !== "undefined" && window.matchMedia("(prefers-color-scheme: light)").matches) {
    return "light";
  }
  return "dark";
}

function applyTheme(theme) {
  document.documentElement.setAttribute("data-theme", theme);
  try {
    localStorage.setItem(THEME_KEY, theme);
  } catch {
    /* ignore */
  }
}

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

const TOKEN_KEY = "uaii_dash_token";

function getToken() {
  try {
    return localStorage.getItem(TOKEN_KEY) || "";
  } catch {
    return "";
  }
}

function setToken(t) {
  try {
    if (t) localStorage.setItem(TOKEN_KEY, t);
    else localStorage.removeItem(TOKEN_KEY);
  } catch {
    /* ignore */
  }
}

async function api(path, opts = {}) {
  const headers = { ...(opts.headers || {}) };
  if (!(opts.body instanceof FormData)) {
    headers["Content-Type"] = headers["Content-Type"] || "application/json";
  }
  const tok = getToken();
  if (tok) headers.Authorization = `Bearer ${tok}`;
  const res = await fetch(path, { ...opts, headers });
  const data = await res.json().catch(() => ({}));
  if (!res.ok) {
    const err = new Error(data.error || data?.error?.message || res.statusText || "request failed");
    err.data = data;
    err.status = res.status;
    throw err;
  }
  return data;
}

export default function App() {
  const [page, setPage] = useState("chat");
  const [health, setHealth] = useState(null);
  const [auth, setAuth] = useState(null);
  const [loginToken, setLoginToken] = useState("");
  const [doctor, setDoctor] = useState(null);
  const [models, setModels] = useState([]);
  const [modelDir, setModelDir] = useState("");
  const [logs, setLogs] = useState([]);
  const [settings, setSettings] = useState(null);
  const [demo, setDemo] = useState("gguf");
  const [chatMode, setChatMode] = useState("gguf");
  const [prompt, setPrompt] = useState("Hello from UAII operator UI");
  const [systemPrompt, setSystemPrompt] = useState("You are a helpful assistant running on UAII.");
  const [maxNewTokens, setMaxNewTokens] = useState(64);
  const [temperature, setTemperature] = useState(0);
  const [topP, setTopP] = useState(1);
  const [topK, setTopK] = useState(0);
  const [repPenalty, setRepPenalty] = useState(1);
  const [seed, setSeed] = useState("");
  const [streamChat, setStreamChat] = useState(true);
  const [selectedModel, setSelectedModel] = useState("__demo__");
  const [messages, setMessages] = useState([]);
  const [busy, setBusy] = useState(false);
  const [jobId, setJobId] = useState("");
  const [error, setError] = useState("");
  const [benchOut, setBenchOut] = useState(null);
  const [output, setOutput] = useState("");
  const [streamPreview, setStreamPreview] = useState("");
  const [theme, setTheme] = useState(() => readTheme());
  const chatEndRef = useRef(null);

  const authorized = useMemo(() => {
    if (!auth) return false;
    return auth.authorized !== false;
  }, [auth]);

  useEffect(() => {
    applyTheme(theme);
  }, [theme]);

  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: "smooth", block: "end" });
  }, [messages, streamPreview, page]);

  function toggleTheme() {
    setTheme((t) => (t === "dark" ? "light" : "dark"));
  }

  const refreshAuth = useCallback(async () => {
    const data = await api("/api/auth/status");
    setAuth(data);
    return data;
  }, []);

  const refreshHealth = useCallback(async () => {
    try {
      setHealth(await api("/api/health"));
    } catch (e) {
      setHealth({ ok: false, error: e.message });
    }
  }, []);

  const refreshModels = useCallback(async () => {
    const data = await api("/api/models");
    setModels(data.models || []);
    setModelDir(data.modelDir || "");
    if (!selectedModel && data.models?.[0]) setSelectedModel(data.models[0].name);
  }, [selectedModel]);

  const refreshLogs = useCallback(async () => {
    setLogs((await api("/api/logs")).logs || []);
  }, []);

  const refreshSettings = useCallback(async () => {
    setSettings(await api("/api/settings"));
  }, []);

  useEffect(() => {
    (async () => {
      await refreshHealth();
      try {
        const a = await refreshAuth();
        if (a.authorized) {
          await Promise.all([
            refreshModels().catch(() => {}),
            refreshLogs().catch(() => {}),
            refreshSettings().catch(() => {}),
          ]);
        }
      } catch (e) {
        setError(e.message);
      }
    })();
  }, [refreshHealth, refreshAuth, refreshModels, refreshLogs, refreshSettings]);

  async function doLogin(e) {
    e?.preventDefault();
    setError("");
    try {
      await api("/api/auth/verify", {
        method: "POST",
        body: JSON.stringify({ token: loginToken }),
      });
      setToken(loginToken);
      const a = await refreshAuth();
      if (!a.authorized) throw new Error("Token rejected");
      await Promise.all([refreshModels(), refreshLogs(), refreshSettings(), refreshHealth()]);
    } catch (err) {
      setError(err.message || "Login failed");
    }
  }

  async function cancelJob() {
    try {
      if (jobId) {
        await api(`/api/jobs/${jobId}/cancel`, { method: "POST" });
      } else {
        await api("/api/jobs/cancel-running", { method: "POST" });
      }
    } catch {
      /* job may already be finished */
    }
  }

  async function runDoctor() {
    setBusy(true);
    setError("");
    setJobId("");
    try {
      const data = await api("/api/doctor");
      if (data.jobId) setJobId(data.jobId);
      setDoctor(data);
      setOutput(data.stdout || data.stderr || "");
      await refreshLogs();
    } catch (e) {
      setError(e.message);
      setOutput((e.data && (e.data.stdout || e.data.stderr)) || e.message);
    } finally {
      setBusy(false);
      setJobId("");
    }
  }

  async function sendChat() {
    setBusy(true);
    setError("");
    setJobId("");
    setStreamPreview("");
    const userMsg = { role: "user", content: prompt };
    const nextMessages = [...messages, userMsg];
    setMessages(nextMessages);

    const model =
      chatMode === "gguf"
        ? selectedModel || "__demo__"
        : chatMode === "demo"
          ? "__demo__"
          : selectedModel;

    const payload = {
      mode: chatMode === "demo" ? "gguf" : chatMode,
      prompt,
      system: systemPrompt,
      demo,
      model,
      messages: nextMessages.map((m) => ({ role: m.role, content: m.content })),
      max_new_tokens: maxNewTokens,
      temperature,
      top_p: topP,
      top_k: topK,
      repetition_penalty: repPenalty,
      stream: streamChat && (chatMode === "gguf" || chatMode === "demo"),
      input: "x=1,2,3,4",
      demoModel: model === "__demo__" || chatMode === "demo",
    };
    if (seed !== "" && seed != null) {
      const n = Number(seed);
      if (Number.isFinite(n)) payload.seed = n;
    }

    try {
      if (payload.stream) {
        const headers = { "Content-Type": "application/json" };
        const tok = getToken();
        if (tok) headers.Authorization = `Bearer ${tok}`;
        const res = await fetch("/api/chat/stream", {
          method: "POST",
          headers,
          body: JSON.stringify(payload),
        });
        if (!res.ok) {
          const errBody = await res.json().catch(() => ({}));
          throw new Error(errBody.error || res.statusText);
        }
        const reader = res.body.getReader();
        const dec = new TextDecoder();
        let buf = "";
        let full = "";
        while (true) {
          const { done, value } = await reader.read();
          if (done) break;
          buf += dec.decode(value, { stream: true });
          const chunks = buf.split("\n\n");
          buf = chunks.pop() || "";
          for (const chunk of chunks) {
            const line = chunk.split("\n").find((l) => l.startsWith("data: "));
            if (!line) continue;
            let ev;
            try {
              ev = JSON.parse(line.slice(6));
            } catch {
              continue;
            }
            if (ev.type === "token") {
              full += ev.text || "";
              setStreamPreview(full);
            } else if (ev.type === "done") {
              full = ev.content || full;
            } else if (ev.type === "error") {
              throw new Error(ev.error || "stream error");
            }
          }
        }
        setMessages((m) => [...m, { role: "assistant", content: full || "(empty)" }]);
        setOutput(full);
        setStreamPreview("");
      } else {
        const data = await api("/api/chat", {
          method: "POST",
          body: JSON.stringify(payload),
        });
        if (data.jobId) setJobId(data.jobId);
        setMessages((m) => [
          ...m,
          { role: "assistant", content: data.content || data.stdout || "(empty)" },
        ]);
        setOutput(data.content || data.stdout || "");
      }
      await refreshLogs();
      await refreshModels();
    } catch (e) {
      setError(e.message);
      setMessages((m) => [
        ...m,
        { role: "assistant", content: (e.data && (e.data.content || e.data.stderr)) || e.message },
      ]);
      setStreamPreview("");
    } finally {
      setBusy(false);
      setJobId("");
    }
  }

  async function resetChatSession() {
    try {
      await api("/api/chat/reset", {
        method: "POST",
        body: JSON.stringify({
          model: selectedModel || "__demo__",
          demoModel: selectedModel === "__demo__" || chatMode === "demo",
        }),
      });
    } catch (e) {
      setError(e.message);
    }
  }

  async function runModel(name) {
    setBusy(true);
    setError("");
    setPage("chat");
    try {
      const data = await api("/api/run/model", {
        method: "POST",
        body: JSON.stringify({ name }),
      });
      const text = data.message
        ? `${data.message}\n\n${data.convert?.stdout || data.stdout || ""}`
        : [data.stdout, data.stderr].filter(Boolean).join("\n");
      setOutput(text);
      setMessages((m) => [...m, { role: "assistant", content: text }]);
      await refreshLogs();
      await refreshModels();
      if (data.irName) setSelectedModel(data.irName);
    } catch (e) {
      setError(e.message);
      setOutput((e.data && (e.data.stdout || e.data.stderr)) || e.message);
    } finally {
      setBusy(false);
    }
  }

  async function onUpload(ev) {
    const file = ev.target.files?.[0];
    if (!file) return;
    setBusy(true);
    setError("");
    try {
      const fd = new FormData();
      fd.append("file", file);
      const headers = {};
      const tok = getToken();
      if (tok) headers.Authorization = `Bearer ${tok}`;
      const res = await fetch("/api/models/upload", { method: "POST", body: fd, headers });
      const data = await res.json();
      if (!res.ok) throw new Error(data.error || "upload failed");
      await refreshModels();
      await refreshLogs();
    } catch (e) {
      setError(e.message);
    } finally {
      setBusy(false);
      ev.target.value = "";
    }
  }

  async function runBench() {
    setBusy(true);
    setError("");
    setJobId("");
    setBenchOut(null);
    try {
      const data = await api("/api/bench", {
        method: "POST",
        body: JSON.stringify({ suite: "gemm,session,quant", providers: "all", trials: 7 }),
      });
      if (data.jobId) setJobId(data.jobId);
      setBenchOut(data);
      await refreshLogs();
    } catch (e) {
      setError(e.message);
      setBenchOut(e.data || null);
    } finally {
      setBusy(false);
      setJobId("");
    }
  }

  async function saveSettings(ev) {
    ev.preventDefault();
    const fd = new FormData(ev.target);
    setBusy(true);
    setError("");
    try {
      const body = {
        uaiiBin: fd.get("uaiiBin"),
        benchBin: fd.get("benchBin"),
        modelDir: fd.get("modelDir"),
        threads: Number(fd.get("threads") || 0),
        gemm: fd.get("gemm"),
        backend: fd.get("backend"),
        bind: fd.get("bind"),
        port: Number(fd.get("port") || 8787),
      };
      const tok = fd.get("token");
      if (tok) body.token = tok;
      await api("/api/settings", { method: "PUT", body: JSON.stringify(body) });
      if (tok) setToken(String(tok));
      await refreshSettings();
      await refreshHealth();
      await refreshAuth();
    } catch (e) {
      setError(e.message);
    } finally {
      setBusy(false);
    }
  }

  // Login gate for self-host
  if (auth?.authRequired && !authorized) {
    return (
      <div className="login-shell">
        <form className="login-card" onSubmit={doLogin}>
          <div className="brand">
            UAII
            <span>Operator UI</span>
          </div>
          <button className="theme-toggle" type="button" onClick={toggleTheme} aria-label="Toggle theme">
            <span className="dot" />
            {theme === "dark" ? "Dark mode" : "Light mode"} — switch
          </button>
          <p className="lede">Self-host mode — enter the operator token to continue.</p>
          {error ? (
            <div className="alert">
              <div>
                <strong>Error</strong>
                <div>{error}</div>
              </div>
            </div>
          ) : null}
          <div className="field">
            <label>Bearer token</label>
            <input
              type="password"
              value={loginToken}
              onChange={(e) => setLoginToken(e.target.value)}
              autoFocus
              required
              placeholder="Paste token"
            />
          </div>
          <button className="btn primary" type="submit">
            Unlock
          </button>
        </form>
      </div>
    );
  }

  const titles = {
    chat: "Chat / Run",
    models: "Model library",
    runtime: "Runtime",
    bench: "Benchmarks",
    logs: "Job log",
    settings: "Settings",
  };

  const gemmRows = benchOut?.parsed?.gemm_by_provider || [];

  return (
    <div className="app">
      <aside className="nav">
        <div className="brand">
          UAII
          <span>Operator UI</span>
        </div>
        <nav className="nav-list" aria-label="Primary">
          {NAV.map((n) => (
            <button
              key={n.id}
              type="button"
              className={page === n.id ? "active" : ""}
              onClick={() => setPage(n.id)}
            >
              {n.label}
            </button>
          ))}
        </nav>
        <div className="nav-tools">
          <button className="theme-toggle" type="button" onClick={toggleTheme} aria-label="Toggle theme">
            <span className="dot" />
            {theme === "dark" ? "Dark" : "Light"}
          </button>
        </div>
        <div className="nav-foot">
          Mode: {health?.mode || "…"}
          <br />
          v{health?.version || "…"}
          <br />
          Thin shell over uaii
        </div>
      </aside>

      <div className="main">
        <header className="top">
          <h1>{titles[page]}</h1>
          <div className="top-meta">
            <span className={`pill ${health?.uaiiBinExists ? "ok" : "bad"}`}>
              {health?.uaiiBinExists ? "uaii ready" : "uaii missing"}
            </span>
            <span className="pill">{health?.mode || "…"}</span>
            {health?.uaiiLaunchMode ? <span className="pill">{health.uaiiLaunchMode}</span> : null}
            <span className="pill">
              {health?.bind}:{health?.port}
            </span>
          </div>
        </header>

        <div className="content">
          {error ? (
            <div className="alert" role="alert">
              <div>
                <strong>Error</strong>
                <div>{error}</div>
              </div>
              <button className="btn ghost" type="button" onClick={() => setError("")}>
                Dismiss
              </button>
            </div>
          ) : null}

          {page === "chat" && (
            <div className="chat-layout">
              <div className="panel">
                <p className="panel-title">Session</p>
                <p className="lede" style={{ marginBottom: "0.85rem" }}>
                  Local LLM chat via <code>uaii generate</code> / warm <code>uaii chat --jsonl</code>.
                  OpenAI clients: <code>POST /v1/chat/completions</code>.
                </p>
                <div className="toolbar">
                  <div className="field grow" style={{ margin: 0 }}>
                    <label>Mode</label>
                    <select value={chatMode} onChange={(e) => setChatMode(e.target.value)}>
                      <option value="gguf">LLM chat (GGUF)</option>
                      <option value="demo">Tiny demo LLM</option>
                      <option value="tokenize">Tokenize only</option>
                      <option value="ir">Run float IR</option>
                      <option value="cli-demo">CLI smoke demos</option>
                    </select>
                  </div>
                  {(chatMode === "gguf" || chatMode === "ir") && (
                    <div className="field grow" style={{ margin: 0, minWidth: 200 }}>
                      <label>Model</label>
                      <select value={selectedModel} onChange={(e) => setSelectedModel(e.target.value)}>
                        <option value="__demo__">uaii-tiny-demo (built-in)</option>
                        {models
                          .filter((m) =>
                            chatMode === "gguf"
                              ? m.kind === "gguf"
                              : m.kind === "uaii-ir" || m.kind === "gguf",
                          )
                          .map((m) => (
                            <option key={m.name} value={m.name}>
                              {m.name} ({m.mib} MiB)
                            </option>
                          ))}
                      </select>
                    </div>
                  )}
                  {chatMode === "cli-demo" && (
                    <div className="field grow" style={{ margin: 0 }}>
                      <label>CLI demo</label>
                      <select value={demo} onChange={(e) => setDemo(e.target.value)}>
                        {DEMOS.map((d) => (
                          <option key={d} value={d}>
                            {d}
                          </option>
                        ))}
                      </select>
                    </div>
                  )}
                  {(chatMode === "gguf" || chatMode === "demo") && (
                    <div className="field" style={{ margin: 0, width: 130 }}>
                      <label>Max tokens</label>
                      <input
                        type="number"
                        min={1}
                        max={8192}
                        value={maxNewTokens}
                        onChange={(e) => setMaxNewTokens(Number(e.target.value) || 64)}
                      />
                    </div>
                  )}
                  <button className="btn" type="button" disabled={busy} onClick={runDoctor}>
                    Doctor
                  </button>
                </div>
                {(chatMode === "gguf" || chatMode === "demo") && (
                  <>
                    <div className="field" style={{ marginTop: "0.85rem", marginBottom: 0 }}>
                      <label>System prompt</label>
                      <textarea
                        rows={2}
                        value={systemPrompt}
                        onChange={(e) => setSystemPrompt(e.target.value)}
                      />
                    </div>
                    <div className="row wrap" style={{ marginTop: "0.75rem", gap: "0.65rem" }}>
                      <div className="field" style={{ margin: 0, width: 110 }}>
                        <label>Temperature</label>
                        <input
                          type="number"
                          min={0}
                          max={2}
                          step={0.05}
                          value={temperature}
                          onChange={(e) => setTemperature(Number(e.target.value))}
                          title="0 = greedy; &gt;0 samples"
                        />
                      </div>
                      <div className="field" style={{ margin: 0, width: 100 }}>
                        <label>Top-p</label>
                        <input
                          type="number"
                          min={0}
                          max={1}
                          step={0.05}
                          value={topP}
                          onChange={(e) => setTopP(Number(e.target.value))}
                        />
                      </div>
                      <div className="field" style={{ margin: 0, width: 90 }}>
                        <label>Top-k</label>
                        <input
                          type="number"
                          min={0}
                          max={200}
                          value={topK}
                          onChange={(e) => setTopK(Number(e.target.value) || 0)}
                        />
                      </div>
                      <div className="field" style={{ margin: 0, width: 110 }}>
                        <label>Rep. penalty</label>
                        <input
                          type="number"
                          min={1}
                          max={2}
                          step={0.05}
                          value={repPenalty}
                          onChange={(e) => setRepPenalty(Number(e.target.value) || 1)}
                        />
                      </div>
                      <div className="field" style={{ margin: 0, width: 110 }}>
                        <label>Seed</label>
                        <input
                          type="text"
                          inputMode="numeric"
                          placeholder="random"
                          value={seed}
                          onChange={(e) => setSeed(e.target.value)}
                        />
                      </div>
                    </div>
                  </>
                )}
              </div>

              <div className="chat-thread" aria-live="polite">
                {messages.length === 0 && !streamPreview ? (
                  <div className="chat-empty">
                    <h3>Start a run</h3>
                    <p>Pick a model, type a prompt, then Run. Streaming shows tokens as they arrive.</p>
                  </div>
                ) : (
                  <>
                    {messages.map((m, i) => (
                      <div key={i} className={`bubble ${m.role}`}>
                        <div className="bubble-role">{m.role}</div>
                        <pre>{m.content}</pre>
                      </div>
                    ))}
                    {streamPreview ? (
                      <div className="bubble assistant streaming">
                        <div className="bubble-role">assistant · streaming</div>
                        <pre>{streamPreview}</pre>
                      </div>
                    ) : null}
                    <div ref={chatEndRef} />
                  </>
                )}
              </div>

              <div className="composer">
                <div className="field" style={{ marginBottom: 0 }}>
                  <label>Prompt</label>
                  <textarea
                    className="prompt"
                    value={prompt}
                    onChange={(e) => setPrompt(e.target.value)}
                    onKeyDown={(e) => {
                      if (e.key === "Enter" && (e.metaKey || e.ctrlKey) && !busy) {
                        e.preventDefault();
                        sendChat();
                      }
                    }}
                    placeholder="Ask the local model…"
                  />
                </div>
                <div className="composer-actions">
                  <label className="check">
                    <input
                      type="checkbox"
                      checked={streamChat}
                      onChange={(e) => setStreamChat(e.target.checked)}
                    />
                    Stream tokens
                  </label>
                  <button className="btn primary" type="button" disabled={busy || !prompt.trim()} onClick={sendChat}>
                    {busy ? "Generating…" : "Run"}
                  </button>
                  {busy && (
                    <button className="btn danger" type="button" onClick={cancelJob}>
                      Stop
                    </button>
                  )}
                  <button
                    className="btn ghost"
                    type="button"
                    onClick={() => {
                      setMessages([]);
                      setOutput("");
                      setStreamPreview("");
                      resetChatSession();
                    }}
                  >
                    Clear
                  </button>
                  <span className="composer-hint">Ctrl/⌘ + Enter to run</span>
                </div>
              </div>
            </div>
          )}

          {page === "models" && (
            <>
              <div className="panel">
                <p className="panel-title">Library</p>
                <p className="lede" style={{ marginBottom: "0.85rem" }}>
                  Directory: <code>{modelDir || "…"}</code> — import GGUF/IR, convert, run.
                </p>
                <div className="row" style={{ marginBottom: 0 }}>
                  <label className="btn primary">
                    Import file
                    <input type="file" hidden onChange={onUpload} />
                  </label>
                  <button className="btn" type="button" onClick={() => refreshModels()}>
                    Refresh
                  </button>
                  <span className="pill">{models.length} files</span>
                </div>
              </div>
              <div className="table-wrap">
                <table>
                  <thead>
                    <tr>
                      <th>Name</th>
                      <th>Kind</th>
                      <th>Size</th>
                      <th>Modified</th>
                      <th>Actions</th>
                    </tr>
                  </thead>
                  <tbody>
                    {models.length === 0 ? (
                      <tr>
                        <td colSpan={5}>
                          <div className="chat-empty" style={{ padding: "1.5rem" }}>
                            <h3>No models yet</h3>
                            <p>Import a .gguf / .uaii.json, or use Chat → tiny demo.</p>
                          </div>
                        </td>
                      </tr>
                    ) : (
                      models.map((m) => (
                        <tr key={m.name}>
                          <td>
                            <strong>{m.name}</strong>
                          </td>
                          <td>
                            <span className="pill">{m.kind}</span>
                          </td>
                          <td>{m.mib} MiB</td>
                          <td>{new Date(m.mtime).toLocaleString()}</td>
                          <td>
                            <div className="row" style={{ margin: 0 }}>
                              <button
                                className="btn"
                                type="button"
                                disabled={busy}
                                onClick={() => runModel(m.name)}
                              >
                                Run / convert
                              </button>
                              <button
                                className="btn ghost"
                                type="button"
                                onClick={async () => {
                                  if (!window.confirm(`Delete ${m.name}?`)) return;
                                  await api(`/api/models/${encodeURIComponent(m.name)}`, {
                                    method: "DELETE",
                                  });
                                  refreshModels();
                                }}
                              >
                                Delete
                              </button>
                            </div>
                          </td>
                        </tr>
                      ))
                    )}
                  </tbody>
                </table>
              </div>
            </>
          )}

          {page === "runtime" && (
            <>
              <div className="panel">
                <p className="panel-title">Doctor</p>
                <p className="lede" style={{ marginBottom: "0.85rem" }}>
                  Honest probe from <code>uaii doctor --load-plugins</code>.
                </p>
                <div className="row" style={{ marginBottom: "0.85rem" }}>
                  <button className="btn primary" type="button" disabled={busy} onClick={runDoctor}>
                    {busy ? "Probing…" : "Run doctor"}
                  </button>
                </div>
                <div className="field" style={{ marginBottom: 0 }}>
                  <label>Binary</label>
                  <input readOnly value={health?.uaiiBin || ""} />
                </div>
              </div>
              <div className="pre">
                {doctor
                  ? doctor.stdout || doctor.stderr || JSON.stringify(doctor, null, 2)
                  : output || "Click Run doctor."}
              </div>
            </>
          )}

          {page === "bench" && (
            <>
              <div className="panel">
                <p className="panel-title">Microbench</p>
                <p className="lede" style={{ marginBottom: "0.85rem" }}>
                  Runs <code>uaii_bench</code> when built. Short trials for UI responsiveness.
                </p>
                <div className="row" style={{ marginBottom: 0 }}>
                  <button className="btn primary" type="button" disabled={busy} onClick={runBench}>
                    {busy ? "Benchmarking…" : "Run microbench"}
                  </button>
                  {busy && page === "bench" && (
                    <button className="btn danger" type="button" onClick={cancelJob}>
                      Stop
                    </button>
                  )}
                  <span className={`pill ${health?.benchBinExists ? "ok" : "bad"}`}>
                    {health?.benchBinExists ? "bench found" : "bench missing"}
                  </span>
                </div>
              </div>
              {gemmRows.length > 0 && (
                <div className="table-wrap" style={{ marginBottom: "1rem" }}>
                  <table>
                    <thead>
                      <tr>
                        <th>Provider</th>
                        <th>N</th>
                        <th>GFLOP/s</th>
                        <th>Median ms</th>
                      </tr>
                    </thead>
                    <tbody>
                      {gemmRows.flatMap((p) =>
                        (p.sizes || []).map((s) => (
                          <tr key={`${p.provider}-${s.n}`}>
                            <td>
                              {p.provider} ({p.name})
                            </td>
                            <td>{s.n}</td>
                            <td>{s.gflops_median}</td>
                            <td>{s.uaii_ms?.median_ms}</td>
                          </tr>
                        )),
                      )}
                    </tbody>
                  </table>
                </div>
              )}
              <div className="pre">
                {benchOut
                  ? JSON.stringify(
                      benchOut.parsed || { stdout: benchOut.stdout, error: benchOut.stderr },
                      null,
                      2,
                    )
                  : "No bench result yet."}
              </div>
            </>
          )}

          {page === "logs" && (
            <>
              <div className="panel">
                <p className="panel-title">Job log</p>
                <div className="row" style={{ marginBottom: 0 }}>
                  <button className="btn" type="button" onClick={refreshLogs}>
                    Refresh
                  </button>
                  <button
                    className="btn ghost"
                    type="button"
                    onClick={async () => {
                      await api("/api/logs", { method: "DELETE" });
                      refreshLogs();
                    }}
                  >
                    Clear
                  </button>
                  <span className="pill">{logs.length} entries</span>
                </div>
              </div>
              {logs.length === 0 ? (
                <div className="chat-empty">
                  <h3>No jobs yet</h3>
                  <p>Runs from Chat, Doctor, and Benchmarks show up here.</p>
                </div>
              ) : (
                logs.map((l, i) => (
                  <div className="log-item" key={`${l.ts}-${i}`}>
                    <div className="log-meta">
                      <span className={l.kind}>{l.kind}</span> · {l.ts}
                      {l.ms != null ? ` · ${l.ms} ms` : ""}
                    </div>
                    <div style={{ fontFamily: "var(--mono)", fontSize: "0.82rem" }}>{l.cmd}</div>
                    {l.preview ? (
                      <div className="pre" style={{ marginTop: "0.5rem", maxHeight: "8rem" }}>
                        {l.preview}
                      </div>
                    ) : null}
                  </div>
                ))
              )}
            </>
          )}

          {page === "settings" && settings && (
            <>
              <div className="panel">
                <p className="panel-title">Host & paths</p>
                <p className="lede" style={{ marginBottom: "1rem" }}>
                  Local default: <code>127.0.0.1</code>. Self-host: bind <code>0.0.0.0</code> + token,
                  then restart. Config: <code>{settings.configPath}</code>
                </p>
                <form onSubmit={saveSettings}>
                  <div className="settings-grid">
                    <div className="field">
                      <label>Bind address</label>
                      <input name="bind" defaultValue={settings.bind || "127.0.0.1"} />
                    </div>
                    <div className="field">
                      <label>Port</label>
                      <input name="port" type="number" defaultValue={settings.port || 8787} />
                    </div>
                    <div className="field full">
                      <label>uaii binary</label>
                      <input name="uaiiBin" defaultValue={settings.uaiiBin || ""} />
                    </div>
                    <div className="field full">
                      <label>uaii_bench binary</label>
                      <input name="benchBin" defaultValue={settings.benchBin || ""} />
                    </div>
                    <div className="field full">
                      <label>Model directory</label>
                      <input name="modelDir" defaultValue={settings.modelDir || ""} />
                    </div>
                    <div className="field">
                      <label>Backend</label>
                      <select name="backend" defaultValue={settings.backend || "cpu"}>
                        <option value="cpu">cpu</option>
                        <option value="cuda">cuda</option>
                        <option value="metal">metal</option>
                        <option value="vulkan">vulkan</option>
                      </select>
                    </div>
                    <div className="field">
                      <label>UAII_GEMM</label>
                      <input
                        name="gemm"
                        defaultValue={settings.gemm || ""}
                        placeholder="ref | openblas | onednn"
                      />
                    </div>
                    <div className="field">
                      <label>UAII_NUM_THREADS</label>
                      <input name="threads" type="number" defaultValue={settings.threads || 0} />
                    </div>
                    <div className="field">
                      <label>Auth token (self-host)</label>
                      <input
                        name="token"
                        type="password"
                        placeholder={settings.tokenSet ? "set to replace" : "set for LAN"}
                      />
                    </div>
                  </div>
                  <div className="row" style={{ marginTop: "0.5rem", marginBottom: 0 }}>
                    <button className="btn primary" type="submit" disabled={busy}>
                      Save settings
                    </button>
                    <button className="btn ghost" type="button" onClick={toggleTheme}>
                      Theme: {theme}
                    </button>
                  </div>
                </form>
              </div>
              <div className="pre">
                {`# Self-host example
UAII_DASH_BIND=0.0.0.0 UAII_DASH_TOKEN=secret npm start

# OpenAI-compatible
curl http://HOST:8787/v1/models -H "Authorization: Bearer secret"
curl http://HOST:8787/v1/chat/completions -H "Authorization: Bearer secret" \\
  -H "Content-Type: application/json" \\
  -d '{"model":"uaii-tiny-demo","messages":[{"role":"user","content":"hi"}]}'`}
              </div>
            </>
          )}
        </div>
      </div>
    </div>
  );
}
