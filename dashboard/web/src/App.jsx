import { useCallback, useEffect, useMemo, useRef, useState } from "react";

const NAV_GROUPS = [
  {
    label: "Workspace",
    items: [
      { id: "chat", label: "Chat", icon: "chat" },
      { id: "models", label: "Models", icon: "models" },
    ],
  },
  {
    label: "Operations",
    items: [
      { id: "runtime", label: "Runtime", icon: "runtime" },
      { id: "bench", label: "Benchmarks", icon: "bench" },
      { id: "logs", label: "Logs", icon: "logs" },
      { id: "settings", label: "Settings", icon: "settings" },
    ],
  },
];

const PAGE_META = {
  chat: {
    title: "Chat",
    sub: "Run local GGUF models with streaming and sampling controls.",
  },
  models: {
    title: "Models",
    sub: "Import, convert, and manage weights in your model library.",
  },
  runtime: {
    title: "Runtime",
    sub: "Probe binaries, backends, and plugins with doctor.",
  },
  bench: {
    title: "Benchmarks",
    sub: "Absolute microbenchmarks from uaii_bench — cite the JSON.",
  },
  logs: {
    title: "Logs",
    sub: "Recent CLI jobs from chat, doctor, convert, and benches.",
  },
  settings: {
    title: "Settings",
    sub: "Bind address, binaries, GEMM, threads, and operator token.",
  },
};

function Icon({ name }) {
  const common = {
    viewBox: "0 0 24 24",
    fill: "none",
    stroke: "currentColor",
    strokeWidth: "1.75",
    strokeLinecap: "round",
    strokeLinejoin: "round",
    "aria-hidden": true,
  };
  switch (name) {
    case "chat":
      return (
        <svg {...common}>
          <path d="M21 15a2 2 0 0 1-2 2H8l-4 4V5a2 2 0 0 1 2-2h13a2 2 0 0 1 2 2z" />
        </svg>
      );
    case "models":
      return (
        <svg {...common}>
          <path d="M12 2 3 7v10l9 5 9-5V7l-9-5z" />
          <path d="M3 7l9 5 9-5M12 22V12" />
        </svg>
      );
    case "runtime":
      return (
        <svg {...common}>
          <circle cx="12" cy="12" r="3" />
          <path d="M12 2v3M12 19v3M4.2 4.2l2.1 2.1M17.7 17.7l2.1 2.1M2 12h3M19 12h3M4.2 19.8l2.1-2.1M17.7 6.3l2.1-2.1" />
        </svg>
      );
    case "bench":
      return (
        <svg {...common}>
          <path d="M4 19V5M4 19h16M8 17V10M12 17V7M16 17v-5" />
        </svg>
      );
    case "logs":
      return (
        <svg {...common}>
          <path d="M8 6h13M8 12h13M8 18h13M3 6h.01M3 12h.01M3 18h.01" />
        </svg>
      );
    case "settings":
      return (
        <svg {...common}>
          <circle cx="12" cy="12" r="3" />
          <path d="M19.4 15a1.7 1.7 0 0 0 .3 1.8l.1.1a2 2 0 1 1-2.8 2.8l-.1-.1a1.7 1.7 0 0 0-1.8-.3 1.7 1.7 0 0 0-1 1.5V21a2 2 0 1 1-4 0v-.1a1.7 1.7 0 0 0-1-1.5 1.7 1.7 0 0 0-1.8.3l-.1.1a2 2 0 1 1-2.8-2.8l.1-.1a1.7 1.7 0 0 0 .3-1.8 1.7 1.7 0 0 0-1.5-1H3a2 2 0 1 1 0-4h.1a1.7 1.7 0 0 0 1.5-1 1.7 1.7 0 0 0-.3-1.8l-.1-.1a2 2 0 1 1 2.8-2.8l.1.1a1.7 1.7 0 0 0 1.8.3h.1a1.7 1.7 0 0 0 1-1.5V3a2 2 0 1 1 4 0v.1a1.7 1.7 0 0 0 1 1.5 1.7 1.7 0 0 0 1.8-.3l.1-.1a2 2 0 1 1 2.8 2.8l-.1.1a1.7 1.7 0 0 0-.3 1.8v.1a1.7 1.7 0 0 0 1.5 1H21a2 2 0 1 1 0 4h-.1a1.7 1.7 0 0 0-1.5 1z" />
        </svg>
      );
    default:
      return null;
  }
}

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
  const [prompt, setPrompt] = useState("");
  const [systemPrompt, setSystemPrompt] = useState("You are a helpful assistant running on UAII.");
  const [maxNewTokens, setMaxNewTokens] = useState(64);
  const [temperature, setTemperature] = useState(0);
  const [topP, setTopP] = useState(1);
  const [topK, setTopK] = useState(0);
  const [repPenalty, setRepPenalty] = useState(1);
  const [seed, setSeed] = useState("");
  const [streamChat, setStreamChat] = useState(true);
  const [selectedModel, setSelectedModel] = useState("");
  const [messages, setMessages] = useState([]);
  const [busy, setBusy] = useState(false);
  const [jobId, setJobId] = useState("");
  const [error, setError] = useState("");
  const [benchOut, setBenchOut] = useState(null);
  const [output, setOutput] = useState("");
  const [streamPreview, setStreamPreview] = useState("");
  const [theme, setTheme] = useState(() => readTheme());
  const [chatConfigOpen, setChatConfigOpen] = useState(false);
  const chatEndRef = useRef(null);

  const SUGGESTIONS = [
    "Say hello in one short sentence.",
    "List three uses for local LLM inference.",
    "Explain what GGUF is, briefly.",
  ];

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
    const list = data.models || [];
    setModels(list);
    setModelDir(data.modelDir || "");
    const ggufs = list.filter((m) => m.kind === "gguf");
    setSelectedModel((cur) => {
      if (cur && ggufs.some((m) => m.name === cur)) return cur;
      if (cur && list.some((m) => m.name === cur)) return cur;
      return ggufs[0]?.name || "";
    });
  }, []);

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
    setError("");
    setJobId("");
    setStreamPreview("");

    if (chatMode === "gguf") {
      const ggufs = models.filter((m) => m.kind === "gguf");
      if (!ggufs.length || !selectedModel) {
        setError("No model selected. Upload a .gguf in Models, then pick it here.");
        return;
      }
    }
    if (chatMode === "ir" && !selectedModel) {
      setError("No model selected. Upload a .gguf or .uaii.json in Models first.");
      return;
    }

    setBusy(true);
    const userMsg = { role: "user", content: prompt };
    const nextMessages = [...messages, userMsg];
    setMessages(nextMessages);

    const payload = {
      mode: chatMode,
      prompt,
      system: systemPrompt,
      demo,
      model: selectedModel,
      messages: nextMessages.map((m) => ({ role: m.role, content: m.content })),
      max_new_tokens: maxNewTokens,
      temperature,
      top_p: topP,
      top_k: topK,
      repetition_penalty: repPenalty,
      stream: streamChat && chatMode === "gguf",
      input: "x=1,2,3,4",
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
    if (!selectedModel) return;
    try {
      await api("/api/chat/reset", {
        method: "POST",
        body: JSON.stringify({ model: selectedModel }),
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
            <div className="brand-mark">UA</div>
            <div className="brand-text">
              <strong>UAII</strong>
              <span>Operator UI</span>
            </div>
          </div>
          <button className="theme-toggle" type="button" onClick={toggleTheme} aria-label="Toggle theme">
            <span className="dot" />
            {theme === "dark" ? "Dark" : "Light"} theme
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
          <button className="btn primary" type="submit" style={{ width: "100%" }}>
            Unlock console
          </button>
        </form>
      </div>
    );
  }

  const meta = PAGE_META[page] || { title: page, sub: "" };
  const gemmRows = benchOut?.parsed?.gemm_by_provider || [];
  const ggufModels = models.filter((m) => m.kind === "gguf");
  const ggufCount = ggufModels.length;
  const hasChatModel = chatMode !== "gguf" || Boolean(selectedModel && ggufCount);

  return (
    <div className="app">
      <aside className="nav">
        <div className="brand">
          <div className="brand-mark">UA</div>
          <div className="brand-text">
            <strong>UAII</strong>
            <span>Operator UI</span>
          </div>
        </div>

        {NAV_GROUPS.map((group) => (
          <div className="nav-section" key={group.label}>
            <div className="nav-section-label">{group.label}</div>
            <nav className="nav-list" aria-label={group.label}>
              {group.items.map((n) => (
                <button
                  key={n.id}
                  type="button"
                  className={page === n.id ? "active" : ""}
                  onClick={() => setPage(n.id)}
                >
                  <Icon name={n.icon} />
                  {n.label}
                </button>
              ))}
            </nav>
          </div>
        ))}

        <div className="nav-tools">
          <button className="theme-toggle" type="button" onClick={toggleTheme} aria-label="Toggle theme">
            <span className="dot" />
            {theme === "dark" ? "Dark" : "Light"}
          </button>
        </div>

        <div className="nav-foot">
          <div className="nav-status">
            <div className="status-line">
              <span className={`status-dot ${health?.uaiiBinExists ? "on" : "off"}`} />
              {health?.uaiiBinExists ? "Runtime ready" : "Runtime missing"}
            </div>
            <div className="status-line">
              <span className="status-dot on" />
              v{health?.version || "…"} · {health?.mode || "…"}
            </div>
          </div>
          <div className="nav-foot-note">Console over the uaii CLI</div>
        </div>
      </aside>

      <div className="main">
        <header className={`top ${page === "chat" ? "top-compact" : ""}`}>
          <div className="top-left">
            {page !== "chat" ? (
              <>
                <p className="top-kicker">UAII / {meta.title}</p>
                <h1>{meta.title}</h1>
                <p className="top-sub">{meta.sub}</p>
              </>
            ) : (
              <>
                <h1>Chat</h1>
                <p className="top-sub top-sub-inline">Local inference · streaming · sampling</p>
              </>
            )}
          </div>
          <div className="top-meta">
            <span className={`pill ${health?.uaiiBinExists ? "ok" : "bad"}`}>
              {health?.uaiiBinExists ? "uaii ready" : "uaii missing"}
            </span>
            {health?.uaiiLaunchMode ? <span className="pill">{health.uaiiLaunchMode}</span> : null}
            <span className="pill">
              {health?.bind}:{health?.port}
            </span>
            {busy ? <span className="pill">busy</span> : null}
          </div>
        </header>

        <div className={`content ${page === "chat" ? "content-chat" : ""}`}>
          {error ? (
            <div className="alert" role="alert">
              <div>
                <strong>Error</strong>
                <div>{error}</div>
              </div>
              <button className="btn ghost sm" type="button" onClick={() => setError("")}>
                Dismiss
              </button>
            </div>
          ) : null}

          {page === "chat" && (
            <div className="chat-stage">
              <div className="chat-bar">
                <div className="chat-bar-left">
                  <div className="seg seg-tight" role="group" aria-label="Primary mode">
                    <button
                      type="button"
                      className={chatMode === "gguf" ? "active" : ""}
                      onClick={() => setChatMode("gguf")}
                    >
                      Chat
                    </button>
                    <button
                      type="button"
                      className={
                        chatMode === "tokenize" || chatMode === "ir" || chatMode === "cli-demo"
                          ? "active"
                          : ""
                      }
                      onClick={() => {
                        if (
                          chatMode !== "tokenize" &&
                          chatMode !== "ir" &&
                          chatMode !== "cli-demo"
                        ) {
                          setChatMode("tokenize");
                        }
                      }}
                    >
                      Tools
                    </button>
                  </div>

                  {chatMode === "gguf" && (
                    <select
                      className="chat-select"
                      value={selectedModel}
                      onChange={(e) => setSelectedModel(e.target.value)}
                      aria-label="Model"
                    >
                      {!ggufCount ? (
                        <option value="">No models — upload a .gguf</option>
                      ) : (
                        <>
                          {!selectedModel ? <option value="">Select a model…</option> : null}
                          {ggufModels.map((m) => (
                            <option key={m.name} value={m.name}>
                              {m.name}
                            </option>
                          ))}
                        </>
                      )}
                    </select>
                  )}

                  {(chatMode === "tokenize" || chatMode === "ir" || chatMode === "cli-demo") && (
                    <>
                      <select
                        className="chat-select"
                        value={chatMode}
                        onChange={(e) => setChatMode(e.target.value)}
                        aria-label="Tool"
                      >
                        <option value="tokenize">Tokenize</option>
                        <option value="ir">Run IR</option>
                        <option value="cli-demo">CLI demo</option>
                      </select>
                      {chatMode === "ir" && (
                        <select
                          className="chat-select"
                          value={selectedModel}
                          onChange={(e) => setSelectedModel(e.target.value)}
                          aria-label="IR model"
                        >
                          {!models.length ? (
                            <option value="">No models — upload first</option>
                          ) : (
                            <>
                              {!selectedModel ? <option value="">Select a model…</option> : null}
                              {models
                                .filter((m) => m.kind === "uaii-ir" || m.kind === "gguf")
                                .map((m) => (
                                  <option key={m.name} value={m.name}>
                                    {m.name}
                                  </option>
                                ))}
                            </>
                          )}
                        </select>
                      )}
                      {chatMode === "cli-demo" && (
                        <select
                          className="chat-select"
                          value={demo}
                          onChange={(e) => setDemo(e.target.value)}
                          aria-label="Demo"
                        >
                          {DEMOS.map((d) => (
                            <option key={d} value={d}>
                              {d}
                            </option>
                          ))}
                        </select>
                      )}
                    </>
                  )}
                </div>

                <div className="chat-bar-right">
                  {chatMode === "gguf" && (
                    <label className="chat-inline-field">
                      <span>Tokens</span>
                      <input
                        type="number"
                        min={1}
                        max={8192}
                        value={maxNewTokens}
                        onChange={(e) => setMaxNewTokens(Number(e.target.value) || 64)}
                      />
                    </label>
                  )}
                  <button
                    className={`btn ghost sm ${chatConfigOpen ? "active-toggle" : ""}`}
                    type="button"
                    onClick={() => setChatConfigOpen((v) => !v)}
                  >
                    Configure
                  </button>
                  <button
                    className="btn ghost sm"
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
                </div>
              </div>

              {chatConfigOpen && (
                <div className="chat-config">
                  <div className="chat-config-grid">
                    {chatMode === "gguf" && (
                      <div className="field" style={{ margin: 0, gridColumn: "1 / -1" }}>
                        <label>System prompt</label>
                        <textarea
                          rows={2}
                          value={systemPrompt}
                          onChange={(e) => setSystemPrompt(e.target.value)}
                        />
                      </div>
                    )}
                    {chatMode === "gguf" && (
                      <>
                        <div className="field" style={{ margin: 0 }}>
                          <label>Temperature</label>
                          <input
                            type="number"
                            min={0}
                            max={2}
                            step={0.05}
                            value={temperature}
                            onChange={(e) => setTemperature(Number(e.target.value))}
                          />
                        </div>
                        <div className="field" style={{ margin: 0 }}>
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
                        <div className="field" style={{ margin: 0 }}>
                          <label>Top-k</label>
                          <input
                            type="number"
                            min={0}
                            max={200}
                            value={topK}
                            onChange={(e) => setTopK(Number(e.target.value) || 0)}
                          />
                        </div>
                        <div className="field" style={{ margin: 0 }}>
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
                        <div className="field" style={{ margin: 0 }}>
                          <label>Seed</label>
                          <input
                            type="text"
                            inputMode="numeric"
                            placeholder="random"
                            value={seed}
                            onChange={(e) => setSeed(e.target.value)}
                          />
                        </div>
                      </>
                    )}
                    <div className="chat-config-actions">
                      <button className="btn sm" type="button" disabled={busy} onClick={runDoctor}>
                        Doctor
                      </button>
                      <span className="chat-config-hint">
                        {temperature > 0
                          ? `Sampling · temp ${temperature}`
                          : "Greedy decode (temp 0)"}
                      </span>
                    </div>
                  </div>
                </div>
              )}

              <div className="chat-surface">
                <div className="chat-thread" aria-live="polite">
                  {messages.length === 0 && !streamPreview ? (
                    <div className="chat-empty">
                      <div className="chat-empty-mark">UA</div>
                      {!hasChatModel ? (
                        <>
                          <h3>Upload a model to chat</h3>
                          <p>
                            Chat needs a local <code>.gguf</code> file. Import one in Models, then
                            select it above.
                          </p>
                          <div className="suggest-row">
                            <button
                              type="button"
                              className="btn primary"
                              onClick={() => setPage("models")}
                            >
                              Go to Models
                            </button>
                          </div>
                        </>
                      ) : (
                        <>
                          <h3>What do you want to run?</h3>
                          <p>Pick a model above, then send a prompt. Streaming shows tokens live.</p>
                          <div className="suggest-row">
                            {SUGGESTIONS.map((s) => (
                              <button
                                key={s}
                                type="button"
                                className="suggest-chip"
                                onClick={() => setPrompt(s)}
                              >
                                {s}
                              </button>
                            ))}
                          </div>
                        </>
                      )}
                    </div>
                  ) : (
                    <div className="chat-messages">
                      {messages.map((m, i) => (
                        <div key={i} className={`msg ${m.role}`}>
                          <div className="msg-avatar">{m.role === "user" ? "You" : "UA"}</div>
                          <div className="msg-body">
                            <div className="msg-role">{m.role === "user" ? "You" : "Assistant"}</div>
                            <pre>{m.content}</pre>
                          </div>
                        </div>
                      ))}
                      {streamPreview ? (
                        <div className="msg assistant streaming">
                          <div className="msg-avatar">UA</div>
                          <div className="msg-body">
                            <div className="msg-role">
                              <span className="live" />
                              Assistant
                            </div>
                            <pre>{streamPreview}</pre>
                          </div>
                        </div>
                      ) : null}
                      <div ref={chatEndRef} />
                    </div>
                  )}
                </div>

                <div className="composer-dock">
                  <div className="composer-shell">
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
                      placeholder={
                        hasChatModel
                          ? "Ask your local model…"
                          : "Upload a .gguf in Models before chatting…"
                      }
                      rows={2}
                      disabled={chatMode === "gguf" && !hasChatModel}
                    />
                    <div className="composer-actions">
                      <label className="check">
                        <input
                          type="checkbox"
                          checked={streamChat}
                          onChange={(e) => setStreamChat(e.target.checked)}
                          disabled={chatMode === "gguf" && !hasChatModel}
                        />
                        Stream
                      </label>
                      <span className="composer-hint">
                        <span className="kbd">⌘</span>
                        <span className="kbd">↵</span>
                      </span>
                      {busy && (
                        <button className="btn danger sm" type="button" onClick={cancelJob}>
                          Stop
                        </button>
                      )}
                      <button
                        className="btn primary"
                        type="button"
                        disabled={busy || !prompt.trim() || (chatMode === "gguf" && !hasChatModel)}
                        onClick={sendChat}
                      >
                        {busy ? "Running…" : "Run"}
                      </button>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          )}

          {page === "models" && (
            <>
              <div className="stats-row">
                <div className="stat">
                  <div className="stat-label">Files</div>
                  <div className="stat-value">{models.length}</div>
                  <div className="stat-hint">In library</div>
                </div>
                <div className="stat">
                  <div className="stat-label">GGUF</div>
                  <div className="stat-value">{ggufCount}</div>
                  <div className="stat-hint">Ready for chat</div>
                </div>
                <div className="stat">
                  <div className="stat-label">Directory</div>
                  <div className="stat-value" style={{ fontSize: "0.92rem", wordBreak: "break-all" }}>
                    {modelDir ? modelDir.split(/[/\\]/).pop() : "…"}
                  </div>
                  <div className="stat-hint">{modelDir || "not set"}</div>
                </div>
              </div>
              <div className="panel">
                <div className="panel-head">
                  <div>
                    <p className="panel-title">Library</p>
                    <p className="panel-desc">
                      Import GGUF / IR, convert, or open in Chat.
                    </p>
                  </div>
                  <div className="row" style={{ margin: 0 }}>
                    <label className="btn primary sm">
                      Import
                      <input type="file" hidden onChange={onUpload} />
                    </label>
                    <button className="btn sm" type="button" onClick={() => refreshModels()}>
                      Refresh
                    </button>
                  </div>
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
                            <p>Import a .gguf / .uaii.json to use in Chat.</p>
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
                                className="btn sm"
                                type="button"
                                disabled={busy}
                                onClick={() => runModel(m.name)}
                              >
                                Run / convert
                              </button>
                              <button
                                className="btn ghost sm"
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
                <div className="panel-head">
                  <div>
                    <p className="panel-title">Doctor</p>
                    <p className="panel-desc">
                      Honest probe from <code>uaii doctor --load-plugins</code>.
                    </p>
                  </div>
                  <button className="btn primary sm" type="button" disabled={busy} onClick={runDoctor}>
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
                <div className="panel-head">
                  <div>
                    <p className="panel-title">Microbench</p>
                    <p className="panel-desc">
                      Short <code>uaii_bench</code> trials for the console — cite CI JSON for publish.
                    </p>
                  </div>
                  <div className="row" style={{ margin: 0 }}>
                    <button className="btn primary sm" type="button" disabled={busy} onClick={runBench}>
                      {busy ? "Running…" : "Run"}
                    </button>
                    {busy && page === "bench" && (
                      <button className="btn danger sm" type="button" onClick={cancelJob}>
                        Stop
                      </button>
                    )}
                    <span className={`pill ${health?.benchBinExists ? "ok" : "bad"}`}>
                      {health?.benchBinExists ? "bench found" : "bench missing"}
                    </span>
                  </div>
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
                <div className="panel-head">
                  <div>
                    <p className="panel-title">Job log</p>
                    <p className="panel-desc">{logs.length} recent entries</p>
                  </div>
                  <div className="row" style={{ margin: 0 }}>
                    <button className="btn sm" type="button" onClick={refreshLogs}>
                      Refresh
                    </button>
                    <button
                      className="btn ghost sm"
                      type="button"
                      onClick={async () => {
                        await api("/api/logs", { method: "DELETE" });
                        refreshLogs();
                      }}
                    >
                      Clear
                    </button>
                  </div>
                </div>
              </div>
              {logs.length === 0 ? (
                <div className="chat-empty">
                  <h3>No jobs yet</h3>
                  <p>Runs from Chat, Doctor, and Benchmarks show up here.</p>
                </div>
              ) : (
                <div className="log-list">
                  {logs.map((l, i) => (
                    <div className="log-item" key={`${l.ts}-${i}`}>
                      <div className="log-meta">
                        <span className={`pill ${l.kind === "ok" ? "ok" : "bad"}`}>{l.kind}</span>
                        <span>{l.ts}</span>
                        {l.ms != null ? <span>{l.ms} ms</span> : null}
                      </div>
                      <div style={{ fontFamily: "var(--mono)", fontSize: "0.82rem" }}>{l.cmd}</div>
                      {l.preview ? (
                        <div className="pre" style={{ marginTop: "0.5rem", maxHeight: "8rem" }}>
                          {l.preview}
                        </div>
                      ) : null}
                    </div>
                  ))}
                </div>
              )}
            </>
          )}

          {page === "settings" && settings && (
            <>
              <div className="panel">
                <div className="panel-head">
                  <div>
                    <p className="panel-title">Host & paths</p>
                    <p className="panel-desc">
                      Local: <code>127.0.0.1</code>. LAN: bind <code>0.0.0.0</code> + token, then
                      restart. Config: <code>{settings.configPath}</code>
                    </p>
                  </div>
                </div>
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
  -d '{"model":"uaii-file-your-model.gguf","messages":[{"role":"user","content":"hi"}]}'`}
              </div>
            </>
          )}
        </div>
      </div>
    </div>
  );
}
