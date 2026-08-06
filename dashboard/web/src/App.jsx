import { useCallback, useEffect, useState } from "react";

const NAV = [
  { id: "chat", label: "Chat / Run" },
  { id: "models", label: "Models" },
  { id: "runtime", label: "Runtime" },
  { id: "logs", label: "Logs" },
  { id: "settings", label: "Settings" },
];

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

async function api(path, opts) {
  const res = await fetch(path, {
    headers: { "Content-Type": "application/json", ...(opts?.headers || {}) },
    ...opts,
  });
  const data = await res.json().catch(() => ({}));
  if (!res.ok) {
    const err = new Error(data.error || res.statusText || "request failed");
    err.data = data;
    throw err;
  }
  return data;
}

export default function App() {
  const [page, setPage] = useState("chat");
  const [health, setHealth] = useState(null);
  const [doctor, setDoctor] = useState(null);
  const [models, setModels] = useState([]);
  const [modelDir, setModelDir] = useState("");
  const [logs, setLogs] = useState([]);
  const [settings, setSettings] = useState(null);
  const [demo, setDemo] = useState("gguf");
  const [busy, setBusy] = useState(false);
  const [output, setOutput] = useState("");
  const [error, setError] = useState("");

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
  }, []);

  const refreshLogs = useCallback(async () => {
    const data = await api("/api/logs");
    setLogs(data.logs || []);
  }, []);

  const refreshSettings = useCallback(async () => {
    setSettings(await api("/api/settings"));
  }, []);

  useEffect(() => {
    refreshHealth();
    refreshModels().catch(() => {});
    refreshLogs().catch(() => {});
    refreshSettings().catch(() => {});
  }, [refreshHealth, refreshModels, refreshLogs, refreshSettings]);

  async function runDoctor() {
    setBusy(true);
    setError("");
    try {
      const data = await api("/api/doctor");
      setDoctor(data);
      setOutput(data.stdout || data.stderr || "");
      await refreshLogs();
      await refreshHealth();
    } catch (e) {
      setError(e.message);
      setDoctor(e.data || null);
      setOutput((e.data && (e.data.stdout || e.data.stderr)) || e.message);
    } finally {
      setBusy(false);
    }
  }

  async function runDemo() {
    setBusy(true);
    setError("");
    setOutput("");
    try {
      const data = await api("/api/run/demo", {
        method: "POST",
        body: JSON.stringify({ demo }),
      });
      setOutput(
        [data.stdout, data.stderr].filter(Boolean).join("\n--- stderr ---\n") ||
          JSON.stringify(data, null, 2),
      );
      await refreshLogs();
      await refreshModels();
    } catch (e) {
      setError(e.message);
      setOutput((e.data && (e.data.stdout || e.data.stderr)) || e.message);
    } finally {
      setBusy(false);
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
      setOutput(
        data.message
          ? `${data.message}\n\n${data.convert?.stdout || ""}`
          : [data.stdout, data.stderr].filter(Boolean).join("\n"),
      );
      await refreshLogs();
      await refreshModels();
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
      const res = await fetch("/api/models/upload", { method: "POST", body: fd });
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

  async function saveSettings(ev) {
    ev.preventDefault();
    const fd = new FormData(ev.target);
    setBusy(true);
    try {
      await api("/api/settings", {
        method: "PUT",
        body: JSON.stringify({
          uaiiBin: fd.get("uaiiBin"),
          modelDir: fd.get("modelDir"),
          threads: Number(fd.get("threads") || 0),
          gemm: fd.get("gemm"),
          backend: fd.get("backend"),
          token: fd.get("token"),
        }),
      });
      await refreshSettings();
      await refreshHealth();
      await refreshModels();
    } catch (e) {
      setError(e.message);
    } finally {
      setBusy(false);
    }
  }

  const titles = {
    chat: "Chat / Run",
    models: "Model library",
    runtime: "Runtime",
    logs: "Job log",
    settings: "Settings",
  };

  return (
    <div className="app">
      <aside className="nav">
        <div className="brand">
          UAII
          <span>Dashboard</span>
        </div>
        <nav className="nav-list">
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
        <div className="nav-foot">
          Local / self-host console
          <br />
          Thin shell over <code>uaii</code>
        </div>
      </aside>

      <div className="main">
        <header className="top">
          <h1>{titles[page]}</h1>
          <div className="row" style={{ margin: 0 }}>
            <span
              className={`pill ${health?.uaiiBinExists || health?.uaiiBin === "uaii" ? "ok" : "bad"}`}
            >
              {health?.uaiiBinExists
                ? "uaii found"
                : health?.error
                  ? "server down"
                  : "uaii missing"}
            </span>
            <span className="pill">
              {health?.bind || "…"}:{health?.port || "…"}
            </span>
          </div>
        </header>

        <div className="content">
          {error ? (
            <p className="lede" style={{ color: "var(--danger)" }}>
              {error}
            </p>
          ) : null}

          {page === "chat" && (
            <>
              <p className="lede">
                Run built-in demos through the UAII CLI, or convert a local GGUF from the Models
                page. Inference always goes through <code>uaii</code> — this UI does not reimplement
                the runtime.
              </p>
              <div className="row">
                <div className="field" style={{ margin: 0, minWidth: 180 }}>
                  <label>Demo</label>
                  <select value={demo} onChange={(e) => setDemo(e.target.value)}>
                    {DEMOS.map((d) => (
                      <option key={d} value={d}>
                        {d}
                      </option>
                    ))}
                  </select>
                </div>
                <button className="btn primary" type="button" disabled={busy} onClick={runDemo}>
                  {busy ? "Running…" : "Run"}
                </button>
                <button className="btn" type="button" disabled={busy} onClick={runDoctor}>
                  Doctor
                </button>
              </div>
              <div className="pre">{output || "Output appears here."}</div>
            </>
          )}

          {page === "models" && (
            <>
              <p className="lede">
                Models directory: <code>{modelDir || "…"}</code>
              </p>
              <div className="row">
                <label className="btn">
                  Import file
                  <input type="file" hidden onChange={onUpload} />
                </label>
                <button className="btn" type="button" onClick={() => refreshModels()}>
                  Refresh
                </button>
              </div>
              <div className="table-wrap">
                <table>
                  <thead>
                    <tr>
                      <th>Name</th>
                      <th>Kind</th>
                      <th>Size</th>
                      <th>Modified</th>
                      <th />
                    </tr>
                  </thead>
                  <tbody>
                    {models.length === 0 ? (
                      <tr>
                        <td colSpan={5} style={{ color: "var(--muted)" }}>
                          No models yet. Import a .gguf / .uaii.json or run demo gguf.
                        </td>
                      </tr>
                    ) : (
                      models.map((m) => (
                        <tr key={m.name}>
                          <td>{m.name}</td>
                          <td>{m.kind}</td>
                          <td>{m.mib} MiB</td>
                          <td>{new Date(m.mtime).toLocaleString()}</td>
                          <td>
                            <button
                              className="btn"
                              type="button"
                              disabled={busy}
                              onClick={() => runModel(m.name)}
                            >
                              Run / convert
                            </button>
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
              <p className="lede">
                Honest environment probe from <code>uaii doctor --load-plugins</code>.
              </p>
              <div className="row">
                <button className="btn primary" type="button" disabled={busy} onClick={runDoctor}>
                  {busy ? "Probing…" : "Run doctor"}
                </button>
              </div>
              <div className="field">
                <label>Binary</label>
                <input readOnly value={health?.uaiiBin || ""} />
              </div>
              <div className="pre">
                {doctor
                  ? doctor.stdout || doctor.stderr || JSON.stringify(doctor, null, 2)
                  : "Click Run doctor."}
              </div>
            </>
          )}

          {page === "logs" && (
            <>
              <div className="row">
                <button className="btn" type="button" onClick={refreshLogs}>
                  Refresh
                </button>
                <button
                  className="btn"
                  type="button"
                  onClick={async () => {
                    await api("/api/logs", { method: "DELETE" });
                    refreshLogs();
                  }}
                >
                  Clear
                </button>
              </div>
              <div>
                {logs.length === 0 ? (
                  <p className="lede">No jobs yet.</p>
                ) : (
                  logs.map((l, i) => (
                    <div className="log-item" key={`${l.ts}-${i}`}>
                      <div className="log-meta">
                        <span className={l.kind}>{l.kind}</span> · {l.ts}
                        {l.ms != null ? ` · ${l.ms} ms` : ""}
                        {l.code != null ? ` · exit ${l.code}` : ""}
                      </div>
                      <div style={{ fontFamily: "var(--mono)", fontSize: "0.82rem" }}>
                        {l.cmd}
                      </div>
                      {l.preview ? (
                        <div className="pre" style={{ marginTop: "0.5rem", maxHeight: "8rem" }}>
                          {l.preview}
                        </div>
                      ) : null}
                    </div>
                  ))
                )}
              </div>
            </>
          )}

          {page === "settings" && settings && (
            <>
              <p className="lede">
                Persists to <code>{settings.configPath}</code>. Default bind is localhost; set a
                token before exposing on LAN.
              </p>
              <form onSubmit={saveSettings}>
                <div className="field">
                  <label>uaii binary path</label>
                  <input name="uaiiBin" defaultValue={settings.uaiiBin || ""} />
                </div>
                <div className="field">
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
                  <label>UAII_GEMM (ref / openblas / onednn)</label>
                  <input name="gemm" defaultValue={settings.gemm || ""} placeholder="optional" />
                </div>
                <div className="field">
                  <label>UAII_NUM_THREADS (0 = default)</label>
                  <input name="threads" type="number" defaultValue={settings.threads || 0} />
                </div>
                <div className="field">
                  <label>Auth token (required if non-loopback)</label>
                  <input
                    name="token"
                    type="password"
                    defaultValue=""
                    placeholder={settings.tokenSet ? "(unchanged if left blank — enter to set)" : "optional"}
                  />
                </div>
                <button className="btn primary" type="submit" disabled={busy}>
                  Save
                </button>
              </form>
            </>
          )}
        </div>
      </div>
    </div>
  );
}
