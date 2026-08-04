import Link from "next/link";

export default function HomePage() {
  return (
    <main>
      <section className="hero">
        <h1>UAII Runtime</h1>
        <p>
          Universal AI Inference Runtime — any model → UAII IR → any hardware.
          Static documentation for the C API, Python SDK, and CLI.
        </p>
        <div className="cta-row">
          <Link className="btn btn-primary" href="/docs/getting-started/">
            Get started
          </Link>
          <Link className="btn btn-ghost" href="/docs/python/">
            Python SDK
          </Link>
        </div>
      </section>

      <section className="grid">
        <div className="tile">
          <h3>Load</h3>
          <p>Open GGUF, Safetensors, or UAII IR through one session API.</p>
        </div>
        <div className="tile">
          <h3>Run</h3>
          <p>Execute on CPU or host-fallback GPU backends with the same graph.</p>
        </div>
        <div className="tile">
          <h3>Profile</h3>
          <p>Capture kernel/IO timelines as Chrome Trace JSON from Python or CLI.</p>
        </div>
      </section>
    </main>
  );
}
