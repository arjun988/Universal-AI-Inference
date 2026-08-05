import Link from "next/link";
import { GITHUB_REPO_URL } from "@/lib/site";

export function SiteFooter() {
  return (
    <footer className="site-footer">
      <div className="rail">
        <div>
          <div className="foot-brand">UAII</div>
          <p className="foot-note">
            Universal AI Inference Runtime — any model to any hardware. MIT licensed.
          </p>
          <p className="foot-note" style={{ marginTop: "0.85rem" }}>
            <a href={GITHUB_REPO_URL} target="_blank" rel="noopener noreferrer">
              github.com/arjun988/Universal-AI-Inference
            </a>
          </p>
        </div>
        <div>
          <h4>Product</h4>
          <Link href="/docs/features/">Features</Link>
          <Link href="/docs/benchmarks/">Benchmarks</Link>
          <Link href="/docs/backends/">Backends</Link>
          <Link href="/docs/examples/">Examples</Link>
          <Link href="/docs/configuration/">Configuration</Link>
        </div>
        <div>
          <h4>Developers</h4>
          <Link href="/docs/getting-started/">Quick start</Link>
          <Link href="/docs/cli/">CLI</Link>
          <Link href="/docs/python/">Python SDK</Link>
          <Link href="/docs/c-api/">C API</Link>
          <a href={GITHUB_REPO_URL} target="_blank" rel="noopener noreferrer">
            GitHub
          </a>
        </div>
      </div>
    </footer>
  );
}
