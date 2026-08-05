import Link from "next/link";

export function SiteFooter() {
  return (
    <footer className="site-footer">
      <div className="rail">
        <div>
          <div className="foot-brand">UAII</div>
          <p className="foot-note">
            Universal AI Inference Runtime — any model to any hardware. MIT licensed.
          </p>
        </div>
        <div>
          <h4>Product</h4>
          <Link href="/docs/features/">Features</Link>
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
          <Link href="/docs/architecture/">Architecture</Link>
        </div>
      </div>
    </footer>
  );
}
