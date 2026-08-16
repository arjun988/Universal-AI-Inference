import Link from "next/link";

export default function NotFound() {
  return (
    <main className="home-section">
      <div className="rail">
        <p className="section-label">404</p>
        <h2>Page not found</h2>
        <p className="section-lede">
          This page does not exist. Start from the documentation index or return home.
        </p>
        <div className="hero-cta" style={{ margin: 0 }}>
          <Link className="btn btn-primary" href="/">
            Back home
          </Link>
          <Link className="btn btn-secondary" href="/docs/getting-started/">
            Quick start
          </Link>
        </div>
      </div>
    </main>
  );
}
