import Link from "next/link";

export default function NotFound() {
  return (
    <main className="home-section">
      <div className="rail">
        <p className="section-label">404</p>
        <h2>Page not found</h2>
        <p className="section-lede">That route is not part of the UAII docs site.</p>
        <Link className="btn btn-primary" href="/">
          Back home
        </Link>
      </div>
    </main>
  );
}
