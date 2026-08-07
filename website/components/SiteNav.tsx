"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { GITHUB_REPO_URL } from "@/lib/site";
import { ThemeToggle } from "./ThemeToggle";

const links = [
  { href: "/docs/features/", label: "Product" },
  { href: "/docs/dashboard/", label: "Dashboard" },
  { href: "/docs/benchmarks/", label: "Benchmarks" },
  { href: "/docs/examples/", label: "Examples" },
  { href: "/docs/getting-started/", label: "Docs" },
];

export function SiteNav() {
  const pathname = usePathname();
  const isHome = pathname === "/";

  return (
    <div className="nav-wrap">
      <div className="rail">
        <header className="nav">
          <Link href="/" className="brand">
            UAII
            <small>Runtime</small>
          </Link>

          <nav className="nav-links" aria-label="Primary">
            {links.map((l) => {
              const active =
                !isHome && (pathname === l.href || pathname?.startsWith(l.href));
              return (
                <Link
                  key={l.href}
                  href={l.href}
                  className={active ? "active" : undefined}
                >
                  {l.label}
                </Link>
              );
            })}
          </nav>

          <div className="nav-actions">
            <a
              className="btn btn-secondary btn-sm github-link"
              href={GITHUB_REPO_URL}
              target="_blank"
              rel="noopener noreferrer"
            >
              <svg
                viewBox="0 0 24 24"
                width="16"
                height="16"
                aria-hidden="true"
                fill="currentColor"
              >
                <path d="M12 .5C5.73.5.5 5.74.5 12.02c0 5.1 3.29 9.43 7.86 10.96.58.11.79-.25.79-.56 0-.28-.01-1.02-.02-2-3.2.7-3.88-1.54-3.88-1.54-.53-1.36-1.3-1.72-1.3-1.72-1.06-.73.08-.72.08-.72 1.17.08 1.79 1.2 1.79 1.2 1.04 1.78 2.73 1.27 3.4.97.1-.76.41-1.27.74-1.56-2.55-.29-5.23-1.28-5.23-5.7 0-1.26.45-2.29 1.19-3.1-.12-.29-.52-1.46.11-3.04 0 0 .97-.31 3.18 1.18a11.1 11.1 0 0 1 5.8 0c2.2-1.49 3.17-1.18 3.17-1.18.64 1.58.24 2.75.12 3.04.74.81 1.18 1.84 1.18 3.1 0 4.43-2.69 5.4-5.25 5.69.42.36.79 1.08.79 2.18 0 1.57-.01 2.84-.01 3.23 0 .31.21.68.8.56A10.53 10.53 0 0 0 23.5 12C23.5 5.74 18.27.5 12 .5z" />
              </svg>
              GitHub
            </a>
            <ThemeToggle />
            <Link className="btn btn-primary btn-sm" href="/docs/getting-started/">
              Quick start
            </Link>
          </div>
        </header>
      </div>
    </div>
  );
}
