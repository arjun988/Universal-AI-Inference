"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { ThemeToggle } from "./ThemeToggle";

const links = [
  { href: "/docs/features/", label: "Product" },
  { href: "/docs/examples/", label: "Examples" },
  { href: "/docs/backends/", label: "Backends" },
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
