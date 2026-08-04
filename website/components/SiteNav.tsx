"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";

const links = [
  { href: "/docs/getting-started/", label: "Getting started" },
  { href: "/docs/python/", label: "Python SDK" },
  { href: "/docs/c-api/", label: "C API" },
  { href: "/docs/cli/", label: "CLI" },
  { href: "/docs/architecture/", label: "Architecture" },
  { href: "/docs/marketplace/", label: "Marketplace" },
];

export function SiteNav() {
  const pathname = usePathname();
  return (
    <header className="nav">
      <Link href="/" className="brand">
        UAII <span>Runtime</span>
      </Link>
      <nav className="nav-links" aria-label="Documentation">
        {links.map((l) => {
          const active = pathname === l.href || pathname?.startsWith(l.href);
          return (
            <Link key={l.href} href={l.href} className={active ? "active" : undefined}>
              {l.label}
            </Link>
          );
        })}
      </nav>
    </header>
  );
}
