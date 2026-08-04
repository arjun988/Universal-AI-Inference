import Link from "next/link";

const side = [
  { href: "/docs/getting-started/", label: "Getting started" },
  { href: "/docs/python/", label: "Python SDK" },
  { href: "/docs/c-api/", label: "C API stability" },
  { href: "/docs/cli/", label: "CLI reference" },
  { href: "/docs/architecture/", label: "Architecture" },
  { href: "/docs/marketplace/", label: "Plugin marketplace" },
];

export function DocLayout({
  title,
  active,
  children,
}: {
  title: string;
  active: string;
  children: React.ReactNode;
}) {
  return (
    <div className="doc">
      <aside className="side">
        {side.map((s) => (
          <Link
            key={s.href}
            href={s.href}
            className={s.href === active ? "active" : undefined}
          >
            {s.label}
          </Link>
        ))}
      </aside>
      <article className="content">
        <h1>{title}</h1>
        {children}
      </article>
    </div>
  );
}
