import Link from "next/link";

const groups = [
  {
    label: "Start",
    items: [
      { href: "/docs/getting-started/", label: "Quick start" },
      { href: "/docs/features/", label: "Features" },
      { href: "/docs/benchmarks/", label: "Benchmarks" },
      { href: "/docs/examples/", label: "Examples" },
      { href: "/docs/configuration/", label: "Configuration" },
    ],
  },
  {
    label: "Integrate",
    items: [
      { href: "/docs/cli/", label: "CLI" },
      { href: "/docs/python/", label: "Python SDK" },
      { href: "/docs/c-api/", label: "C API" },
    ],
  },
  {
    label: "Reference",
    items: [
      { href: "/docs/backends/", label: "Backends" },
      { href: "/docs/architecture/", label: "Architecture" },
      { href: "/docs/plugins/", label: "Plugins" },
    ],
  },
];

export function DocLayout({
  title,
  active,
  lede,
  children,
}: {
  title: string;
  active: string;
  lede?: string;
  children: React.ReactNode;
}) {
  return (
    <div className="doc-page">
      <div className="doc-shell">
        <div className="doc">
          <aside className="side" aria-label="Documentation">
            {groups.map((g) => (
              <div key={g.label}>
                <div className="side-label">{g.label}</div>
                {g.items.map((s) => (
                  <Link
                    key={s.href}
                    href={s.href}
                    className={s.href === active ? "active" : undefined}
                  >
                    {s.label}
                  </Link>
                ))}
              </div>
            ))}
          </aside>
          <article className="content">
            <h1>{title}</h1>
            {lede ? <p className="lede">{lede}</p> : null}
            {children}
          </article>
        </div>
      </div>
    </div>
  );
}
