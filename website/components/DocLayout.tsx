import Link from "next/link";

const groups = [
  {
    label: "Start",
    items: [
      { href: "/docs/getting-started/", label: "Quick start" },
      { href: "/docs/huggingface/", label: "Hugging Face" },
      { href: "/docs/features/", label: "Features" },
      { href: "/docs/dashboard/", label: "Operator UI" },
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
  {
    label: "Community",
    items: [{ href: "/docs/contribute/", label: "Contribute" }],
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
              <nav className="side-group" key={g.label} aria-label={g.label}>
                <p className="side-label">{g.label}</p>
                <div className="side-links">
                  {g.items.map((s) => (
                    <Link
                      key={s.href}
                      href={s.href}
                      aria-current={s.href === active ? "page" : undefined}
                      className={s.href === active ? "active" : undefined}
                    >
                      {s.label}
                    </Link>
                  ))}
                </div>
              </nav>
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
