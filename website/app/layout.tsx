import type { Metadata } from "next";
import { Figtree, Fraunces, IBM_Plex_Mono } from "next/font/google";
import "./globals.css";
import { SiteNav } from "@/components/SiteNav";

const body = Figtree({
  subsets: ["latin"],
  variable: "--font-body-loaded",
});

const display = Fraunces({
  subsets: ["latin"],
  variable: "--font-display-loaded",
});

const mono = IBM_Plex_Mono({
  subsets: ["latin"],
  weight: ["400", "500"],
  variable: "--font-mono-loaded",
});

export const metadata: Metadata = {
  title: {
    default: "UAII Runtime Docs",
    template: "%s · UAII Runtime",
  },
  description:
    "Documentation for Universal AI Inference Runtime — any model to any hardware.",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" className={`${body.variable} ${display.variable} ${mono.variable}`}>
      <body
        style={
          {
            ["--font-body" as string]: "var(--font-body-loaded), system-ui, sans-serif",
            ["--font-display" as string]: "var(--font-display-loaded), Georgia, serif",
            ["--font-mono" as string]: "var(--font-mono-loaded), ui-monospace, monospace",
          } as React.CSSProperties
        }
      >
        <div className="shell">
          <SiteNav />
          {children}
          <footer className="footer">
            Universal AI Inference Runtime · MIT · Static docs (Next.js export, no backend)
          </footer>
        </div>
      </body>
    </html>
  );
}
