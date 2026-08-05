import type { Metadata } from "next";
import { IBM_Plex_Mono, Plus_Jakarta_Sans, Syne } from "next/font/google";
import "./globals.css";
import { SiteFooter } from "@/components/SiteFooter";
import { SiteNav } from "@/components/SiteNav";
import { ThemeProvider } from "@/components/ThemeProvider";

const body = Plus_Jakarta_Sans({
  subsets: ["latin"],
  weight: ["400", "500", "600", "700"],
  variable: "--font-body-loaded",
});

const display = Syne({
  subsets: ["latin"],
  weight: ["600", "700", "800"],
  variable: "--font-display-loaded",
});

const mono = IBM_Plex_Mono({
  subsets: ["latin"],
  weight: ["400", "500"],
  variable: "--font-mono-loaded",
});

export const metadata: Metadata = {
  title: {
    default: "UAII — Universal AI Inference Runtime",
    template: "%s · UAII",
  },
  description:
    "Open-source inference runtime: load models into UAII IR and run on CPU or GPU through one session API, CLI, C ABI, and Python SDK.",
};

const themeBoot = `(function(){try{var k='uaii-theme';var t=localStorage.getItem(k);if(t!=='dark'&&t!=='light'){t=window.matchMedia('(prefers-color-scheme: light)').matches?'light':'dark';}document.documentElement.setAttribute('data-theme',t);}catch(e){document.documentElement.setAttribute('data-theme','dark');}})();`;

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html
      lang="en"
      data-theme="dark"
      suppressHydrationWarning
      className={`${body.variable} ${display.variable} ${mono.variable}`}
    >
      <head>
        <script dangerouslySetInnerHTML={{ __html: themeBoot }} />
      </head>
      <body
        style={
          {
            ["--font-body" as string]:
              "var(--font-body-loaded), 'Segoe UI', system-ui, sans-serif",
            ["--font-display" as string]:
              "var(--font-display-loaded), 'Segoe UI', system-ui, sans-serif",
            ["--font-mono" as string]:
              "var(--font-mono-loaded), ui-monospace, monospace",
          } as React.CSSProperties
        }
      >
        <ThemeProvider>
          <div className="app">
            <SiteNav />
            {children}
            <SiteFooter />
          </div>
        </ThemeProvider>
      </body>
    </html>
  );
}
