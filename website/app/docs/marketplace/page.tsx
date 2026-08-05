import { DocLayout } from "@/components/DocLayout";
import Link from "next/link";

export const metadata = { title: "Marketplace" };

export default function Page() {
  return (
    <DocLayout
      title="Plugin marketplace"
      active="/docs/marketplace/"
      lede="No marketplace service ships with UAII today. Use the local plugin ABI instead."
    >
      <p>
        See the <Link href="/docs/plugins/">Plugins</Link> guide for ABI exports, example
        plugins, and how to load them with <code>uaii doctor --load-plugins</code>.
      </p>
    </DocLayout>
  );
}
