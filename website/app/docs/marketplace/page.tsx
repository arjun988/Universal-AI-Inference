import { DocLayout } from "@/components/DocLayout";
import Link from "next/link";

export const metadata = { title: "Marketplace" };

export default function Page() {
  return (
    <DocLayout
      title="Plugin marketplace"
      active="/docs/marketplace/"
      lede="No hosted registry ships with UAII today. Extensions load through the local plugin ABI."
    >
      <p>
        The <Link href="/docs/plugins/">Plugins</Link> guide covers the required ABI exports, the
        reference plugins in the repository, and how to load them with{" "}
        <code>uaii doctor --load-plugins</code>.
      </p>
    </DocLayout>
  );
}
