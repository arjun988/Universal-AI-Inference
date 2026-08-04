import { DocLayout } from "@/components/DocLayout";

export const metadata = { title: "Marketplace" };

export default function Page() {
  return (
    <DocLayout title="Plugin marketplace" active="/docs/marketplace/">
      <p>
        Design notes only — no marketplace service ships with UAII today. Future third-party
        loaders, ops, and backends will publish against the existing C plugin ABI.
      </p>
      <ul>
        <li>ABI-first packages with capability manifests</li>
        <li>Optional signed artifacts and checksum verification</li>
        <li>Multiple registries; no single-vendor lock-in</li>
      </ul>
      <p>
        Full sketch: <code>docs/marketplace.md</code>.
      </p>
    </DocLayout>
  );
}
