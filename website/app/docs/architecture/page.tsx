import { DocLayout } from "@/components/DocLayout";

export const metadata = { title: "Architecture" };

export default function Page() {
  return (
    <DocLayout title="Architecture" active="/docs/architecture/">
      <p>
        Pipeline: Model → Loader → <strong>UAII IR</strong> → Planner → Scheduler → Backend
        kernels, with storage streaming and profiler traces.
      </p>
      <ul>
        <li>
          <code>uaii-ir</code> — graph IR, validator, serialize
        </li>
        <li>
          <code>uaii-runtime</code> — session lifecycle
        </li>
        <li>
          <code>uaii-planner</code> — fusion, memory/storage plans
        </li>
        <li>
          <code>uaii-backends</code> — CPU + GPU host-fallback backends
        </li>
        <li>
          <code>uaii-capi</code> — stable C ABI for SDKs
        </li>
      </ul>
      <p>
        Deep dive: repository file <code>docs/architecture.md</code>.
      </p>
    </DocLayout>
  );
}
