import { DocLayout } from "@/components/DocLayout";

export const metadata = { title: "CLI" };

export default function Page() {
  return (
    <DocLayout title="CLI reference" active="/docs/cli/">
      <p>
        Binary name: <code>uaii</code>. Global flags: <code>--config</code>,{" "}
        <code>--log-level</code>, <code>--load-plugins</code>.
      </p>
      <pre>{`uaii doctor
uaii validate <ir>
uaii inspect <ir>
uaii graph <ir> --format plan
uaii convert model.gguf -o model.uaii.json
uaii run --demo toy_mlp|optimize|streaming|parity|profile|quant
uaii run <ir> --input x=1,2,3,4 --backend cpu
uaii profile --demo --output uaii_profile.json
uaii benchmark --demo
uaii cache status|clear`}</pre>
    </DocLayout>
  );
}
