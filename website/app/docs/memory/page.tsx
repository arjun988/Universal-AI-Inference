import { DocLayout } from "@/components/DocLayout";
import Link from "next/link";

export const metadata = { title: "Memory" };

export default function Page() {
  return (
    <DocLayout
      title="Memory"
      active="/docs/memory/"
      lede="One paging dial. Laptop is the default. Phone is the lower stop. Weights stay on disk and the session pages what does not fit."
    >
      <div className="table-wrap">
        <table className="data">
          <thead>
            <tr>
              <th>Preset</th>
              <th>Budget</th>
              <th>KV</th>
              <th>Staging</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td>
                <strong>laptop</strong>
              </td>
              <td>~8 GiB. Shrinks to free RAM when 8 GiB does not fit</td>
              <td>2048 + 4 sink</td>
              <td>2 buffers</td>
            </tr>
            <tr>
              <td>
                <strong>desktop</strong>
              </td>
              <td>~32 GiB, same free-RAM rule</td>
              <td>4096</td>
              <td>2 buffers</td>
            </tr>
            <tr>
              <td>
                <strong>server</strong>
              </td>
              <td>~128 GiB, same free-RAM rule</td>
              <td>8192</td>
              <td>2 buffers</td>
            </tr>
            <tr>
              <td>
                <strong>phone</strong>
              </td>
              <td>~2 GiB, then half of free RAM</td>
              <td>512, or 256 / 128 when the cap is tight</td>
              <td>1 buffer, 4 threads</td>
            </tr>
            <tr>
              <td>
                <strong>none</strong>
              </td>
              <td>No streaming budget</td>
              <td>Grows with context</td>
              <td>Off</td>
            </tr>
          </tbody>
        </table>
      </div>

      <h2>What gets pinned</h2>
      <p>
        Final norm stays resident. Embeddings, <code>lm_head</code>, and the first layers stay
        only while pin plus staging still fits. Experts are never pinned. The LRU uses leftover
        RAM after the trunk. GGUF block-quants stay packed.
      </p>
      <p>
        A single unquantized tensor larger than about half the budget fail-closes. Convert
        Hugging Face BF16 7B and larger to GGUF Q4_K. <code>--ram-gb 8</code> on a phone with 6
        GiB free becomes 3 GiB.
      </p>

      <h2>CLI</h2>
      <pre>{`uaii generate --model model.q4_k.gguf --preset laptop --prompt "hi"
uaii generate --model model.q4_k.gguf --preset phone --prompt "hi"
uaii chat --model model.q4_k.gguf --jsonl --preset phone`}</pre>
      <p>
        <code>uaii doctor</code> prints the resolved phone budget, whether it was clamped, and
        available bytes. Phone builds are in <Link href="/docs/mobile/">Mobile</Link>.
      </p>
    </DocLayout>
  );
}
