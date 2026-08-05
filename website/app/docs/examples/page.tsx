import { DocLayout } from "@/components/DocLayout";

export const metadata = { title: "Examples" };

export default function Page() {
  return (
    <DocLayout
      title="Examples"
      active="/docs/examples/"
      lede="Copy-paste oriented examples for the CLI, IR execution, conversion, tokenization, and profiling."
    >
      <h2>Environment check</h2>
      <pre>{`uaii doctor
uaii doctor --load-plugins
uaii --config configs/uaii.toml --log-level debug doctor`}</pre>

      <h2>Validate and inspect IR</h2>
      <pre>{`uaii validate examples/ir/toy_mlp.uaii.json
uaii inspect examples/ir/toy_mlp.uaii.json
uaii graph examples/ir/toy_mlp.uaii.json --format text
uaii graph examples/ir/toy_mlp.uaii.json --format plan`}</pre>

      <h2>Built-in demos</h2>
      <pre>{`uaii run --demo toy_mlp
uaii run --demo tiny_block
uaii run --demo gguf
uaii run --demo safetensors
uaii run --demo moe
uaii run --demo parity
uaii run --demo optimize
uaii run --demo streaming
uaii run --demo profile
uaii run --demo quant --format int8`}</pre>

      <h2>Execute hand-authored IR</h2>
      <pre>{`uaii run examples/ir/toy_mlp.uaii.json \\
  --weight-init ones \\
  --input x=1,2,3,4 \\
  --output y_prob

# Optional backend (host-fallback without vendor SDK)
uaii run examples/ir/toy_mlp.uaii.json \\
  --backend cuda --force-host-fallback \\
  --weight-init ones --input x=1,2,3,4 --output y_prob`}</pre>

      <h2>Convert models</h2>
      <pre>{`uaii convert model.gguf -o model.uaii.json
uaii convert model.safetensors -o model.uaii.json
uaii convert model.onnx -o model.uaii.json`}</pre>

      <h2>Tokenize</h2>
      <pre>{`# Demo / simple vocab
uaii tokenize encode hello world

# GPT-2 style BPE
uaii tokenize encode "hello" --bpe vocab.json --merges merges.txt

# SentencePiece (requires UAII_WITH_SENTENCEPIECE)
uaii tokenize encode "hello" --sp tokenizer.model

# From GGUF tokenizer metadata
uaii tokenize encode "hello" --gguf model.gguf`}</pre>

      <h2>Profile and benchmark</h2>
      <pre>{`uaii profile --demo --output uaii_profile.json
uaii benchmark --demo
uaii cache status
uaii cache clear`}</pre>

      <h2>Python</h2>
      <pre>{`import uaii

session = uaii.Session.from_path(
    "examples/ir/toy_mlp.uaii.json",
    weight_init="ones",
    profile=True,
    trace_path="uaii_py_profile.json",
)
session.set_tensor("x", [1.0, 2.0, 3.0, 4.0])
session.run()
print(session.get_tensor("y_prob"))
print(session.profile_summary())`}</pre>

      <h2>C API sketch</h2>
      <pre>{`uaii_session_options opts;
uaii_session_options_init(&opts);
opts.struct_size = sizeof(opts);
opts.backend = "cpu";

uaii_session* s = NULL;
uaii_session_create("model.uaii.json", &opts, &s);
float x[] = {1, 2, 3, 4};
uaii_session_set_f32(s, "x", x, 4);
uaii_session_run(s);
uaii_session_destroy(s);`}</pre>
    </DocLayout>
  );
}
