import { DocLayout } from "@/components/DocLayout";
import Link from "next/link";
import {
  GITHUB_CODE_OF_CONDUCT_URL,
  GITHUB_CONTRIBUTING_URL,
  GITHUB_GOOD_FIRST_ISSUE_URL,
  GITHUB_ISSUES_URL,
  GITHUB_NEW_ISSUE_URL,
  GITHUB_PULLS_URL,
  GITHUB_REPO_URL,
} from "@/lib/site";

export const metadata = { title: "Contribute" };

export default function Page() {
  return (
    <DocLayout
      title="Contribute"
      active="/docs/contribute/"
      lede="UAII is MIT licensed and developed in the open. Every change starts with a GitHub issue, so scope is agreed before code is written."
    >
      <h2>Where to start</h2>
      <p>
        Pick the entry point that matches how much context you already have. If you are new to
        the codebase, an issue labelled{" "}
        <a href={GITHUB_GOOD_FIRST_ISSUE_URL} target="_blank" rel="noopener noreferrer">
          good first issue
        </a>{" "}
        is the fastest way in.
      </p>
      <div className="table-wrap">
        <table className="data">
          <thead>
            <tr>
              <th>You want to</th>
              <th>Do this</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td>
                <strong>Report a bug</strong>
              </td>
              <td>
                Open a{" "}
                <a href={GITHUB_NEW_ISSUE_URL} target="_blank" rel="noopener noreferrer">
                  bug report
                </a>{" "}
                with OS, compiler, CMake version, and the exact failing command
              </td>
            </tr>
            <tr>
              <td>
                <strong>Propose a feature</strong>
              </td>
              <td>
                Open a{" "}
                <a href={GITHUB_NEW_ISSUE_URL} target="_blank" rel="noopener noreferrer">
                  feature request
                </a>{" "}
                describing the workload it unblocks, not only the API you have in mind
              </td>
            </tr>
            <tr>
              <td>
                <strong>Fix something small</strong>
              </td>
              <td>
                Browse{" "}
                <a
                  href={GITHUB_GOOD_FIRST_ISSUE_URL}
                  target="_blank"
                  rel="noopener noreferrer"
                >
                  good first issues
                </a>{" "}
                and comment to claim one
              </td>
            </tr>
            <tr>
              <td>
                <strong>Add an architecture or operator</strong>
              </td>
              <td>
                Start with <Link href="/docs/plugins/">Plugins</Link>. New formats and operators
                belong in a plugin before they belong in core
              </td>
            </tr>
            <tr>
              <td>
                <strong>Improve the docs</strong>
              </td>
              <td>
                Edit <code>docs/</code> for the repository or <code>website/</code> for this site
              </td>
            </tr>
          </tbody>
        </table>
      </div>

      <h2>Step 1 — Open or claim an issue</h2>
      <p>
        Discussion happens on{" "}
        <a href={GITHUB_ISSUES_URL} target="_blank" rel="noopener noreferrer">
          GitHub Issues
        </a>
        . Use the templates so triage is quick, and comment on an existing issue to claim it
        rather than opening a duplicate. Wait for a maintainer to confirm scope on anything that
        touches core, the C ABI, or backend selection. Small fixes with an obvious correct answer
        do not need that round trip.
      </p>

      <h2>Step 2 — Fork and build</h2>
      <p>
        Fork the{" "}
        <a href={GITHUB_REPO_URL} target="_blank" rel="noopener noreferrer">
          repository
        </a>
        , then branch from <code>main</code> with a descriptive name such as{" "}
        <code>fix/gguf-iq4nl-stride</code>.
      </p>
      <pre>{`git clone https://github.com/<you>/Universal-AI-Inference.git
cd Universal-AI-Inference
git checkout -b fix/short-description

cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \\
  -DUAII_BUILD_TESTS=ON -DUAII_BUILD_PLUGINS=ON
cmake --build build --config Release --parallel`}</pre>
      <p>
        Full prerequisites are on the <Link href="/docs/getting-started/">Quick start</Link>{" "}
        page. On Windows, App Control frequently blocks unsigned executables; build under WSL
        with <code>bash scripts/build_uaii_wsl.sh</code> if that happens.
      </p>

      <h2>Step 3 — Verify before you push</h2>
      <pre>{`uaii doctor --load-plugins
ctest --test-dir build -C Release --output-on-failure
clang-format -i <files you touched>`}</pre>
      <p>
        Extend the tests that cover the behaviour you changed. Performance work should include{" "}
        <code>uaii_bench</code> output for the affected suite so the claim is reviewable. See{" "}
        <Link href="/docs/benchmarks/">Benchmarks</Link> for the reproduction commands.
      </p>

      <h2>Step 4 — Open the pull request</h2>
      <p>
        Push your branch and open a{" "}
        <a href={GITHUB_PULLS_URL} target="_blank" rel="noopener noreferrer">
          pull request
        </a>{" "}
        against <code>main</code>. A reviewable PR does five things:
      </p>
      <ol className="steps">
        <li>Links the issue it resolves</li>
        <li>Explains why the change is needed, not only what it does</li>
        <li>Stays focused on one concern, so unrelated cleanup ships separately</li>
        <li>Updates the docs describing any behaviour or public API you changed</li>
        <li>Passes CI, including the CodeQL and wheel workflows</li>
      </ol>
      <p>
        Expect review comments on ABI compatibility and honesty of claims. Both are enforced
        deliberately.
      </p>

      <h2>Design rules</h2>
      <p>
        These constraints keep the runtime modular, and reviews will hold you to them:
      </p>
      <ul>
        <li>Core never imports model-specific code</li>
        <li>External formats convert to UAII IR before execution</li>
        <li>Backends understand UAII IR and planned kernels, nothing else</li>
        <li>Prefer a new plugin over a change to core</li>
        <li>
          Return <code>uaii::Error</code> rather than throwing on hot paths; log through{" "}
          <code>uaii::log::*</code> with a short component name
        </li>
        <li>
          Plugin entry points stay <code>extern &quot;C&quot;</code> and versioned
        </li>
        <li>
          Documentation states what is measured and what is a scaffold. Host-fallback GPU
          backends and demo loaders are described as such until real device paths land
        </li>
      </ul>

      <h2>Where code lives</h2>
      <div className="table-wrap">
        <table className="data">
          <thead>
            <tr>
              <th>Path</th>
              <th>Role</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td>
                <code>include/uaii/</code>
              </td>
              <td>Public headers and interfaces</td>
            </tr>
            <tr>
              <td>
                <code>libs/uaii-core/</code>
              </td>
              <td>Errors, logging, configuration, plugin host</td>
            </tr>
            <tr>
              <td>
                <code>libs/uaii-*/</code>
              </td>
              <td>Modular libraries: ir, runtime, kernels, backends, loaders, capi</td>
            </tr>
            <tr>
              <td>
                <code>bindings/python/</code>
              </td>
              <td>Python SDK</td>
            </tr>
            <tr>
              <td>
                <code>plugins/</code>
              </td>
              <td>Reference plugins for discovery and operator registration</td>
            </tr>
            <tr>
              <td>
                <code>website/</code>
              </td>
              <td>This documentation site, built with Next.js static export</td>
            </tr>
            <tr>
              <td>
                <code>docs/</code>
              </td>
              <td>Architecture, features, roadmap, and stability policy</td>
            </tr>
            <tr>
              <td>
                <code>tests/</code>
              </td>
              <td>Smoke and runtime integration tests</td>
            </tr>
          </tbody>
        </table>
      </div>

      <h2>Editing this site</h2>
      <pre>{`cd website
npm install
npm run dev    # http://localhost:3000
npm run build  # static export`}</pre>
      <p>
        Pages are React components under <code>website/app/</code>. Add a documentation page by
        creating <code>app/docs/&lt;slug&gt;/page.tsx</code> and registering it in the sidebar
        groups in <code>components/DocLayout.tsx</code>.
      </p>

      <h2>Reference</h2>
      <ul>
        <li>
          <a href={GITHUB_CONTRIBUTING_URL} target="_blank" rel="noopener noreferrer">
            CONTRIBUTING.md
          </a>{" "}
          — the authoritative version of this process
        </li>
        <li>
          <a href={GITHUB_CODE_OF_CONDUCT_URL} target="_blank" rel="noopener noreferrer">
            Code of Conduct
          </a>{" "}
          — expected behaviour in issues and reviews
        </li>
        <li>
          <Link href="/docs/architecture/">Architecture</Link> — module boundaries before you
          change them
        </li>
        <li>
          <Link href="/docs/plugins/">Plugins</Link> — the extension path for formats, operators,
          and backends
        </li>
      </ul>
    </DocLayout>
  );
}
