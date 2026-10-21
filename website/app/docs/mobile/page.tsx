import { DocLayout } from "@/components/DocLayout";
import Link from "next/link";

export const metadata = { title: "Mobile" };

export default function Page() {
  return (
    <DocLayout
      title="Mobile"
      active="/docs/mobile/"
      lede="Android and iOS use the phone preset through the C API. Success is peak RSS and the process staying up."
    >
      <h2>Preset</h2>
      <p>
        About 2 GiB, one staging buffer, 4 threads, KV window 512 plus 4 sink tokens. The budget
        is then half of what the OS can still give this process. Details and the laptop rule are
        on <Link href="/docs/memory/">Memory</Link>.
      </p>
      <div className="table-wrap">
        <table className="data">
          <thead>
            <tr>
              <th>OS</th>
              <th>Free RAM</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td>Windows</td>
              <td>
                <code>GlobalMemoryStatusEx</code>
              </td>
            </tr>
            <tr>
              <td>Linux / Android</td>
              <td>
                <code>/proc/meminfo</code> MemAvailable
              </td>
            </tr>
            <tr>
              <td>iOS</td>
              <td>
                <code>os_proc_available_memory()</code>
              </td>
            </tr>
            <tr>
              <td>macOS</td>
              <td>free + inactive pages</td>
            </tr>
          </tbody>
        </table>
      </div>

      <h2>GPU</h2>
      <p>
        iOS Metal runs MatMul. Android Vulkan runs mapped f32 MatMul, including{" "}
        <code>transpose_b</code>. Attention and RoPE stay on CPU. Q4_K stays in the CPU quant
        GEMM, which uses NEON on arm64. The sample selects the backend with <code>auto</code>.
      </p>

      <h2>C API</h2>
      <pre>{`uaii_memory_preset_info info;
memset(&info, 0, sizeof info);
info.struct_size = sizeof info;
uaii_resolve_memory_preset("phone", 0, &info);

uaii_session_options opts;
uaii_session_options_init(&opts);
uaii_session_options_apply_preset(&opts, &info);
opts.backend = "auto";`}</pre>
      <p>
        <code>uaii_session_set_background(session, 1)</code> makes generate and run return{" "}
        <code>paused: app backgrounded</code>. <code>uaii_session_request_cancel</code> returns{" "}
        <code>cancelled</code>. Generate clears cancel when it starts.{" "}
        <code>uaii_session_memory_warning</code> drops KV and keeps the longrope scale length.
        Callers built against 0.3.0 keep working: new fields are appended.
      </p>

      <h2>Android</h2>
      <pre>{`cmake -S examples/mobile -B build-android -G Ninja \\
  -DCMAKE_TOOLCHAIN_FILE="$ANDROID_NDK/build/cmake/android.toolchain.cmake" \\
  -DANDROID_ABI=arm64-v8a \\
  -DANDROID_PLATFORM=android-24 \\
  -DANDROID_STL=c++_static \\
  -DCMAKE_BUILD_TYPE=Release
cmake --build build-android --target uaii_phone`}</pre>
      <p>
        Ship <code>libuaii_capi.so</code> next to the binary. <code>uaii_phone</code> with no
        arguments prints the resolved budget. <code>uaii_phone model.q4_k.gguf</code> generates a
        few tokens from ids 1, 2, 3.
      </p>

      <h2>iOS</h2>
      <pre>{`cmake -S examples/mobile -B build-ios -G Xcode \\
  -DCMAKE_SYSTEM_NAME=iOS \\
  -DCMAKE_OSX_ARCHITECTURES=arm64 \\
  -DCMAKE_OSX_DEPLOYMENT_TARGET=15.0 \\
  -DCMAKE_BUILD_TYPE=Release`}</pre>
      <p>
        Embed <code>libuaii_capi.dylib</code>. On device, <code>uaii doctor</code> prints{" "}
        <code>platform: ios</code> or <code>platform: android</code>.
      </p>
    </DocLayout>
  );
}
