# Phone builds

The phone preset is the same paging dial as laptop, stopped lower: about **2 GiB**, one staging buffer, **4** threads, and a KV window of **512** tokens plus **4** sink tokens. `resolve_memory_preset` then clamps that budget to half of the memory the OS can still give this process.

| OS | Available-RAM source |
|---|---|
| Windows | `GlobalMemoryStatusEx` `ullAvailPhys` |
| Linux / Android | `/proc/meminfo` `MemAvailable` |
| iOS | `os_proc_available_memory()` |
| macOS | `host_statistics64` free + inactive pages |

A typed `--ram-gb 8` on a phone with 6 GiB free becomes 3 GiB. When the cap is under 1 GiB the KV window becomes 128. When it is under 1.5 GiB the window becomes 256. A cap of exactly 1 GiB keeps 256. The expert LRU is also capped at one eighth of that budget. If available memory is unknown, the requested budget is kept. Laptop, desktop, and server use a different cap: the target stands when it fits in free RAM, and becomes the free amount when it does not.

Hugging Face BF16 checkpoints at 7B and above stay fail-closed. Convert them to GGUF Q4_K. Vision, SSM, and Hugging Face MLA stay out of this path.

On iOS, Metal runs MatMul. On Android, Vulkan runs mapped f32 MatMul, including `transpose_b` for a dequantized tile. Attention and RoPE stay on CPU. Q4_K stays in the CPU quant GEMM, which uses NEON dot products on arm64. `uaii doctor` prints the resolved phone budget, whether it was clamped, available bytes, and that split.

Watch **peak RSS** and whether the process stays up. Tokens per second is not the success metric.

## C API (0.4.0)

```c
uaii_memory_preset_info info;
memset(&info, 0, sizeof info);
info.struct_size = sizeof info;
uaii_resolve_memory_preset("phone", 0, &info);

uaii_session_options opts;
uaii_session_options_init(&opts);
uaii_session_options_apply_preset(&opts, &info);
```

`uaii_session_set_background(session, 1)` makes `generate` and `run` return `paused: app backgrounded` until it is cleared. `uaii_session_request_cancel` makes them return `cancelled`. `generate` clears cancel when it starts, so an old cancel does not stick. `uaii_session_memory_warning` drops the KV cache and leaves streamed weights on the plan. `uaii_process_rss_bytes` and `uaii_system_available_bytes` are the counters to log.

Callers built against C API 0.3.0 keep working: new `uaii_session_options` fields are appended, and the library copies only `struct_size` bytes.

## Android (NDK, arm64-v8a)

```bash
cmake -S examples/mobile -B build-android -G Ninja \
  -DCMAKE_TOOLCHAIN_FILE="$ANDROID_NDK/build/cmake/android.toolchain.cmake" \
  -DANDROID_ABI=arm64-v8a \
  -DANDROID_PLATFORM=android-24 \
  -DANDROID_STL=c++_static \
  -DCMAKE_BUILD_TYPE=Release
cmake --build build-android --target uaii_phone
```

Ship `libuaii_capi.so` next to the binary. `uaii_phone` with no arguments prints the resolved budget. `uaii_phone model.q4_k.gguf` loads that file and generates a few tokens from ids `1, 2, 3`.

## iOS (arm64, deployment target 15)

```bash
cmake -S examples/mobile -B build-ios -G Xcode \
  -DCMAKE_SYSTEM_NAME=iOS \
  -DCMAKE_OSX_ARCHITECTURES=arm64 \
  -DCMAKE_OSX_DEPLOYMENT_TARGET=15.0 \
  -DCMAKE_BUILD_TYPE=Release
```

Embed `libuaii_capi.dylib` in the app. On device, `uaii doctor` prints `platform: ios` or `platform: android`.

## Host check

`uaii_memory_preset_test` and `uaii_storage_plan_test` cover the preset, the available-RAM clamp, and the single staging slot. A 7B run on a device is outside CI.

To link the sample on a desktop toolchain:

```bash
cmake -S . -B build -DUAII_BUILD_MOBILE_SAMPLE=ON
cmake --build build --target uaii_phone
```
