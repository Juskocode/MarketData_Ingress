# Troubleshooting playbook

## Configuration cannot find TensorRT

Configure production builds with `-DMD_REQUIRE_TENSORRT=ON`. The build needs CUDA Toolkit discovery plus TensorRT 10+ headers and the `nvinfer` and `nvonnxparser` libraries. If installed outside standard paths, export `TENSORRT_ROOT` with `include/` and `lib/` or `lib64/` beneath it.

Do not accept the warning-only mock build as production evidence. Confirm that CMake prints `TensorRT <major>.x detected`.

## Exit code 2: backend or model load failure

Common causes are a missing model, a serialized engine built for a different GPU/TensorRT stack, fewer or more than one input/output tensor, a non-FP32 boundary tensor, or a dynamic shape. Rebuild a fixed-shape engine on the deployment GPU and rerun with `--backend tensorrt`.

TensorRT engines are hardware and software specific. Store the source ONNX and build recipe; do not assume a plan file is portable.

## Exit code 3: latency SLO missed

Inspect `slo.scope` first. For `device_compute`, profile kernels and fusion. For `end_to_end`, also inspect transfers, host copies, enqueue overhead, and synchronization.

Capture a baseline with `trtexec`, then use Nsight Systems when the application and `trtexec` disagree. Lock or record GPU clocks, power mode, temperature, and concurrent GPU consumers. Increase warmup before drawing conclusions.

The JSON `environment` object captures the runtime-visible TensorRT version, CUDA runtime and driver versions, GPU name, compute capability, and total device memory. If any field is missing on a TensorRT run, treat the artifact as incomplete and inspect CUDA initialization.

## Exit code 4: device metric unavailable

`--target-scope device` was requested on a backend without CUDA-event timing, normally the mock backend. This is an intentional policy failure. Use TensorRT or gate `e2e` for CPU/mock diagnostics.

## Exit code 5: output validation failed

`--verify-identity` found at least one output element outside `--tolerance`, or the output shape differed from the input. Use this gate only with the native identity engine. A failure can indicate incorrect bindings, an engine/runtime incompatibility, memory corruption, or an invalid assumption about the loaded model.

## Exit code 6: CUDA graph unavailable

Graph capture was requested on a non-GPU backend or TensorRT/CUDA rejected the capture. Loops, conditionals, data-dependent shapes, synchronous plugins, and changing context state can prevent capture. Retry without `--cuda-graph`, then compare against `trtexec --useCudaGraph` and inspect the capture error before deciding whether ordinary `enqueueV3` is acceptable.

## Input element mismatch

The runtime requires `input_size * batch` to exactly equal the serialized engine's fixed input volume. Inspect the engine with `trtexec --loadEngine=<path> --dumpLayerInfo` and pass matching CLI values.

Dynamic dimensions are rejected. For edge latency, create one optimized engine per supported shape rather than changing optimization profiles on the hot path.

## Device p99 is low but end-to-end p99 is high

The difference is host staging, H2D/D2H copies, enqueue cost, synchronization, and result materialization. Keep data resident on the GPU where the surrounding pipeline allows it, use CUDA graphs for enqueue-bound micro-models, and avoid synchronizing after every inference when request semantics permit pipelining.

## Tail latency is unstable

Check thermal throttling, power-state transitions, GPU contention, CPU scheduling, NUMA placement, pageable-memory regressions, and debug/profiling instrumentation. This runner uses pinned buffers and a dedicated stream; operating-system and hardware noise still remain.

## Correctness before speed

This scaffold tests transport and timing behavior, not model accuracy. Before deployment, compare TensorRT outputs against an authoritative validation set and define numerical tolerances for the chosen precision. A fast engine with unvalidated outputs is not deployable.
