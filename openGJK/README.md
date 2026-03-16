
[![Language Bindings](https://github.com/MattiaMontanari/openGJK/actions/workflows/ci-examples.yml/badge.svg)](https://github.com/MattiaMontanari/openGJK/actions/workflows/ci-examples.yml)
[![Tests](https://github.com/MattiaMontanari/openGJK/actions/workflows/ci-tests.yml/badge.svg)](https://github.com/MattiaMontanari/openGJK/actions/workflows/ci-tests.yml)

# OpenGJK

A fast and robust implementation of the Gilbert-Johnson-Keerthi (GJK) algorithm for computing minimum distances between convex polytopes. Available in three flavors:

- **Scalar** (`scalar/`): Portable C implementation with interfaces for C#, Go, Matlab, Python, and Zig
- **SIMD** (`simd/`): High-performance C++ implementation using [Google Highway](https://github.com/google/highway) for automatic SIMD acceleration (SSE4, AVX2, AVX-512, NEON)
- **GPU** (`gpu/`): CUDA implementation with warp-level parallelism for batch collision detection on NVIDIA GPUs

A Unity Plug-in [is also available in another repository](https://github.com/MattiaMontanari/urban-couscous).

Useful links: [API references](https://www.mattiamontanari.com/opengjk/docsapi/), [documentation](https://www.mattiamontanari.com/opengjk/docs/) and [automated benchmarks](https://www.mattiamontanari.com/opengjk/docs/benchmarks/).

## Getting started

On Linux, Mac or Windows, install a basic C/C++ toolchain - for example: git, compiler and cmake.

### Prerequisites

**Required:**

- Git
- C/C++ compiler (GCC, Clang, or MSVC)
- CMake (version 3.5 or higher)

**Recommended for faster builds:**

- Ninja build system (provides ~60% faster compilation)

```bash
# Install Ninja (if not already installed)
# Ubuntu/Debian: sudo apt install ninja-build
# macOS: brew install ninja
# Windows: choco install ninja
```

Next, clone this repo:

``` bash
git clone https://github.com/MattiaMontanari/openGJK
```

Then use these commands to build and run an example:

``` bash
cmake -E make_directory build
cmake -E chdir build cmake -DCMAKE_BUILD_TYPE=Release -G Ninja .. 
cmake --build build 
cmake -E chdir build/scalar/examples/c ./example_lib_opengjk_ce
```

The successful output should be:

>
> `Distance between bodies 3.653650`
>

However, if you do get an error - any error - please file a bug. Support requests are welcome.

## CMake Options

OpenGJK supports several build options to customize compilation. Use them by passing `-D<OPTION>=<VALUE>` to cmake:

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release -DBUILD_SIMD=ON -DUSE_32BITS=OFF
```

### Global Options (root CMakeLists.txt)

| Option | Default | Type | Description |
|--------|---------|------|-------------|
| `BUILD_SCALAR` | ON | BOOL | Build scalar (C) implementation |
| `BUILD_SIMD` | ON | BOOL | Build SIMD (C++ Highway) implementation |
| `BUILD_GPU` | OFF | BOOL | Build GPU (CUDA) implementation |
| `BUILD_TESTS` | ON | BOOL | Build unit tests (cmocka, gtest) |
| `BUILD_EXAMPLES` | ON | BOOL | Build example applications |
| `USE_32BITS` | ON | BOOL | Use 32-bit float instead of 64-bit double |

### Scalar-specific Options (scalar/CMakeLists.txt)

| Option | Default | Type | Description |
|--------|---------|------|-------------|
| `OPENGJK_SCALAR_BUILD_SHARED` | ON | BOOL | Build shared library instead of static-only |
| `OPENGJK_SCALAR_SINGLE_PRECISION` | OFF | BOOL | Use single precision (float) — *overridden by `USE_32BITS`* |

### SIMD-specific Options (simd/CMakeLists.txt)

| Option | Default | Type | Description |
|--------|---------|------|-------------|
| `USE_MINIMAL_SIMD` | OFF | BOOL | Prefer smallest viable SIMD width (128-bit for float, 256-bit for double) instead of widest available |

> **⚠️ Critical Note for Integrators**  
> When you compile your own code against OpenGJK, ensure that **floating-point precision and macros are consistent** between the library and your code:
>
> - Mismatched `USE_32BITS` settings cause link errors or silent ABI incompatibilities
> - Always use `find_package(opengjk)` in your CMake project to automatically inherit the same flags and defines

### SIMD Build (simd/)

The SIMD implementation requires the [Google Highway](https://github.com/google/highway) library. It's fetched automatically via CMake FetchContent:

```bash
cd simd
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
ctest --test-dir build
```

#### Supported SIMD Targets

The GJK algorithm operates on 3D vectors, requiring 4 SIMD lanes (3 coordinates + 1 padding). This constrains which instruction sets work with each precision:

| Target | Width | float (32-bit) | double (64-bit) | Notes |
|--------|-------|----------------|-----------------|-------|
| **SSE4** | 128-bit | ✅ 4 lanes | ❌ 2 lanes | x86/x64 |
| **AVX2** | 256-bit | ✅ 8 lanes | ✅ 4 lanes | x86/x64, minimum for double |
| **AVX-512** | 512-bit | ✅ 16 lanes | ✅ 8 lanes | Modern x86, can disable with `MINIMAL_WIDTH` |
| **NEON** | 128-bit | ✅ 4 lanes | ❌ 2 lanes | ARM64 (Apple Silicon, Raspberry Pi) |
| **SVE/SVE2** | Variable | ❌ | ❌ | **Not supported** - uses sizeless types incompatible with our simplex arrays |

#### CI Test Matrix

All SIMD targets are tested in CI: [Build and Test workflow](https://github.com/MattiaMontanari/openGJK/actions/workflows/ci-tests.yml)

| Platform | Architecture | float | double | SIMD Target |
|----------|--------------|-------|--------|-------------|
| **Linux** (Ubuntu) | x86_64 | ✅ | ✅ | AVX2 |
| **macOS** (Intel) | x86_64 | ✅ | ✅ | SSE4/AVX2 |
| **macOS** (Apple Silicon) | ARM64 | ✅ | ❌ scalar | NEON |
| **Windows** | x86_64 | ✅ | ✅ | AVX2/AVX-512 |

Example with custom options:

```bash
cmake -E chdir build cmake -DCMAKE_BUILD_TYPE=Release -DBUILD_MONO=ON -DFORCE_CXX_COMPILER=ON -G Ninja ..
```

### GPU Build (gpu/)

The GPU implementation uses CUDA for massively parallel collision detection with warp-level parallelism. Includes both GJK (distance computation) and EPA (penetration depth/witness points).

**Prerequisites:**
- NVIDIA GPU with CUDA support (compute capability 6.0+)
- CUDA Toolkit (11.0 or higher)
- CMake 3.18 or higher

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release -DBUILD_GPU=ON -DBUILD_SCALAR=OFF -DBUILD_SIMD=OFF
cmake --build build --config Release
cd build/gpu/examples/simple_collision/Release
./example_lib_opengjk_gpu.exe
```

The successful output should be:

> `Distance between bodies 3.653650`

GPU-specific notes:
- The GPU build inherits global options like `USE_32BITS` for precision control (see [Global Options](#global-options-rootcmakelists) above)

See [gpu/README.md](gpu/README.md) for API details, performance benchmarks, and advanced usage.

Based on [OpenGJK-GPU](https://github.com/vismaychuriwala/OpenGJK-GPU) by Vismay Churiwala and Marcus Hedlund.

## Use OpenGJK in your project

The best source to learn how to use OpenGJK are the examples. They are listed [here](https://www.mattiamontanari.com/opengjk/docs/examples/) for C, C#, Go, Matlab, Zig and Python. I aim to publish a few more for Julia.

Take a look at the `examples` folder in this repo and have fun. File a request if you wish to see more!

## Contribute

You are very welcome to:

- Create pull requests of any kind
- Let me know if you are using this library and find it useful
- Open issues with request for support because they will help you and many others
- Cite this repository ([a sweet GitHub feature](https://docs.github.com/en/repositories/managing-your-repositorys-settings-and-features/customizing-your-repository/about-citation-files#about-citation-files)) or my paper: Montanari, M. et at, *Improving the GJK Algorithm for Faster and More Reliable Distance Queries Between Convex Objects* (2017). ACM Trans. Graph.
