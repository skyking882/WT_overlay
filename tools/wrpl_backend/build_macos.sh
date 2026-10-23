#!/usr/bin/env bash
# Local macOS build of the wrpl_export backend (docs/wrpl_import_spec.md §2):
#   tools/wrpl_backend/build_macos.sh [x86_64|arm64] [FORK_SRC]
# FORK_SRC is a WrplReplayParser checkout of branch wt-overlay (default third_party/WrplReplayParser, else the sibling
# ../WrplReplayParser). Needs Xcode (clang and its universal python3 3.9) and cmake (Homebrew). x86_64 runs under
# Rosetta on Apple Silicon; arm64 uses Dagor's NEON vecmath. Build folders: ${WT_WRPL_BUILD:-~/.cache/wt_wrpl_build};
# result: <build folder>/dist-macos-<x64|arm64>/wrpl_export-macos-<x64|arm64>.zip and the unpacked wrpl_export/.
# Use it with:  WT_WRPL_BACKEND=<build folder>/dist-macos-x64/wrpl_export python3 -m wt_overlay.wrpl REPLAY.wrpl
set -euo pipefail
arch=${1:-x86_64}
repo=$(cd -- "$(dirname -- "$0")/../.." && pwd)
src=${2:-}
if [ -z "$src" ]; then
  if [ -f "$repo/third_party/WrplReplayParser/CMakeLists.txt" ]; then src=$repo/third_party/WrplReplayParser
  else src=$(cd -- "$repo/.." && pwd)/WrplReplayParser; fi
fi
[ -f "$src/CMakeLists.txt" ] || { echo "WrplReplayParser source not found: $src" >&2; exit 1; }
if [ -n "$(git -C "$src" status --porcelain)" ] && [ "${ALLOW_DIRTY:-0}" != 1 ]; then
  echo "$src has uncommitted changes; the bundle records the fork commit (ALLOW_DIRTY=1 to build anyway)" >&2; exit 1
fi
base=${WT_WRPL_BUILD:-$HOME/.cache/wt_wrpl_build}
py=${PYTHON3:-/Applications/Xcode.app/Contents/Developer/usr/bin/python3}
cmake=${CMAKE:-$(command -v cmake || echo /opt/homebrew/bin/cmake)}
case $arch in
  x86_64) platform=macos-x64
          flags=("-DCMAKE_SYSTEM_PROCESSOR=x86_64" "-DCMAKE_C_FLAGS=-msse4.2 -mpopcnt"
                 "-DCMAKE_CXX_FLAGS=-msse4.2 -mpopcnt -include smmintrin.h") ;;
  arm64)  platform=macos-arm64; flags=() ;;
  *) echo "usage: $0 [x86_64|arm64] [FORK_SRC]" >&2; exit 2 ;;
esac
build=$base/build-$arch
mkdir -p "$base"
"$cmake" -S "$src" -B "$build" -DCMAKE_OSX_ARCHITECTURES="$arch" -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_POSITION_INDEPENDENT_CODE=ON -DENABLE_LTO=OFF -DPARSER_ENABLE_PYTHON_BINDINGS=ON \
  -DPARSER_ENABLE_INSPECTOR=OFF -DMI_BUILD_TESTS=OFF -DMI_BUILD_SHARED=OFF \
  -DPYTHON_VERSION_TO_USE=3.9 -DPython_EXECUTABLE="$py" ${flags[@]+"${flags[@]}"}
"$cmake" --build "$build" --target PyReplayParser signal_tracer -j"$(sysctl -n hw.ncpu)"
cp "$build/3rdParty/signal_tracer/signal_tracer" "$build/python/main/"
venv=$base/venv-$arch
[ -x "$venv/bin/python" ] || arch "-$arch" "$py" -m venv "$venv"
arch "-$arch" "$venv/bin/python" -m pip install -q "pyinstaller==6.22.3"
arch "-$arch" "$venv/bin/python" -B "$repo/tools/wrpl_backend/build.py" --module-dir "$build/python/main" \
  --platform "$platform" --fork-src "$src" --dist "$base/dist-$platform" --work "$base/work-$platform" --zip
