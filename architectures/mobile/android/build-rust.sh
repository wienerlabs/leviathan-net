#!/usr/bin/env bash
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
workspace="$here/.."
manifest="$workspace/Cargo.toml"
sdk="${ANDROID_HOME:-${ANDROID_SDK_ROOT:-$HOME/Library/Android/sdk}}"
ndk="${ANDROID_NDK_HOME:-$(find "$sdk/ndk" -mindepth 1 -maxdepth 1 -type d | sort -V | tail -n 1)}"
host="$(uname -s | tr '[:upper:]' '[:lower:]')-x86_64"
toolchain="$ndk/toolchains/llvm/prebuilt/$host/bin"
api=29
target=aarch64-linux-android
lib=libleviathan_multitude_mobile

export CC_aarch64_linux_android="$toolchain/aarch64-linux-android$api-clang"
export AR_aarch64_linux_android="$toolchain/llvm-ar"
export CARGO_TARGET_AARCH64_LINUX_ANDROID_LINKER="$CC_aarch64_linux_android"
export CARGO_TARGET_AARCH64_LINUX_ANDROID_RUSTFLAGS="-C link-arg=-Wl,-z,max-page-size=16384"
export CARGO_PROFILE_RELEASE_STRIP=true

cargo build --manifest-path "$manifest" -p leviathan-multitude-mobile --lib --release --target "$target"
jni="$here/app/src/main/jniLibs/arm64-v8a"
mkdir -p "$jni"
cp "$workspace/target/$target/release/$lib.so" "$jni/"

cargo build --manifest-path "$manifest" -p leviathan-multitude-mobile --lib
host_lib="$(find "$workspace/target/debug" -maxdepth 1 \( -name "$lib.dylib" -o -name "$lib.so" \) | head -n 1)"
cargo run --manifest-path "$manifest" -p leviathan-multitude-mobile --features bindgen --bin uniffi-bindgen -- \
    generate --library "$host_lib" --language kotlin --out-dir "$here/app/src/main/kotlin"
