#!/usr/bin/env bash
# Build red's Linux release tarball inside Ubuntu 22.04, from any Linux host.
#
#     packaging/linux/build_release.sh
#
# The release has to be built on the oldest Ubuntu it should run on (glibc
# only runs forward), and the build box may be newer -- so this builds in the
# Dockerfile beside it: CUDA off, system FFmpeg, Arrow from Apache's repo. It
# uses its own build folder, release-linux/, leaving release/ (the host's own
# build) alone, and runs as the calling user so dist/ is not owned by root.
# Output: dist/red-<version>-linux-x64.tar.gz (make_tarball.sh) and
# dist/red-<version>-linux-x86_64.AppImage (make_appimage.sh).
#
# Needs Docker (sudo apt install docker.io; add yourself to the docker group,
# or run this with sudo). Submodules must be checked out:
#     git submodule update --init --recursive
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
IMAGE=red-build-ubuntu22.04

command -v docker > /dev/null || { echo "docker not found: sudo apt install docker.io" >&2; exit 1; }
[ -e "$REPO/lib/imgui/imgui.h" ] ||
    { echo "submodules missing: git submodule update --init --recursive" >&2; exit 1; }

echo "== build environment ($IMAGE)"
docker build -t "$IMAGE" -f "$REPO/packaging/linux/Dockerfile" "$REPO/packaging/linux"

echo "== build and package red"
# HOME and safe.directory: the container user has no home of its own, and git
# refuses a repo owned by a uid it does not know.
docker run --rm \
    --user "$(id -u):$(id -g)" \
    -e HOME=/tmp \
    -e GIT_CONFIG_COUNT=1 -e GIT_CONFIG_KEY_0=safe.directory -e GIT_CONFIG_VALUE_0='*' \
    -v "$REPO:/red" -w /red \
    "$IMAGE" \
    bash -c 'cmake -S . -B release-linux -DCMAKE_BUILD_TYPE=Release -DRED_ENABLE_CUDA=OFF &&
             cmake --build release-linux -j"$(nproc)" &&
             packaging/linux/make_tarball.sh release-linux dist &&
             packaging/linux/make_appimage.sh dist'
