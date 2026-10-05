# Ubuntu 22.04 build environment for red's Linux release.
#
# A Linux binary runs on the glibc it was built against and newer, so the
# release is built on the oldest Ubuntu it should support -- here, inside this
# image, whatever the host runs. Used by build_release.sh; not run directly.
FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential cmake pkg-config git patchelf file \
        ca-certificates lsb-release wget gnupg \
        libeigen3-dev libceres-dev libglfw3-dev libglew-dev libopenblas-dev \
        libavcodec-dev libavformat-dev libavutil-dev libswscale-dev \
        libgtest-dev libgmock-dev \
    && rm -rf /var/lib/apt/lists/*

# Apache Arrow + Parquet (tailcycle datasets) come from Apache's own apt
# repository; Ubuntu does not package them. apt-get update first: the layer
# above cleared the package lists, and the repository package needs gnupg
# resolved from them.
RUN apt-get update \
    && wget -q https://apache.jfrog.io/artifactory/arrow/ubuntu/apache-arrow-apt-source-latest-jammy.deb \
    && apt-get install -y ./apache-arrow-apt-source-latest-jammy.deb \
    && rm apache-arrow-apt-source-latest-jammy.deb \
    && apt-get update && apt-get install -y --no-install-recommends \
        libarrow-dev libparquet-dev \
    && rm -rf /var/lib/apt/lists/*

# appimagetool and the static type-2 runtime, for make_appimage.sh. The static
# runtime carries FUSE itself, so the AppImage needs no libfuse2 on the
# user's machine. (Continuous builds: the project publishes no other.)
RUN mkdir -p /opt/appimage \
    && wget -q -O /opt/appimage/appimagetool \
        https://github.com/AppImage/appimagetool/releases/download/continuous/appimagetool-x86_64.AppImage \
    && wget -q -O /opt/appimage/runtime-x86_64 \
        https://github.com/AppImage/type2-runtime/releases/download/continuous/runtime-x86_64 \
    && chmod +x /opt/appimage/appimagetool
