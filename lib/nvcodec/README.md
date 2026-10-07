# NVDEC interface files

For CUDA builds (`RED_ENABLE_CUDA=ON`), NVDEC hardware video decoding. Red's
downloadable releases are built with CUDA off and use none of this.

- `nvcuvid.h`, `cuviddec.h`, `nvEncodeAPI.h`: from the
  [NVIDIA Video Codec SDK](https://developer.nvidia.com/video-codec-sdk), under
  NVIDIA's MIT licence, stated at the top of each file ("This copyright notice
  applies to this header file only").
- `nvcuvid.def`: Red's own list of the `nvcuvid.dll` functions it links
  against. On Windows the build turns it into an import library with
  `lib.exe /def` (see CMakeLists.txt), instead of using the SDK's
  `nvcuvid.lib`, whose licence does not allow passing it on. Linux links the
  driver's `libnvcuvid.so` directly.

The decoder itself (`nvcuvid.dll` / `libnvcuvid.so`) is part of the NVIDIA
driver and is never shipped with Red.
