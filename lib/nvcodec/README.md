# NVIDIA Video Codec SDK interface files

Used only by CUDA builds on Windows (`RED_ENABLE_CUDA=ON`), for NVDEC hardware
video decoding. Red's downloadable releases are built with CUDA off and contain
none of this.

- `nvcuvid.h`, `cuviddec.h`, `nvEncodeAPI.h`: NVIDIA's MIT licence, stated at
  the top of each file ("This copyright notice applies to this header file
  only").
- `x64/` and `Win32/` `nvcuvid.lib`, `nvencodeapi.lib`: import libraries from
  the [NVIDIA Video Codec SDK](https://developer.nvidia.com/video-codec-sdk),
  covered by the SDK's licence agreement, which comes with the SDK download.
  The decoder itself (`nvcuvid.dll`) is part of the NVIDIA driver and is never
  shipped with Red.
