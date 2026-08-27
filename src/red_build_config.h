#pragma once
// Single source of truth for whether this build contains any CUDA/NVDEC code.
//
// macOS never does (VideoToolbox + Metal). Every other platform does by
// default; configuring with -DRED_ENABLE_CUDA=OFF defines RED_NO_CUDA, which
// drops the .cu sources and the libcuda/libnvcuvid link so the binary loads on
// a machine that has no NVIDIA driver at all -- those links are DT_NEEDED, so
// without this the loader fails before main() and no runtime probe ever runs.
//
// RED_HAVE_CUDA is about what is COMPILED IN. Whether the CUDA path is
// actually USED at runtime is a separate question answered by
// red::decode_backend() in decode_backend.h.
#if !defined(__APPLE__) && !defined(RED_NO_CUDA)
#define RED_HAVE_CUDA 1
#endif
