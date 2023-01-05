del "C:/Users/yaoyao/src/red-refactor/release/imgui.init"
del "C:/Users/yaoyao/src/red-refactor/release/regui.exe"

nvcc -c src/create_image_cuda.cu -arch=sm_80 -o release/create_image_cuda.o
nvcc -c src/ColorSpace.cu -arch=sm_80 -o release/ColorSpace.o
@set DIR_IMGUI=lib/imgui
@set DIR_IMPLOT=lib/implot

clang++.exe -Wno-everything -g -std=c++17 src/main.cpp src/decoder.cpp src/FFmpegDemuxer.cpp src/NvDecoder.cpp ^
%DIR_IMGUI%/imgui.cpp %DIR_IMGUI%/imgui_demo.cpp %DIR_IMGUI%/imgui_draw.cpp %DIR_IMGUI%/imgui_tables.cpp %DIR_IMGUI%/imgui_widgets.cpp %DIR_IMGUI%/backends/imgui_impl_glfw.cpp %DIR_IMGUI%/backends/imgui_impl_opengl3.cpp ^
%DIR_IMPLOT%/implot.cpp %DIR_IMPLOT%/implot_items.cpp %DIR_IMPLOT%/implot_demo.cpp ^
-I lib -I lib/nvcodec -I lib/IconFontCppHeaders ^
-I lib/imgui-filebrowser -I lib/FFmpeg/include -I %DIR_IMGUI%/backends -I %DIR_IMGUI% -I %DIR_IMPLOT% -I "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.6/include" -I lib/GLFW ^
release/create_image_cuda.o release/ColorSpace.o -o release/redgui.exe ^
-L "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.6/lib/x64" -lcudart -lcuda ^
-L lib/nvcodec/x64 -lnvencodeapi -lnvcuvid ^
-L lib/FFmpeg/lib/x64 -lavcodec -lavformat -lavutil -lswresample ^
-L lib/GL/lib/x64 -lglew32 ^
-L lib/GLFW/lib-vc2019 -lglfw3 ^
-lopengl32 -lgdi32 -luser32 -lshell32 -lmsvcrt -lvcruntime -lucrt
