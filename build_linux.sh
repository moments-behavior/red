#!/bin/bash
mkdir -p release;
rm -f release/redgui;
#cp src/Roboto-Regular.ttf Roboto-Regular.ttf
#cp src/fa-solid-900.ttf fa-solid-900.ttf

nvcc -c src/create_image_cuda.cu -arch=sm_80 -o release/create_image_cuda.o
nvcc -c src/ColorSpace.cu -arch=sm_80 -o release/ColorSpace.o
nvcc -c src/kernel.cu -arch=sm_80 -o release/kernel.o


DIR_IMGUI="lib/imgui"
DIR_IMPLOT="lib/implot"

g++ -std=c++11 -I$DIR_IMGUI -g -Wall -Wformat `pkg-config --cflags glfw3` -c -o release/imgui.o $DIR_IMGUI/imgui.cpp
g++ -std=c++11 -I$DIR_IMGUI -g -Wall -Wformat `pkg-config --cflags glfw3` -c -o release/imgui_demo.o $DIR_IMGUI/imgui_demo.cpp
g++ -std=c++11 -I$DIR_IMGUI -g -Wall -Wformat `pkg-config --cflags glfw3` -c -o release/imgui_draw.o $DIR_IMGUI/imgui_draw.cpp
g++ -std=c++11 -I$DIR_IMGUI -g -Wall -Wformat `pkg-config --cflags glfw3` -c -o release/imgui_tables.o $DIR_IMGUI/imgui_tables.cpp
g++ -std=c++11 -I$DIR_IMGUI -g -Wall -Wformat `pkg-config --cflags glfw3` -c -o release/imgui_widgets.o $DIR_IMGUI/imgui_widgets.cpp
g++ -std=c++11 -I$DIR_IMGUI -g -Wall -Wformat `pkg-config --cflags glfw3` -c -o release/imgui_impl_glfw.o $DIR_IMGUI/backends/imgui_impl_glfw.cpp
g++ -std=c++11 -I$DIR_IMGUI -g -Wall -Wformat `pkg-config --cflags glfw3` -c -o release/imgui_impl_opengl3.o $DIR_IMGUI/backends/imgui_impl_opengl3.cpp

g++ -std=c++17 -I$DIR_IMPLOT -I$DIR_IMGUI -g -Wall -c -o release/implot.o $DIR_IMPLOT/implot.cpp
g++ -std=c++17 -I$DIR_IMPLOT -I$DIR_IMGUI -g -Wall -c -o release/implot_items.o $DIR_IMPLOT/implot_items.cpp
g++ -std=c++17 -I$DIR_IMPLOT -I$DIR_IMGUI -g -Wall -c -o release/implot_demo.o $DIR_IMPLOT/implot_demo.cpp

g++ -Ofast -mssse3 -ffast-math -std=c++17 \
    release/ColorSpace.o \
    -o release/*.o \
    -Ilib/nvcodec \
    -o release/redgui -I ./src/ src/*.cpp \
    -I/usr/local/cuda/include \
    -I$DIR_IMPLOT \
    -I$DIR_IMGUI \
    -I$DIR_IMGUI/backends \
    -Ilib/IconFontCppHeaders \
    -Ilib/imgui-filebrowser \
    -L/usr/local/cuda/lib64/ -lcudart -lcuda -lnvcuvid -lnppicc -lnppidei -lnvidia-encode -lnppc -lnppig -lnppial \
    -lGLEW -lGLU -lGL \
    -lpthread \
    `pkg-config --static --libs glfw3` \
    -I$HOME/nvidia/ffmpeg/build/include/ \
    -L$HOME/nvidia/ffmpeg/build/lib/ -lavformat -lswscale -lswresample -lavutil -lavcodec \
    -I/usr/local/include/opencv4 \
    -L/usr/local/lib \
    -lopencv_sfm -lopencv_core -lopencv_bgsegm -lopencv_imgcodecs -lopencv_imgproc -lopencv_video -lopencv_highgui -lopencv_videoio -lopencv_calib3d -lopencv_dnn \
    -I$HOME/build/TensorRT-8.6.1.6/include \
    -L$HOME/build/TensorRT-8.6.1.6/lib/ -lnvinfer -lnvinfer_plugin

./release/redgui