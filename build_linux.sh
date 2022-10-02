#!/bin/bash
mkdir -p release;
rm -f release/streamer_example;
cp src/Roboto-Regular.ttf release/Roboto-Regular.ttf
cp src/shader_picture.fs release/shader_picture.fs
cp src/shader_picture.vs release/shader_picture.vs

nvcc -c src/create_image_cuda.cu -arch=sm_80 -o release/create_image_cuda.o
nvcc -c src/ColorSpace.cu -arch=sm_80 -o release/ColorSpace.o


DIR_IMGUI="lib/imgui"
g++ -std=c++11 -I$DIR_IMGUI -g -Wall -Wformat `pkg-config --cflags glfw3` -c -o release/imgui.o $DIR_IMGUI/imgui.cpp
g++ -std=c++11 -I$DIR_IMGUI -g -Wall -Wformat `pkg-config --cflags glfw3` -c -o release/imgui_demo.o $DIR_IMGUI/imgui_demo.cpp
g++ -std=c++11 -I$DIR_IMGUI -g -Wall -Wformat `pkg-config --cflags glfw3` -c -o release/imgui_draw.o $DIR_IMGUI/imgui_draw.cpp
g++ -std=c++11 -I$DIR_IMGUI -g -Wall -Wformat `pkg-config --cflags glfw3` -c -o release/imgui_tables.o $DIR_IMGUI/imgui_tables.cpp
g++ -std=c++11 -I$DIR_IMGUI -g -Wall -Wformat `pkg-config --cflags glfw3` -c -o release/imgui_widgets.o $DIR_IMGUI/imgui_widgets.cpp
g++ -std=c++11 -I$DIR_IMGUI -g -Wall -Wformat `pkg-config --cflags glfw3` -c -o release/imgui_impl_glfw.o $DIR_IMGUI/backends/imgui_impl_glfw.cpp
g++ -std=c++11 -I$DIR_IMGUI -g -Wall -Wformat `pkg-config --cflags glfw3` -c -o release/imgui_impl_opengl3.o $DIR_IMGUI/backends/imgui_impl_opengl3.cpp


g++ -Ofast -ffast-math -std=c++11 \
    release/ColorSpace.o \
    -o release/*.o \
    -Ilib/nvcodec \
    -o release/streamer_example -I ./src/ src/*.cpp \
    -I/usr/local/cuda-11.4/include \
    -I$DIR_IMGUI \
    -I$DIR_IMGUI/backends \
    -L/usr/local/cuda-11.4/lib64/ -lcudart -lcuda -lnppicc -lnvcuvid \
    -lGLEW -lGLU -lGL \
    `pkg-config --static --libs glfw3` \
    `pkg-config --cflags libavformat libswscale libswresample libavutil libavcodec` \
    `pkg-config --libs libavformat libswscale libswresample libavutil libavcodec`
    