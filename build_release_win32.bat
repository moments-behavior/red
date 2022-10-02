@REM Build for Visual Studio compiler. Run your copy of vcvars32.bat or vcvarsall.bat to setup command-line compiler.
@set OUT_DIR=release
@set OUT_EXE=streamer_example
@set INCLUDES=-I glfw\include
@set SOURCES=main.cpp imgui_impl_glfw.cpp imgui_impl_opengl3.cpp imgui.cpp imgui_draw.cpp imgui_widgets.cpp imgui_demo.cpp imgui_tables.cpp glew.c create_image_cuda.cu
@set LIBS=glfw\lib-vc2010-32 glfw3.lib opengl32.lib gdi32.lib shell32.lib
mkdir %OUT_DIR%
cp Roboto-Regular.ttf "%OUT_DIR%/Roboto-Regular.ttf"
nvcc --machine 32 --optimize 2 --compiler-options /MD %INCLUDES% %SOURCES% --output-directory %OUT_DIR%/ --output-file %OUT_DIR%/%OUT_EXE%.exe --library-path %LIBS% --use-local-env
