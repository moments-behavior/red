# Streamer Example
Example of streaming video frames from CUDA computation to OpenGL and rendering with Dear ImGui.

This is based on Dear ImGui examples using GLFW, and just adds the minimum for rendering the result of a CUDA computation with PBO.

# Building

To build on Windows, make sure you have initialized your environment with `vcvarsall.bat x86` and then run the build batch script:
```
C:\path\to\streamer-example>build_release_win32.bat
```

# Usage
Once you've built the program, you should be able to run `streamer_example.exe` and see the following window:

![image](https://user-images.githubusercontent.com/14188457/161409656-3fc40477-97f4-4de3-b1ca-2b9fe672381b.png)

You should see that the gradient pattern is animated. The coin flipping window is there only to show an example of tracking state independently of CUDA operations being called.
