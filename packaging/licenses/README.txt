Red: licences
=============

Red's own source code is under the MIT License (LICENSE.txt in this folder).
Source: https://github.com/moments-behavior/red -- the tag named after this
version (for example v2.0.0) is the exact code this package was built from.

This package also contains other people's software, each under its own
licence, all in this folder:

  vendored/        code compiled into red: Dear ImGui, ImPlot,
                   ImGuiFileDialog (with dirent and stb), cpp-httplib, miniz,
                   IconFontCppHeaders, JSON for Modern C++, stb_image_write,
                   NVIDIA's FFmpegDemuxer
  fonts.txt        the fonts in fonts/: Roboto, Font Awesome, Fork Awesome
  third_party/     the shared libraries shipped beside red (FFmpeg, Apache
                   Arrow, Ceres, ...), one folder each with that library's
                   licence files
  third_party/SOURCES.txt
                   each of those libraries' version and where its source
                   code is
  GPL-3.0.txt, Apache-2.0.txt, OFL-1.1.txt
                   full texts of licences referred to above

The GNU GPL and the macOS and Linux packages
--------------------------------------------
The FFmpeg in the macOS and Linux packages is built with GPL components (such
as the x264 and x265 encoders), and parts of SuiteSparse (used by Ceres) are
under the GPL too. A program combined with GPL libraries may only be passed on
under the GPL, so these packages, as a whole, are distributed under the GNU
General Public License, version 3 (GPL-3.0.txt). You may use, copy, change
and pass them on under its terms.

This does not change the licence of Red's own source code, which stays MIT:
whoever builds Red from source with other libraries gets it under MIT.

The source code for everything in such a package is: Red's, at the version's
tag on GitHub (above); each library's, at the address in
third_party/SOURCES.txt. If one of those addresses no longer works, open an
issue at https://github.com/moments-behavior/red/issues and we will provide
the source.

The Windows package is built with an LGPL build of FFmpeg (no GPL
components). Each of its libraries' licences is in third_party/.
