# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.22

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /root/jupyter_notebooks/Fyp/XFeatSystem

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /root/jupyter_notebooks/Fyp/XFeatSystem/build

# Include any dependencies generated for this target.
include CMakeFiles/xfeat_system.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/xfeat_system.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/xfeat_system.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/xfeat_system.dir/flags.make

CMakeFiles/xfeat_system.dir/src/XFextractor.cpp.o: CMakeFiles/xfeat_system.dir/flags.make
CMakeFiles/xfeat_system.dir/src/XFextractor.cpp.o: ../src/XFextractor.cpp
CMakeFiles/xfeat_system.dir/src/XFextractor.cpp.o: CMakeFiles/xfeat_system.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/root/jupyter_notebooks/Fyp/XFeatSystem/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/xfeat_system.dir/src/XFextractor.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/xfeat_system.dir/src/XFextractor.cpp.o -MF CMakeFiles/xfeat_system.dir/src/XFextractor.cpp.o.d -o CMakeFiles/xfeat_system.dir/src/XFextractor.cpp.o -c /root/jupyter_notebooks/Fyp/XFeatSystem/src/XFextractor.cpp

CMakeFiles/xfeat_system.dir/src/XFextractor.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/xfeat_system.dir/src/XFextractor.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /root/jupyter_notebooks/Fyp/XFeatSystem/src/XFextractor.cpp > CMakeFiles/xfeat_system.dir/src/XFextractor.cpp.i

CMakeFiles/xfeat_system.dir/src/XFextractor.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/xfeat_system.dir/src/XFextractor.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /root/jupyter_notebooks/Fyp/XFeatSystem/src/XFextractor.cpp -o CMakeFiles/xfeat_system.dir/src/XFextractor.cpp.s

CMakeFiles/xfeat_system.dir/src/XFeat.cpp.o: CMakeFiles/xfeat_system.dir/flags.make
CMakeFiles/xfeat_system.dir/src/XFeat.cpp.o: ../src/XFeat.cpp
CMakeFiles/xfeat_system.dir/src/XFeat.cpp.o: CMakeFiles/xfeat_system.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/root/jupyter_notebooks/Fyp/XFeatSystem/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object CMakeFiles/xfeat_system.dir/src/XFeat.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/xfeat_system.dir/src/XFeat.cpp.o -MF CMakeFiles/xfeat_system.dir/src/XFeat.cpp.o.d -o CMakeFiles/xfeat_system.dir/src/XFeat.cpp.o -c /root/jupyter_notebooks/Fyp/XFeatSystem/src/XFeat.cpp

CMakeFiles/xfeat_system.dir/src/XFeat.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/xfeat_system.dir/src/XFeat.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /root/jupyter_notebooks/Fyp/XFeatSystem/src/XFeat.cpp > CMakeFiles/xfeat_system.dir/src/XFeat.cpp.i

CMakeFiles/xfeat_system.dir/src/XFeat.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/xfeat_system.dir/src/XFeat.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /root/jupyter_notebooks/Fyp/XFeatSystem/src/XFeat.cpp -o CMakeFiles/xfeat_system.dir/src/XFeat.cpp.s

CMakeFiles/xfeat_system.dir/src/SuperPointExtractor.cpp.o: CMakeFiles/xfeat_system.dir/flags.make
CMakeFiles/xfeat_system.dir/src/SuperPointExtractor.cpp.o: ../src/SuperPointExtractor.cpp
CMakeFiles/xfeat_system.dir/src/SuperPointExtractor.cpp.o: CMakeFiles/xfeat_system.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/root/jupyter_notebooks/Fyp/XFeatSystem/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Building CXX object CMakeFiles/xfeat_system.dir/src/SuperPointExtractor.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/xfeat_system.dir/src/SuperPointExtractor.cpp.o -MF CMakeFiles/xfeat_system.dir/src/SuperPointExtractor.cpp.o.d -o CMakeFiles/xfeat_system.dir/src/SuperPointExtractor.cpp.o -c /root/jupyter_notebooks/Fyp/XFeatSystem/src/SuperPointExtractor.cpp

CMakeFiles/xfeat_system.dir/src/SuperPointExtractor.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/xfeat_system.dir/src/SuperPointExtractor.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /root/jupyter_notebooks/Fyp/XFeatSystem/src/SuperPointExtractor.cpp > CMakeFiles/xfeat_system.dir/src/SuperPointExtractor.cpp.i

CMakeFiles/xfeat_system.dir/src/SuperPointExtractor.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/xfeat_system.dir/src/SuperPointExtractor.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /root/jupyter_notebooks/Fyp/XFeatSystem/src/SuperPointExtractor.cpp -o CMakeFiles/xfeat_system.dir/src/SuperPointExtractor.cpp.s

CMakeFiles/xfeat_system.dir/src/main.cpp.o: CMakeFiles/xfeat_system.dir/flags.make
CMakeFiles/xfeat_system.dir/src/main.cpp.o: ../src/main.cpp
CMakeFiles/xfeat_system.dir/src/main.cpp.o: CMakeFiles/xfeat_system.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/root/jupyter_notebooks/Fyp/XFeatSystem/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Building CXX object CMakeFiles/xfeat_system.dir/src/main.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/xfeat_system.dir/src/main.cpp.o -MF CMakeFiles/xfeat_system.dir/src/main.cpp.o.d -o CMakeFiles/xfeat_system.dir/src/main.cpp.o -c /root/jupyter_notebooks/Fyp/XFeatSystem/src/main.cpp

CMakeFiles/xfeat_system.dir/src/main.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/xfeat_system.dir/src/main.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /root/jupyter_notebooks/Fyp/XFeatSystem/src/main.cpp > CMakeFiles/xfeat_system.dir/src/main.cpp.i

CMakeFiles/xfeat_system.dir/src/main.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/xfeat_system.dir/src/main.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /root/jupyter_notebooks/Fyp/XFeatSystem/src/main.cpp -o CMakeFiles/xfeat_system.dir/src/main.cpp.s

# Object files for target xfeat_system
xfeat_system_OBJECTS = \
"CMakeFiles/xfeat_system.dir/src/XFextractor.cpp.o" \
"CMakeFiles/xfeat_system.dir/src/XFeat.cpp.o" \
"CMakeFiles/xfeat_system.dir/src/SuperPointExtractor.cpp.o" \
"CMakeFiles/xfeat_system.dir/src/main.cpp.o"

# External object files for target xfeat_system
xfeat_system_EXTERNAL_OBJECTS =

xfeat_system: CMakeFiles/xfeat_system.dir/src/XFextractor.cpp.o
xfeat_system: CMakeFiles/xfeat_system.dir/src/XFeat.cpp.o
xfeat_system: CMakeFiles/xfeat_system.dir/src/SuperPointExtractor.cpp.o
xfeat_system: CMakeFiles/xfeat_system.dir/src/main.cpp.o
xfeat_system: CMakeFiles/xfeat_system.dir/build.make
xfeat_system: libsuperpoint_lib.a
xfeat_system: /usr/lib/aarch64-linux-gnu/libopencv_stitching.so.4.5.4d
xfeat_system: /usr/lib/aarch64-linux-gnu/libopencv_alphamat.so.4.5.4d
xfeat_system: /usr/lib/aarch64-linux-gnu/libopencv_aruco.so.4.5.4d
xfeat_system: /usr/lib/aarch64-linux-gnu/libopencv_barcode.so.4.5.4d
xfeat_system: /usr/lib/aarch64-linux-gnu/libopencv_bgsegm.so.4.5.4d
xfeat_system: /usr/lib/aarch64-linux-gnu/libopencv_bioinspired.so.4.5.4d
xfeat_system: /usr/lib/aarch64-linux-gnu/libopencv_ccalib.so.4.5.4d
xfeat_system: /usr/lib/aarch64-linux-gnu/libopencv_dnn_objdetect.so.4.5.4d
xfeat_system: /usr/lib/aarch64-linux-gnu/libopencv_dnn_superres.so.4.5.4d
xfeat_system: /usr/lib/aarch64-linux-gnu/libopencv_dpm.so.4.5.4d
xfeat_system: /usr/lib/aarch64-linux-gnu/libopencv_face.so.4.5.4d
xfeat_system: /usr/lib/aarch64-linux-gnu/libopencv_freetype.so.4.5.4d
xfeat_system: /usr/lib/aarch64-linux-gnu/libopencv_fuzzy.so.4.5.4d
xfeat_system: /usr/lib/aarch64-linux-gnu/libopencv_hdf.so.4.5.4d
xfeat_system: /usr/lib/aarch64-linux-gnu/libopencv_hfs.so.4.5.4d
xfeat_system: /usr/lib/aarch64-linux-gnu/libopencv_img_hash.so.4.5.4d
xfeat_system: /usr/lib/aarch64-linux-gnu/libopencv_intensity_transform.so.4.5.4d
xfeat_system: /usr/lib/aarch64-linux-gnu/libopencv_line_descriptor.so.4.5.4d
xfeat_system: /usr/lib/aarch64-linux-gnu/libopencv_mcc.so.4.5.4d
xfeat_system: /usr/lib/aarch64-linux-gnu/libopencv_quality.so.4.5.4d
xfeat_system: /usr/lib/aarch64-linux-gnu/libopencv_rapid.so.4.5.4d
xfeat_system: /usr/lib/aarch64-linux-gnu/libopencv_reg.so.4.5.4d
xfeat_system: /usr/lib/aarch64-linux-gnu/libopencv_rgbd.so.4.5.4d
xfeat_system: /usr/lib/aarch64-linux-gnu/libopencv_saliency.so.4.5.4d
xfeat_system: /usr/lib/aarch64-linux-gnu/libopencv_shape.so.4.5.4d
xfeat_system: /usr/lib/aarch64-linux-gnu/libopencv_stereo.so.4.5.4d
xfeat_system: /usr/lib/aarch64-linux-gnu/libopencv_structured_light.so.4.5.4d
xfeat_system: /usr/lib/aarch64-linux-gnu/libopencv_superres.so.4.5.4d
xfeat_system: /usr/lib/aarch64-linux-gnu/libopencv_surface_matching.so.4.5.4d
xfeat_system: /usr/lib/aarch64-linux-gnu/libopencv_tracking.so.4.5.4d
xfeat_system: /usr/lib/aarch64-linux-gnu/libopencv_videostab.so.4.5.4d
xfeat_system: /usr/lib/aarch64-linux-gnu/libopencv_viz.so.4.5.4d
xfeat_system: /usr/lib/aarch64-linux-gnu/libopencv_wechat_qrcode.so.4.5.4d
xfeat_system: /usr/lib/aarch64-linux-gnu/libopencv_xobjdetect.so.4.5.4d
xfeat_system: /usr/lib/aarch64-linux-gnu/libopencv_xphoto.so.4.5.4d
xfeat_system: /usr/lib/aarch64-linux-gnu/libopencv_phase_unwrapping.so.4.5.4d
xfeat_system: /usr/lib/aarch64-linux-gnu/libopencv_optflow.so.4.5.4d
xfeat_system: /usr/lib/aarch64-linux-gnu/libopencv_datasets.so.4.5.4d
xfeat_system: /usr/lib/aarch64-linux-gnu/libopencv_plot.so.4.5.4d
xfeat_system: /usr/lib/aarch64-linux-gnu/libopencv_text.so.4.5.4d
xfeat_system: /usr/lib/aarch64-linux-gnu/libopencv_ml.so.4.5.4d
xfeat_system: /usr/lib/aarch64-linux-gnu/libopencv_ximgproc.so.4.5.4d
xfeat_system: /usr/lib/aarch64-linux-gnu/libopencv_objdetect.so.4.5.4d
xfeat_system: /usr/lib/aarch64-linux-gnu/libopencv_photo.so.4.5.4d
xfeat_system: /usr/lib/libvitis_ai_library-dpu_task.so.3.5.0
xfeat_system: /usr/lib/aarch64-linux-gnu/libopencv_highgui.so.4.5.4d
xfeat_system: /usr/lib/aarch64-linux-gnu/libopencv_video.so.4.5.4d
xfeat_system: /usr/lib/aarch64-linux-gnu/libopencv_calib3d.so.4.5.4d
xfeat_system: /usr/lib/aarch64-linux-gnu/libopencv_dnn.so.4.5.4d
xfeat_system: /usr/lib/aarch64-linux-gnu/libopencv_features2d.so.4.5.4d
xfeat_system: /usr/lib/aarch64-linux-gnu/libopencv_flann.so.4.5.4d
xfeat_system: /usr/lib/aarch64-linux-gnu/libopencv_videoio.so.4.5.4d
xfeat_system: /usr/lib/aarch64-linux-gnu/libopencv_imgcodecs.so.4.5.4d
xfeat_system: /usr/lib/aarch64-linux-gnu/libopencv_imgproc.so.4.5.4d
xfeat_system: /usr/lib/libvitis_ai_library-model_config.so.3.5.0
xfeat_system: /usr/lib/aarch64-linux-gnu/libprotobuf.so
xfeat_system: /usr/lib/libvart-runner.so.3.5.0
xfeat_system: /usr/lib/libvitis_ai_library-math.so.3.5.0
xfeat_system: /usr/lib/aarch64-linux-gnu/libopencv_core.so.4.5.4d
xfeat_system: /usr/lib/libvart-util.so.3.5.0
xfeat_system: /usr/lib/libxir.so.3.5.0
xfeat_system: /usr/lib/libunilog.so.3.5.0
xfeat_system: /usr/local/lib/libglog.so.0.5.0
xfeat_system: /usr/lib/aarch64-linux-gnu/libgflags.so.2.2.2
xfeat_system: CMakeFiles/xfeat_system.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/root/jupyter_notebooks/Fyp/XFeatSystem/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_5) "Linking CXX executable xfeat_system"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/xfeat_system.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/xfeat_system.dir/build: xfeat_system
.PHONY : CMakeFiles/xfeat_system.dir/build

CMakeFiles/xfeat_system.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/xfeat_system.dir/cmake_clean.cmake
.PHONY : CMakeFiles/xfeat_system.dir/clean

CMakeFiles/xfeat_system.dir/depend:
	cd /root/jupyter_notebooks/Fyp/XFeatSystem/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /root/jupyter_notebooks/Fyp/XFeatSystem /root/jupyter_notebooks/Fyp/XFeatSystem /root/jupyter_notebooks/Fyp/XFeatSystem/build /root/jupyter_notebooks/Fyp/XFeatSystem/build /root/jupyter_notebooks/Fyp/XFeatSystem/build/CMakeFiles/xfeat_system.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/xfeat_system.dir/depend

