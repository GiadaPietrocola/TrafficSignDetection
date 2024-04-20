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
CMAKE_SOURCE_DIR = /home/achille/Desktop/Projects/TrafficSignDetection-1

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/achille/Desktop/Projects/TrafficSignDetection-1/build

# Include any dependencies generated for this target.
include 3rdparty/ucaslib/CMakeFiles/ucaslib.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include 3rdparty/ucaslib/CMakeFiles/ucaslib.dir/compiler_depend.make

# Include the progress variables for this target.
include 3rdparty/ucaslib/CMakeFiles/ucaslib.dir/progress.make

# Include the compile flags for this target's objects.
include 3rdparty/ucaslib/CMakeFiles/ucaslib.dir/flags.make

3rdparty/ucaslib/CMakeFiles/ucaslib.dir/ucasBreastUtils.cpp.o: 3rdparty/ucaslib/CMakeFiles/ucaslib.dir/flags.make
3rdparty/ucaslib/CMakeFiles/ucaslib.dir/ucasBreastUtils.cpp.o: ../3rdparty/ucaslib/ucasBreastUtils.cpp
3rdparty/ucaslib/CMakeFiles/ucaslib.dir/ucasBreastUtils.cpp.o: 3rdparty/ucaslib/CMakeFiles/ucaslib.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/achille/Desktop/Projects/TrafficSignDetection-1/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object 3rdparty/ucaslib/CMakeFiles/ucaslib.dir/ucasBreastUtils.cpp.o"
	cd /home/achille/Desktop/Projects/TrafficSignDetection-1/build/3rdparty/ucaslib && /usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT 3rdparty/ucaslib/CMakeFiles/ucaslib.dir/ucasBreastUtils.cpp.o -MF CMakeFiles/ucaslib.dir/ucasBreastUtils.cpp.o.d -o CMakeFiles/ucaslib.dir/ucasBreastUtils.cpp.o -c /home/achille/Desktop/Projects/TrafficSignDetection-1/3rdparty/ucaslib/ucasBreastUtils.cpp

3rdparty/ucaslib/CMakeFiles/ucaslib.dir/ucasBreastUtils.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/ucaslib.dir/ucasBreastUtils.cpp.i"
	cd /home/achille/Desktop/Projects/TrafficSignDetection-1/build/3rdparty/ucaslib && /usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/achille/Desktop/Projects/TrafficSignDetection-1/3rdparty/ucaslib/ucasBreastUtils.cpp > CMakeFiles/ucaslib.dir/ucasBreastUtils.cpp.i

3rdparty/ucaslib/CMakeFiles/ucaslib.dir/ucasBreastUtils.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/ucaslib.dir/ucasBreastUtils.cpp.s"
	cd /home/achille/Desktop/Projects/TrafficSignDetection-1/build/3rdparty/ucaslib && /usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/achille/Desktop/Projects/TrafficSignDetection-1/3rdparty/ucaslib/ucasBreastUtils.cpp -o CMakeFiles/ucaslib.dir/ucasBreastUtils.cpp.s

3rdparty/ucaslib/CMakeFiles/ucaslib.dir/ucasConfig.cpp.o: 3rdparty/ucaslib/CMakeFiles/ucaslib.dir/flags.make
3rdparty/ucaslib/CMakeFiles/ucaslib.dir/ucasConfig.cpp.o: ../3rdparty/ucaslib/ucasConfig.cpp
3rdparty/ucaslib/CMakeFiles/ucaslib.dir/ucasConfig.cpp.o: 3rdparty/ucaslib/CMakeFiles/ucaslib.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/achille/Desktop/Projects/TrafficSignDetection-1/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object 3rdparty/ucaslib/CMakeFiles/ucaslib.dir/ucasConfig.cpp.o"
	cd /home/achille/Desktop/Projects/TrafficSignDetection-1/build/3rdparty/ucaslib && /usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT 3rdparty/ucaslib/CMakeFiles/ucaslib.dir/ucasConfig.cpp.o -MF CMakeFiles/ucaslib.dir/ucasConfig.cpp.o.d -o CMakeFiles/ucaslib.dir/ucasConfig.cpp.o -c /home/achille/Desktop/Projects/TrafficSignDetection-1/3rdparty/ucaslib/ucasConfig.cpp

3rdparty/ucaslib/CMakeFiles/ucaslib.dir/ucasConfig.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/ucaslib.dir/ucasConfig.cpp.i"
	cd /home/achille/Desktop/Projects/TrafficSignDetection-1/build/3rdparty/ucaslib && /usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/achille/Desktop/Projects/TrafficSignDetection-1/3rdparty/ucaslib/ucasConfig.cpp > CMakeFiles/ucaslib.dir/ucasConfig.cpp.i

3rdparty/ucaslib/CMakeFiles/ucaslib.dir/ucasConfig.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/ucaslib.dir/ucasConfig.cpp.s"
	cd /home/achille/Desktop/Projects/TrafficSignDetection-1/build/3rdparty/ucaslib && /usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/achille/Desktop/Projects/TrafficSignDetection-1/3rdparty/ucaslib/ucasConfig.cpp -o CMakeFiles/ucaslib.dir/ucasConfig.cpp.s

3rdparty/ucaslib/CMakeFiles/ucaslib.dir/ucasImageUtils.cpp.o: 3rdparty/ucaslib/CMakeFiles/ucaslib.dir/flags.make
3rdparty/ucaslib/CMakeFiles/ucaslib.dir/ucasImageUtils.cpp.o: ../3rdparty/ucaslib/ucasImageUtils.cpp
3rdparty/ucaslib/CMakeFiles/ucaslib.dir/ucasImageUtils.cpp.o: 3rdparty/ucaslib/CMakeFiles/ucaslib.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/achille/Desktop/Projects/TrafficSignDetection-1/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Building CXX object 3rdparty/ucaslib/CMakeFiles/ucaslib.dir/ucasImageUtils.cpp.o"
	cd /home/achille/Desktop/Projects/TrafficSignDetection-1/build/3rdparty/ucaslib && /usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT 3rdparty/ucaslib/CMakeFiles/ucaslib.dir/ucasImageUtils.cpp.o -MF CMakeFiles/ucaslib.dir/ucasImageUtils.cpp.o.d -o CMakeFiles/ucaslib.dir/ucasImageUtils.cpp.o -c /home/achille/Desktop/Projects/TrafficSignDetection-1/3rdparty/ucaslib/ucasImageUtils.cpp

3rdparty/ucaslib/CMakeFiles/ucaslib.dir/ucasImageUtils.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/ucaslib.dir/ucasImageUtils.cpp.i"
	cd /home/achille/Desktop/Projects/TrafficSignDetection-1/build/3rdparty/ucaslib && /usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/achille/Desktop/Projects/TrafficSignDetection-1/3rdparty/ucaslib/ucasImageUtils.cpp > CMakeFiles/ucaslib.dir/ucasImageUtils.cpp.i

3rdparty/ucaslib/CMakeFiles/ucaslib.dir/ucasImageUtils.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/ucaslib.dir/ucasImageUtils.cpp.s"
	cd /home/achille/Desktop/Projects/TrafficSignDetection-1/build/3rdparty/ucaslib && /usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/achille/Desktop/Projects/TrafficSignDetection-1/3rdparty/ucaslib/ucasImageUtils.cpp -o CMakeFiles/ucaslib.dir/ucasImageUtils.cpp.s

3rdparty/ucaslib/CMakeFiles/ucaslib.dir/ucasMachineLearningUtils.cpp.o: 3rdparty/ucaslib/CMakeFiles/ucaslib.dir/flags.make
3rdparty/ucaslib/CMakeFiles/ucaslib.dir/ucasMachineLearningUtils.cpp.o: ../3rdparty/ucaslib/ucasMachineLearningUtils.cpp
3rdparty/ucaslib/CMakeFiles/ucaslib.dir/ucasMachineLearningUtils.cpp.o: 3rdparty/ucaslib/CMakeFiles/ucaslib.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/achille/Desktop/Projects/TrafficSignDetection-1/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Building CXX object 3rdparty/ucaslib/CMakeFiles/ucaslib.dir/ucasMachineLearningUtils.cpp.o"
	cd /home/achille/Desktop/Projects/TrafficSignDetection-1/build/3rdparty/ucaslib && /usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT 3rdparty/ucaslib/CMakeFiles/ucaslib.dir/ucasMachineLearningUtils.cpp.o -MF CMakeFiles/ucaslib.dir/ucasMachineLearningUtils.cpp.o.d -o CMakeFiles/ucaslib.dir/ucasMachineLearningUtils.cpp.o -c /home/achille/Desktop/Projects/TrafficSignDetection-1/3rdparty/ucaslib/ucasMachineLearningUtils.cpp

3rdparty/ucaslib/CMakeFiles/ucaslib.dir/ucasMachineLearningUtils.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/ucaslib.dir/ucasMachineLearningUtils.cpp.i"
	cd /home/achille/Desktop/Projects/TrafficSignDetection-1/build/3rdparty/ucaslib && /usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/achille/Desktop/Projects/TrafficSignDetection-1/3rdparty/ucaslib/ucasMachineLearningUtils.cpp > CMakeFiles/ucaslib.dir/ucasMachineLearningUtils.cpp.i

3rdparty/ucaslib/CMakeFiles/ucaslib.dir/ucasMachineLearningUtils.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/ucaslib.dir/ucasMachineLearningUtils.cpp.s"
	cd /home/achille/Desktop/Projects/TrafficSignDetection-1/build/3rdparty/ucaslib && /usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/achille/Desktop/Projects/TrafficSignDetection-1/3rdparty/ucaslib/ucasMachineLearningUtils.cpp -o CMakeFiles/ucaslib.dir/ucasMachineLearningUtils.cpp.s

3rdparty/ucaslib/CMakeFiles/ucaslib.dir/ucasMultithreading.cpp.o: 3rdparty/ucaslib/CMakeFiles/ucaslib.dir/flags.make
3rdparty/ucaslib/CMakeFiles/ucaslib.dir/ucasMultithreading.cpp.o: ../3rdparty/ucaslib/ucasMultithreading.cpp
3rdparty/ucaslib/CMakeFiles/ucaslib.dir/ucasMultithreading.cpp.o: 3rdparty/ucaslib/CMakeFiles/ucaslib.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/achille/Desktop/Projects/TrafficSignDetection-1/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_5) "Building CXX object 3rdparty/ucaslib/CMakeFiles/ucaslib.dir/ucasMultithreading.cpp.o"
	cd /home/achille/Desktop/Projects/TrafficSignDetection-1/build/3rdparty/ucaslib && /usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT 3rdparty/ucaslib/CMakeFiles/ucaslib.dir/ucasMultithreading.cpp.o -MF CMakeFiles/ucaslib.dir/ucasMultithreading.cpp.o.d -o CMakeFiles/ucaslib.dir/ucasMultithreading.cpp.o -c /home/achille/Desktop/Projects/TrafficSignDetection-1/3rdparty/ucaslib/ucasMultithreading.cpp

3rdparty/ucaslib/CMakeFiles/ucaslib.dir/ucasMultithreading.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/ucaslib.dir/ucasMultithreading.cpp.i"
	cd /home/achille/Desktop/Projects/TrafficSignDetection-1/build/3rdparty/ucaslib && /usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/achille/Desktop/Projects/TrafficSignDetection-1/3rdparty/ucaslib/ucasMultithreading.cpp > CMakeFiles/ucaslib.dir/ucasMultithreading.cpp.i

3rdparty/ucaslib/CMakeFiles/ucaslib.dir/ucasMultithreading.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/ucaslib.dir/ucasMultithreading.cpp.s"
	cd /home/achille/Desktop/Projects/TrafficSignDetection-1/build/3rdparty/ucaslib && /usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/achille/Desktop/Projects/TrafficSignDetection-1/3rdparty/ucaslib/ucasMultithreading.cpp -o CMakeFiles/ucaslib.dir/ucasMultithreading.cpp.s

# Object files for target ucaslib
ucaslib_OBJECTS = \
"CMakeFiles/ucaslib.dir/ucasBreastUtils.cpp.o" \
"CMakeFiles/ucaslib.dir/ucasConfig.cpp.o" \
"CMakeFiles/ucaslib.dir/ucasImageUtils.cpp.o" \
"CMakeFiles/ucaslib.dir/ucasMachineLearningUtils.cpp.o" \
"CMakeFiles/ucaslib.dir/ucasMultithreading.cpp.o"

# External object files for target ucaslib
ucaslib_EXTERNAL_OBJECTS =

libs/libucaslib.a: 3rdparty/ucaslib/CMakeFiles/ucaslib.dir/ucasBreastUtils.cpp.o
libs/libucaslib.a: 3rdparty/ucaslib/CMakeFiles/ucaslib.dir/ucasConfig.cpp.o
libs/libucaslib.a: 3rdparty/ucaslib/CMakeFiles/ucaslib.dir/ucasImageUtils.cpp.o
libs/libucaslib.a: 3rdparty/ucaslib/CMakeFiles/ucaslib.dir/ucasMachineLearningUtils.cpp.o
libs/libucaslib.a: 3rdparty/ucaslib/CMakeFiles/ucaslib.dir/ucasMultithreading.cpp.o
libs/libucaslib.a: 3rdparty/ucaslib/CMakeFiles/ucaslib.dir/build.make
libs/libucaslib.a: 3rdparty/ucaslib/CMakeFiles/ucaslib.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/achille/Desktop/Projects/TrafficSignDetection-1/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_6) "Linking CXX static library ../../libs/libucaslib.a"
	cd /home/achille/Desktop/Projects/TrafficSignDetection-1/build/3rdparty/ucaslib && $(CMAKE_COMMAND) -P CMakeFiles/ucaslib.dir/cmake_clean_target.cmake
	cd /home/achille/Desktop/Projects/TrafficSignDetection-1/build/3rdparty/ucaslib && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/ucaslib.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
3rdparty/ucaslib/CMakeFiles/ucaslib.dir/build: libs/libucaslib.a
.PHONY : 3rdparty/ucaslib/CMakeFiles/ucaslib.dir/build

3rdparty/ucaslib/CMakeFiles/ucaslib.dir/clean:
	cd /home/achille/Desktop/Projects/TrafficSignDetection-1/build/3rdparty/ucaslib && $(CMAKE_COMMAND) -P CMakeFiles/ucaslib.dir/cmake_clean.cmake
.PHONY : 3rdparty/ucaslib/CMakeFiles/ucaslib.dir/clean

3rdparty/ucaslib/CMakeFiles/ucaslib.dir/depend:
	cd /home/achille/Desktop/Projects/TrafficSignDetection-1/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/achille/Desktop/Projects/TrafficSignDetection-1 /home/achille/Desktop/Projects/TrafficSignDetection-1/3rdparty/ucaslib /home/achille/Desktop/Projects/TrafficSignDetection-1/build /home/achille/Desktop/Projects/TrafficSignDetection-1/build/3rdparty/ucaslib /home/achille/Desktop/Projects/TrafficSignDetection-1/build/3rdparty/ucaslib/CMakeFiles/ucaslib.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : 3rdparty/ucaslib/CMakeFiles/ucaslib.dir/depend
