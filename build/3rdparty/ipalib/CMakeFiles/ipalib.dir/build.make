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
include 3rdparty/ipalib/CMakeFiles/ipalib.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include 3rdparty/ipalib/CMakeFiles/ipalib.dir/compiler_depend.make

# Include the progress variables for this target.
include 3rdparty/ipalib/CMakeFiles/ipalib.dir/progress.make

# Include the compile flags for this target's objects.
include 3rdparty/ipalib/CMakeFiles/ipalib.dir/flags.make

3rdparty/ipalib/CMakeFiles/ipalib.dir/ipaConfig.cpp.o: 3rdparty/ipalib/CMakeFiles/ipalib.dir/flags.make
3rdparty/ipalib/CMakeFiles/ipalib.dir/ipaConfig.cpp.o: ../3rdparty/ipalib/ipaConfig.cpp
3rdparty/ipalib/CMakeFiles/ipalib.dir/ipaConfig.cpp.o: 3rdparty/ipalib/CMakeFiles/ipalib.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/achille/Desktop/Projects/TrafficSignDetection-1/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object 3rdparty/ipalib/CMakeFiles/ipalib.dir/ipaConfig.cpp.o"
	cd /home/achille/Desktop/Projects/TrafficSignDetection-1/build/3rdparty/ipalib && /usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT 3rdparty/ipalib/CMakeFiles/ipalib.dir/ipaConfig.cpp.o -MF CMakeFiles/ipalib.dir/ipaConfig.cpp.o.d -o CMakeFiles/ipalib.dir/ipaConfig.cpp.o -c /home/achille/Desktop/Projects/TrafficSignDetection-1/3rdparty/ipalib/ipaConfig.cpp

3rdparty/ipalib/CMakeFiles/ipalib.dir/ipaConfig.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/ipalib.dir/ipaConfig.cpp.i"
	cd /home/achille/Desktop/Projects/TrafficSignDetection-1/build/3rdparty/ipalib && /usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/achille/Desktop/Projects/TrafficSignDetection-1/3rdparty/ipalib/ipaConfig.cpp > CMakeFiles/ipalib.dir/ipaConfig.cpp.i

3rdparty/ipalib/CMakeFiles/ipalib.dir/ipaConfig.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/ipalib.dir/ipaConfig.cpp.s"
	cd /home/achille/Desktop/Projects/TrafficSignDetection-1/build/3rdparty/ipalib && /usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/achille/Desktop/Projects/TrafficSignDetection-1/3rdparty/ipalib/ipaConfig.cpp -o CMakeFiles/ipalib.dir/ipaConfig.cpp.s

# Object files for target ipalib
ipalib_OBJECTS = \
"CMakeFiles/ipalib.dir/ipaConfig.cpp.o"

# External object files for target ipalib
ipalib_EXTERNAL_OBJECTS =

libs/libipalib.a: 3rdparty/ipalib/CMakeFiles/ipalib.dir/ipaConfig.cpp.o
libs/libipalib.a: 3rdparty/ipalib/CMakeFiles/ipalib.dir/build.make
libs/libipalib.a: 3rdparty/ipalib/CMakeFiles/ipalib.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/achille/Desktop/Projects/TrafficSignDetection-1/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX static library ../../libs/libipalib.a"
	cd /home/achille/Desktop/Projects/TrafficSignDetection-1/build/3rdparty/ipalib && $(CMAKE_COMMAND) -P CMakeFiles/ipalib.dir/cmake_clean_target.cmake
	cd /home/achille/Desktop/Projects/TrafficSignDetection-1/build/3rdparty/ipalib && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/ipalib.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
3rdparty/ipalib/CMakeFiles/ipalib.dir/build: libs/libipalib.a
.PHONY : 3rdparty/ipalib/CMakeFiles/ipalib.dir/build

3rdparty/ipalib/CMakeFiles/ipalib.dir/clean:
	cd /home/achille/Desktop/Projects/TrafficSignDetection-1/build/3rdparty/ipalib && $(CMAKE_COMMAND) -P CMakeFiles/ipalib.dir/cmake_clean.cmake
.PHONY : 3rdparty/ipalib/CMakeFiles/ipalib.dir/clean

3rdparty/ipalib/CMakeFiles/ipalib.dir/depend:
	cd /home/achille/Desktop/Projects/TrafficSignDetection-1/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/achille/Desktop/Projects/TrafficSignDetection-1 /home/achille/Desktop/Projects/TrafficSignDetection-1/3rdparty/ipalib /home/achille/Desktop/Projects/TrafficSignDetection-1/build /home/achille/Desktop/Projects/TrafficSignDetection-1/build/3rdparty/ipalib /home/achille/Desktop/Projects/TrafficSignDetection-1/build/3rdparty/ipalib/CMakeFiles/ipalib.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : 3rdparty/ipalib/CMakeFiles/ipalib.dir/depend
