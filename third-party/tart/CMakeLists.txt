cmake_minimum_required(VERSION 3.15)

project(tart)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

find_package(Vulkan)
if( NOT (DEFINED TART_BUILD_LOADER) )
	if(TARGET Vulkan::Vulkan)
		set(TART_BUILD_LOADER 0)
	else()
		set(TART_BUILD_LOADER 1)
	endif()
endif()

include_directories(VulkanMemoryAllocator/include)
add_subdirectory(VulkanMemoryAllocator)

if(TART_BUILD_LOADER)
	add_subdirectory(Vulkan-Loader)
	add_subdirectory(Vulkan-Headers)
	include_directories(Vulkan-Headers/include)
else()
	find_package(Vulkan REQUIRED)
endif()
set(Python_FIND_VIRTUALENV FIRST)
find_package(Python3 COMPONENTS Interpreter Development)
#FindPython(Interpreter Compiler Development Development.Module)

set(TART_ENABLE_SHADER_COMPILERS 0)
if( TART_ENABLE_SHADER_COMPILERS )
	# run the dependency initialization scripts for each compiler, then add them as subdirectories
	execute_process(
		COMMAND ${Python_EXECUTABLE} "${PROJECT_SOURCE_DIR}/glslang/update_glslang_sources.py"
		WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}/glslang
	)
	add_subdirectory(glslang)
endif()

add_subdirectory(${PROJECT_SOURCE_DIR}/subprocess/src/cpp)

# subprocess forgot to put this here :c
set_target_properties(subprocess
    PROPERTIES
        CXX_STANDARD 17
        CXX_STANDARD_REQUIRED YES
        CXX_EXTENSIONS NO
)

include_directories(include)
include_directories(SPIRV-Reflect)

add_library(tart STATIC "src/tart-gpu.cpp" "src/tart-vk-helpers.cpp" "src/tart-compilers.cpp" "src/tart-internal.cpp" "src/tart-device.cpp" "src/tart-shader.cpp" "src/tart-vma-impl.cpp")
target_link_libraries(tart PRIVATE subprocess)
if(TART_BUILD_LOADER)
	target_compile_definitions(tart PUBLIC ENABLE_LOCAL_LOADER)
	set_target_properties(vulkan PROPERTIES SOVERSION 1)
	target_link_libraries(tart PUBLIC vulkan)
else()
	target_link_libraries(tart PUBLIC Vulkan::Vulkan)
endif()

# VMA
#target_link_libraries(tart PUBLIC GPUOpen::VulkanMemoryAllocator)

if( TART_ENABLE_SHADER_COMPILERS )
	# clspv
	#target_link_libraries(tart-compilers PRIVATE clspv_core)
	target_compile_definitions(tart PUBLIC ENABLE_LINKED_SHADER_COMPILERS)
	
	# glslang
	target_link_libraries(tart PRIVATE glslang)
	target_link_libraries(tart PRIVATE glslang-default-resource-limits)
elseif( TART_ENABLE_SHARED_SHADER_COMPILERS )
	# link to existing shared libraries if they exist
	find_package(glslang REQUIRED)
	target_compile_definitions(tart PUBLIC ENABLE_LINKED_SHADER_COMPILERS)
	target_link_libraries(tart PRIVATE glslang)
	target_link_libraries(tart PRIVATE glslang-default-resource-limits)
endif()

if( ENABLE_VALIDATION_LAYERS )
	target_compile_definitions(tart PUBLIC TART_ENABLE_VALIDATION)
endif()

if(TART_ENABLE_TESTS)
	include_directories(tests)
	add_executable(run-tests "tests/tests.cpp")
	target_link_libraries(run-tests PUBLIC tart)
endif()

if( TART_ENABLE_PYBIND )
	set(PYBIND11_FINDPYTHON ON)
	find_package(pybind11 CONFIG REQUIRED)
	
	# this may not work
	pybind11_add_module(pytart src/pytart.cpp)
	target_link_libraries(pytart PUBLIC tart)
	target_link_libraries(pytart PRIVATE pybind11::headers)
	
	install(TARGETS pytart DESTINATION .)
	if(TART_BUILD_LOADER)
		install(TARGETS vulkan DESTINATION .)
	endif()
endif()
message("${CMAKE_CXX_COMPILER}")
