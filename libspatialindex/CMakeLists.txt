#
# top-level CMake configuration file for libspatialindex
#
# (based originally on the PDAL and libLAS files copyright Mateusz Loskot)

cmake_minimum_required(VERSION 3.28)

#------------------------------------------------------------------------------
# libspatialindex general settings
#------------------------------------------------------------------------------


project(spatialindex
  LANGUAGES CXX
  VERSION "2.0.0"
  HOMEPAGE_URL "https://github.com/libspatialindex/libspatialindex"
)

set(CMAKE_CXX_STANDARD 17
        CACHE STRING "C++ standard version to use (default is 17)")
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(SIDX_VERSION_MAJOR "${PROJECT_VERSION_MAJOR}")
set(SIDX_VERSION_MINOR "${PROJECT_VERSION_MINOR}")
set(SIDX_VERSION_PATCH "${PROJECT_VERSION_PATCH}")
set(SIDX_VERSION_STRING "${PROJECT_VERSION}")

set(SIDX_LIB_VERSION "7.0.0")
set(SIDX_LIB_SOVERSION "7")

execute_process(COMMAND
        python -c "import torch;print(torch.utils.cmake_prefix_path)"
        OUTPUT_VARIABLE TORCH_DIR
)

message(STATUS
        "TORCH_DIR is ${TORCH_DIR}")
string(REGEX MATCH ".*/torch/" TORCH_DIR ${TORCH_DIR})
list(APPEND CMAKE_PREFIX_PATH ${TORCH_DIR})


if(EXISTS "/usr/local/cuda/bin/nvcc")
    set(CMAKE_CUDA_ARCHITECTURES 120) # For: Ampere and Ada Lovelace
    set(CMAKE_CUDA_COMPILER "/usr/local/cuda/bin/nvcc")
    enable_language(CUDA)

    set(CAFFE2_USE_CUDNN ON)
    set(USE_CUSPARSELT OFF)
    find_package(CUDAToolkit)
else()
    message(WARNING "CUDA not found - some features will be disabled")
endif ()

find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

include(GNUInstallDirs)


IF(NOT CMAKE_BUILD_TYPE)
    SET(CMAKE_BUILD_TYPE Release CACHE STRING
        "Choose the type of build, options are: None Debug Release RelWithDebInfo MinSizeRel"
        FORCE)
ENDIF()


message(STATUS
  "Configuring CMake ${CMAKE_VERSION} to build spatialindex ${PROJECT_VERSION}")

include(CTest)
option(BUILD_SHARED_LIBS "Build shared libraries" ON)
option(BUILD_TESTING "Enables integrated test suites" OFF)

if(DEFINED SIDX_BUILD_TESTS)
  message(DEPRECATION "SIDX_BUILD_TESTS has been replaced with BUILD_TESTING")
  set(BUILD_TESTING ${SIDX_BUILD_TESTS})
endif()
include(CTest)

# C++11 required, overridable by user


# Name of C++ library
set(SIDX_LIB_NAME spatialindex)
set(SIDX_C_LIB_NAME spatialindex_c)
set(SIDX_UTILS_LIB_NAME spatialindex_utils)

if(WIN32 AND MSVC)
  if(CMAKE_SIZEOF_VOID_P EQUAL 8)
    set(SIDX_LIB_NAME "${SIDX_LIB_NAME}-64")
    set(SIDX_C_LIB_NAME "${SIDX_C_LIB_NAME}-64")
    set(SIDX_UTILS_LIB_NAME "${SIDX_UTILS_LIB_NAME}-64")
  else()
    set(SIDX_LIB_NAME "${SIDX_LIB_NAME}-32")
    set(SIDX_C_LIB_NAME "${SIDX_C_LIB_NAME}-32")
    set(SIDX_UTILS_LIB_NAME "${SIDX_UTILS_LIB_NAME}-32")
  endif()
endif()


set(CMAKE_INCLUDE_DIRECTORIES_PROJECT_BEFORE ON)

include(CheckFunctionExists)

check_function_exists(srand48 HAVE_SRAND48)
check_function_exists(gettimeofday HAVE_GETTIMEOFDAY)
check_function_exists(memset HAVE_MEMSET)
check_function_exists(memcpy HAVE_MEMCPY)
check_function_exists(bcopy HAVE_BCOPY)


include(CheckIncludeFiles)

add_subdirectory(src)

if(BUILD_TESTING)
  add_subdirectory(test)
endif()



#------------------------------------------------------------------------------
# CMake package configuration for find_package(libspatialindex)
#------------------------------------------------------------------------------

include(CMakePackageConfigHelpers)

set(INCLUDE_INSTALL_DIR ${CMAKE_INSTALL_INCLUDEDIR} CACHE PATH "include directory")
set(LIB_INSTALL_DIR ${CMAKE_INSTALL_LIBDIR} CACHE PATH "lib directory")

configure_package_config_file(
  libspatialindexConfig.cmake.in
  ${CMAKE_CURRENT_BINARY_DIR}/libspatialindexConfig.cmake
  INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/libspatialindex
  PATH_VARS INCLUDE_INSTALL_DIR LIB_INSTALL_DIR)

write_basic_package_version_file(
  ${CMAKE_CURRENT_BINARY_DIR}/libspatialindexConfigVersion.cmake
  VERSION ${SIDX_VERSION_STRING}
  COMPATIBILITY AnyNewerVersion)

install(FILES
  ${CMAKE_CURRENT_BINARY_DIR}/libspatialindexConfig.cmake
  ${CMAKE_CURRENT_BINARY_DIR}/libspatialindexConfigVersion.cmake
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/libspatialindex)

#------------------------------------------------------------------------------
# pkg-config support
#------------------------------------------------------------------------------
if(NOT WIN32)
  configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/src/libspatialindex.pc.in
    ${CMAKE_CURRENT_BINARY_DIR}/libspatialindex.pc
    @ONLY)

  install(FILES
    ${CMAKE_CURRENT_BINARY_DIR}/libspatialindex.pc
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/pkgconfig)
endif()

#------------------------------------------------------------------------------
# CPack controls
#------------------------------------------------------------------------------

set(CPACK_PACKAGE_VERSION_MAJOR ${PROJECT_VERSION_MAJOR})
set(CPACK_PACKAGE_VERSION_MINOR ${PROJECT_VERSION_MINOR})
set(CPACK_PACKAGE_VERSION_PATCH ${PROJECT_VERSION_MINOR})
set(CPACK_PACKAGE_VERSION_TWEAK ${PROJECT_VERSION_TWEAK})
set(CPACK_PACKAGE_NAME "libspatialindex")

set(CPACK_SOURCE_GENERATOR "TBZ2;TGZ")
set(CPACK_PACKAGE_VENDOR "libspatialindex Development Team")
set(CPACK_RESOURCE_FILE_LICENSE "${PROJECT_SOURCE_DIR}/COPYING")

set(CPACK_SOURCE_PACKAGE_FILE_NAME
    "${CMAKE_PROJECT_NAME}-src-${PROJECT_VERSION}")

set(CPACK_SOURCE_IGNORE_FILES
  "/\\\\.gitattributes"
  "/\\\\.vagrant"
  "/\\\\.DS_Store"
  "/CVS/"
  "/\\\\.git/"
  "\\\\.swp$"
  "~$"
  "\\\\.\\\\#"
  "/\\\\#"
  "CMakeScripts/"
  "_CPack_Packages"
  "cmake_install.cmake"
  "/bin/"
  "/scripts/"
  "/azure-pipelines.yml"
  ".gitignore"
  ".ninja*"
  "HOWTORELEASE.txt"
  "README"
  "build/"
  "CMakeFiles"
  "CTestTestfile.cmake"
  "/docs/build/"
  "/doc/presentations/"
  "package-release.sh"
  "docker-package.sh"
  ".gz"
  ".bz2"
)

include(CPack)
add_custom_target(dist COMMAND ${CMAKE_MAKE_PROGRAM} package_source)
