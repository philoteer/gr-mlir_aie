find_package(PkgConfig)

PKG_CHECK_MODULES(PC_GR_MLIR_AIE gnuradio-mlir_aie)

FIND_PATH(
    GR_MLIR_AIE_INCLUDE_DIRS
    NAMES gnuradio/mlir_aie/api.h
    HINTS $ENV{MLIR_AIE_DIR}/include
        ${PC_MLIR_AIE_INCLUDEDIR}
    PATHS ${CMAKE_INSTALL_PREFIX}/include
          /usr/local/include
          /usr/include
)

FIND_LIBRARY(
    GR_MLIR_AIE_LIBRARIES
    NAMES gnuradio-mlir_aie
    HINTS $ENV{MLIR_AIE_DIR}/lib
        ${PC_MLIR_AIE_LIBDIR}
    PATHS ${CMAKE_INSTALL_PREFIX}/lib
          ${CMAKE_INSTALL_PREFIX}/lib64
          /usr/local/lib
          /usr/local/lib64
          /usr/lib
          /usr/lib64
          )

include("${CMAKE_CURRENT_LIST_DIR}/gnuradio-mlir_aieTarget.cmake")

INCLUDE(FindPackageHandleStandardArgs)
FIND_PACKAGE_HANDLE_STANDARD_ARGS(GR_MLIR_AIE DEFAULT_MSG GR_MLIR_AIE_LIBRARIES GR_MLIR_AIE_INCLUDE_DIRS)
MARK_AS_ADVANCED(GR_MLIR_AIE_LIBRARIES GR_MLIR_AIE_INCLUDE_DIRS)
