##===- Makefile -----------------------------------------------------------===##
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
##===----------------------------------------------------------------------===##
.DEFAULT_GOAL := compile

srcdir := $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))

USE_CHESS = 1
include $(srcdir)/../makefile-common

AIE_INCLUDE_DIR ?= ${AIETOOLS_DIR}/data/versal_prod/lib
AIE2_INCLUDE_DIR ?= ${AIETOOLS_DIR}/data/aie_ml/lib

CHESSCCWRAP2_FLAGS = aie2 -I ${AIETOOLS_DIR}/include
CHESSCCWRAP2P_FLAGS = aie2p -I ${AIETOOLS_DIR}/include

targetname = cint16ToCbfloat16
devicename ?= $(if $(filter 1,$(NPU2)),npu2,npu)

ifeq ($(devicename),npu2)
KERNEL_CHESSCCWRAP_FLAGS = ${CHESSCCWRAP2P_FLAGS}
else
KERNEL_CHESSCCWRAP_FLAGS = ${CHESSCCWRAP2_FLAGS}
endif

build/aie.mlir: ${srcdir}/aie2.py
	mkdir -p ${@D}
	python3 $< ${devicename} > $@

build/convert.o: ${srcdir}/convert.cc
	mkdir -p ${@D}
	cd ${@D} && xchesscc_wrapper ${KERNEL_CHESSCCWRAP_FLAGS} -c $< -o ${@F}

build/final.xclbin: build/aie.mlir build/convert.o
	mkdir -p ${@D}
	cd ${@D} && $(AIECC) $(AIECC_FLAGS) --xclbin-name=${@F} \
			--npu-insts-name=insts.bin $(<:%=../%)

# aiecc.py emits insts.bin as a side effect of the xclbin target.
build/insts.bin: build/final.xclbin
	@:

clean:
	rm -rf build _build

compile: clean build/final.xclbin build/insts.bin
