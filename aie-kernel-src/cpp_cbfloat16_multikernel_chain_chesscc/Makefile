##===- Makefile -----------------------------------------------------------===##
#
# This file licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
##===----------------------------------------------------------------------===##
.DEFAULT_GOAL := compile

srcdir := $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))

export LD_LIBRARY_PATH := /opt/vitis_aie_essentials/lib/lnx64.o/:$(LD_LIBRARY_PATH)
export RDI_DATADIR := /opt/vitis_aie_essentials/data
export PATH := /opt/vitis_aie_essentials/tps/lnx64/target_aie_ml/bin/LNa64bin/:$(PATH)

# mlir-aie/programming_examples/makefile-common
AIETOOLS_DIR = /opt/vitis_aie_essentials/
AIE_INCLUDE_DIR ?= ${AIETOOLS_DIR}/data/versal_prod/lib
AIE2_INCLUDE_DIR ?= ${AIETOOLS_DIR}/data/aie_ml/lib

MLIR_AIE_DIR ?= $(shell python3 -c "from aie.utils.config import root_path; print(root_path())")

# Inter-core object fifo depth: 1 => single-buffered (bubbled pipeline),
# >= 2 => double-buffered (properly fed pipeline). See README.md.
FIFO_DEPTH ?= 1

CHESSCC1_FLAGS = -f -p me -P ${AIE_INCLUDE_DIR} -I ${AIETOOLS_DIR}/include
CHESSCC2_FLAGS = -f -p me -P ${AIE2_INCLUDE_DIR} -I ${AIETOOLS_DIR}/include
CHESS_FLAGS = -P ${AIE_INCLUDE_DIR}

CHESSCCWRAP1_FLAGS = aie -I ${AIETOOLS_DIR}/include
CHESSCCWRAP2_FLAGS = aie2 -I ${AIETOOLS_DIR}/include
CHESSCCWRAP2P_FLAGS = aie2p -I ${AIETOOLS_DIR}/include

all: build/final.xclbin build/insts.bin

devicename ?= $(if $(filter 1,$(NPU2)),npu2,npu)

ifeq ($(devicename),npu2)
KERNEL_CHESSCCWRAP_FLAGS = ${CHESSCCWRAP2P_FLAGS}
else
KERNEL_CHESSCCWRAP_FLAGS = ${CHESSCCWRAP2_FLAGS}
endif

# One object per stage kernel (stateful; see the stage_*.cc files)
KERNEL_OBJS = \
	build/stage_scale_mul.o \
	build/stage_rotate90.o \
	build/stage_add_const.o

build/aie.mlir: ${srcdir}/aie2.py
	mkdir -p ${@D}
	FIFO_DEPTH=$(FIFO_DEPTH) python3 $< ${devicename} > $@

build/stage_scale_mul.o: ${srcdir}/stage_scale_mul.cc ${srcdir}/../aie_kernel_utils.h
	mkdir -p ${@D}
	cd ${@D} && xchesscc_wrapper ${KERNEL_CHESSCCWRAP_FLAGS} -c $< -o ${@F}

build/stage_rotate90.o: ${srcdir}/stage_rotate90.cc ${srcdir}/../aie_kernel_utils.h
	mkdir -p ${@D}
	cd ${@D} && xchesscc_wrapper ${KERNEL_CHESSCCWRAP_FLAGS} -c $< -o ${@F}

build/stage_add_const.o: ${srcdir}/stage_add_const.cc ${srcdir}/../aie_kernel_utils.h
	mkdir -p ${@D}
	cd ${@D} && xchesscc_wrapper ${KERNEL_CHESSCCWRAP_FLAGS} -c $< -o ${@F}

build/final.xclbin: build/aie.mlir ${KERNEL_OBJS}
	mkdir -p ${@D}
	cd ${@D} && aiecc.py --aie-generate-xclbin --no-compile-host --xclbin-name=${@F} \
				--aie-generate-npu-insts --npu-insts-name=insts.bin $(<:%=../%)

build/insts.bin: build/final.xclbin
	@test -f $@

test: all
	python3 ${srcdir}/test.py

clean:
	rm -rf build _build

compile: clean build/final.xclbin build/insts.bin

.PHONY: all test clean compile
