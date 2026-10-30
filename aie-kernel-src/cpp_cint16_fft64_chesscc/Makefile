# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

.DEFAULT_GOAL := all

srcdir := $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))
USE_CHESS = 1
include $(srcdir)/../makefile-common
CHESSCCWRAP2_FLAGS = aie2 -I ${AIETOOLS_DIR}/include
CHESSCCWRAP2P_FLAGS = aie2p -I ${AIETOOLS_DIR}/include
devicename ?= $(if $(filter 1,$(NPU2)),npu2,npu)

.PHONY: all clean test

all: build/final.xclbin build/insts.bin

build/aie.mlir: ${srcdir}/aie2.py
	mkdir -p ${@D}
	python3 $< ${devicename} > $@

build/fft64.o: ${srcdir}/fft64.cc ${srcdir}/fft64_twiddles.h
	mkdir -p ${@D}
ifeq ($(devicename),npu)
	cd ${@D} && xchesscc_wrapper ${CHESSCCWRAP2_FLAGS} -c $< -o ${@F}
else ifeq ($(devicename),npu2)
	cd ${@D} && xchesscc_wrapper ${CHESSCCWRAP2P_FLAGS} -c $< -o ${@F}
else
	$(error Unsupported device '${devicename}')
endif

build/final.xclbin: build/aie.mlir build/fft64.o
	cd ${@D} && $(AIECC) $(AIECC_FLAGS) \
		--xclbin-name=${@F} \
		--npu-insts-name=insts.bin $(<:%=../%)

build/insts.bin: build/final.xclbin
	@test -f $@

test: all
	python3 ${srcdir}/test.py

clean:
	rm -rf build
