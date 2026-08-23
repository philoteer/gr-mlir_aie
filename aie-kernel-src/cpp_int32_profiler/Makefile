# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

.DEFAULT_GOAL := all

srcdir := $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))
export LD_LIBRARY_PATH := /opt/vitis_aie_essentials/lib/lnx64.o/:$(LD_LIBRARY_PATH)
export RDI_DATADIR := /opt/vitis_aie_essentials/data
export PATH := /opt/vitis_aie_essentials/tps/lnx64/target_aie_ml/bin/LNa64bin/:$(PATH)

AIETOOLS_DIR ?= /opt/vitis_aie_essentials
AIE_INCLUDE_DIR ?= ${AIETOOLS_DIR}/data/versal_prod/lib
AIE2_INCLUDE_DIR ?= ${AIETOOLS_DIR}/data/aie_ml/lib
CHESSCCWRAP2_FLAGS = aie2 -I ${AIETOOLS_DIR}/include
CHESSCCWRAP2P_FLAGS = aie2p -I ${AIETOOLS_DIR}/include

devicename ?= $(if $(filter 1,$(NPU2)),npu2,npu)

.PHONY: all clean test profile

all: build/final.xclbin build/insts.bin

build/aie.mlir: ${srcdir}/aie2.py
	mkdir -p ${@D}
	python3 $< ${devicename} > $@

build/profile_kernel.o: ${srcdir}/profile_kernel.cc
	mkdir -p ${@D}
ifeq ($(devicename),npu)
	cd ${@D} && xchesscc_wrapper ${CHESSCCWRAP2_FLAGS} -c $< -o ${@F}
else ifeq ($(devicename),npu2)
	cd ${@D} && xchesscc_wrapper ${CHESSCCWRAP2P_FLAGS} -c $< -o ${@F}
else
	$(error Unsupported device '${devicename}')
endif

build/final.xclbin: build/aie.mlir build/profile_kernel.o
	cd ${@D} && aiecc.py --aie-generate-xclbin --no-compile-host \
		--xclbin-name=${@F} \
		--aie-generate-npu-insts --npu-insts-name=insts.bin $(<:%=../%)

build/insts.bin: build/final.xclbin
	@test -f $@

test: all
	python3 ${srcdir}/test.py

profile: all
	python3 ${srcdir}/profile.py

clean:
	rm -rf build
