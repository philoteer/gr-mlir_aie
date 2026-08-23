##===- Makefile -----------------------------------------------------------===##
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
##===----------------------------------------------------------------------===##
.DEFAULT_GOAL := compile

srcdir := $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))

export LD_LIBRARY_PATH := /opt/vitis_aie_essentials/lib/lnx64.o/:$(LD_LIBRARY_PATH)
export RDI_DATADIR := /opt/vitis_aie_essentials/data
export PATH := /opt/vitis_aie_essentials/tps/lnx64/target_aie_ml/bin/LNa64bin/:$(PATH)

AIETOOLS_DIR = /opt/vitis_aie_essentials/
CHESSCCWRAP2_FLAGS = aie2 -I ${AIETOOLS_DIR}/include
CHESSCCWRAP2P_FLAGS = aie2p -I ${AIETOOLS_DIR}/include

targetname = cbfloat16ToCint16
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
	cd ${@D} && aiecc.py --aie-generate-xclbin --no-compile-host --xclbin-name=${@F} \
			--aie-generate-npu-insts --npu-insts-name=insts.bin $(<:%=../%)

# aiecc.py emits insts.bin as a side effect of the xclbin target.
build/insts.bin: build/final.xclbin
	@:

clean:
	rm -rf build _build

compile: clean build/final.xclbin build/insts.bin
