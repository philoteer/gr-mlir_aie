srcdir := $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))
include $(srcdir)/../makefile-common
aie_py_src=passthrough_pykernel.py
devicename ?= $(if $(filter 1,$(NPU2)),npu2,npu)

.PHONY: all template clean
.DEFAULT_GOAL := compile

all: build/final.xclbin

build/aie.mlir: ${srcdir}/${aie_py_src}
	mkdir -p ${@D}
	python3 $< ${devicename} > $@

build/final.xclbin: build/aie.mlir
	mkdir -p ${@D}
	cd ${@D} && $(AIECC) $(AIECC_FLAGS) \
		--xclbin-name=${@F} --npu-insts-name=insts.bin $(<:%=../%)

clean:
	rm -rf build _build

compile: clean build/final.xclbin build/insts.bin
