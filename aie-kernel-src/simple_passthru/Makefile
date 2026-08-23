srcdir := $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))
aie_py_src=passthrough_pykernel.py

.PHONY: all template clean
.DEFAULT_GOAL := compile

all: build/final.xclbin

build/aie.mlir: ${srcdir}/${aie_py_src}
	mkdir -p ${@D}
	python3 $<  > $@

build/final.xclbin: build/aie.mlir
	mkdir -p ${@D}
	cd ${@D} && aiecc.py --aie-generate-xclbin --aie-generate-npu-insts --no-compile-host \
		--no-xchesscc --no-xbridge \
		--xclbin-name=${@F} --npu-insts-name=insts.bin $(<:%=../%)

clean:
	rm -rf build _build

compile: clean build/final.xclbin build/insts.bin

