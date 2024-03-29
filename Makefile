.PHONY: lib, pybind, clean, format, all

all: lib


lib:
	@mkdir -p build
	@cd build; cmake ..
	@cd build; $(MAKE)

format:
	python3 -m black .
	clang-format -i src/*.cc src/*.cu

clean:
	rm -rf build SparseNDArray/sparse_ndarray_backend*.so
