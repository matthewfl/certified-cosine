TARGET=certified_cosine_benchmark
SHARED_TARGET=libcertified_cosine.so
MAIN=certified_cosine_benchmark_main.cc
SOURCE=pre_processing.cc svd_wrapper.cc vector_stats.cc

EIGEN=../eigen/

UNIT_TESTS= $(wildcard test_*.cc)

GIT_VERSION := $(shell git describe --always --long --dirty --abbrev=12 ; date)

CXX:=g++

FLAGS=-I$(EIGEN) -fopenmp -DGIT_VERSION="\"$(GIT_VERSION)\"" -std=c++17 -fno-semantic-interposition -fpic -fconcepts
LIBS=
LDFLAGS=-Wl,-Bsymbolic

MARCH ?= native

MODE ?= optimized

ifeq ($(MODE),optimized-debug)
FLAGS += -O3 -ggdb -march=$(MARCH) -DCERTIFIEDCOSINE_USE_PARALLEL -funroll-loops -fkeep-inline-functions
else ifeq ($(MODE),optimized)
FLAGS += -O3 -march=$(MARCH) -funroll-loops -fno-stack-protector -DNDEBUG -DCERTIFIEDCOSINE_USE_PARALLEL
else # debug
FLAGS += -O0 -ggdb -fkeep-inline-functions
endif

ifeq ($(PROFILE),1)
FLAGS += -pg
endif

$(info $(MODE))
$(info $(FLAGS))


.PHONY: debug all clean unit shared run depend python format


all: $(TARGET)

optimized:
	$(MAKE) MODE=optimized all

debug:
	$(MAKE) MODE=debug all

clean:
	rm -f $(TARGET) $(SHARED_TARGET) unit_tests *.o

# static libgcc so that I can compile locally and run it remotely
$(TARGET): $(MAIN:.cc=.o) $(SOURCE:.cc=.o)
	$(CXX) -o $@ $^ $(LDFLAGS) $(FLAGS) $(LIBS)

shared: $(SHARED_TARGET)

$(SHARED_TARGET): $(SOURCE:.cc=.o)
	$(CXX) -shared -Wl,-Bsymbolic -o $@ $^  $(FLAGS) $(LIBS)

run: all
	./$(TARGET)

unit_tests: $(UNIT_TESTS:.cc=.o) $(SOURCE:.cc=.o)
	$(CXX) $(FLAGS) -ggdb -o unit_tests $(UNIT_TESTS:.cc=.o) pre_processing.o

test_%.o: test_%.cc
	$(CXX) $(FLAGS) -ggdb -c $< -o $@

unit: unit_tests
	./unit_tests


# set the optimized flags on the preprocessing code, as do not care about debugging this
pre_processing.o: FLAGS := $(filter-out -O0 -Og -O3, $(FLAGS)) -O3
svd_wrapper.o: FLAGS := $(filter-out -O0 -Og -O3 -g -ggdb, $(FLAGS)) -O3


%.o: %.cc
	$(CXX) $(FLAGS) -c $< -o $@


# depedency handling for files
# g++ isn't getting the different files seperated when used as a single command???
depend:
	rm -f Makefile.deps
	for i in $(SOURCE) $(UNIT_TESTS) ; do \
		$(CXX) -MM -MF Makefile.deps_o $(FLAGS) $$i ; \
		cat Makefile.deps_o >> Makefile.deps ; \
		rm Makefile.deps_o ; \
	done

-include Makefile.deps


python:
	rm -rf build/
	rm -rf *cpython*.so
	python setup.py develop


format:
	clang-format -i $(shell find . -name '*.cc' -or -name '*.hpp' -not -name catch.hpp)
