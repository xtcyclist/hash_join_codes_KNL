debug=-DNUMA_MAP -O3 -lmemkind -lnuma -DSCATTER #-I/usr/src/kernels/3.10.0-327.36.1.el7.x86_64/include/linux/ #-DBUFFER_SIZE=32 #-DMCDRAM
macro=-DPREFETCHING #-DTIMELOG #-qno-opt-prefetch #-DTIMELOG_DETAIL
include=-I/home/s/shuhao-z/install/include
lib=-L/home/s/shuhao-z/install/lib

all: cpra phj npj tpj

phj: phj.cpp
	icpc $(debug) phj.cpp -lpthread -std=c++0x -lrt -o phj $(macro) $(include) $(lib)

npj: npj.cpp
	icpc -m64 -xmic-avx512 npj.cpp -lpthread -std=c++0x -lrt -o npj $(debug) $(macro) $(include) $(lib)

clean:
	rm phj npj
