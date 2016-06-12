TARGET	= main
CXXSRCS	= $(wildcard *.cpp)
CSRCS	= $(wildcard vl/*.c)
CHEADERS	= $(wildcard *.h)
CXXHEADERS = $(wildcard vl/*.h)
CXXDIR	= $(notdir $(CXXSRCS))
CDIR = $(notdir $(CSRCS))
CXXOBJECTS	= $(patsubst %.cpp,%.o, $(CXXDIR)) 
COBJECTS	= $(patsubst %.c,%.o, $(CDIR)) 

CUDA_ROOT = /usr/local/cuda
OPENCV_ROOT = ../opencv-2.4.9
ARMADILLO_ROOT = ./armadillo-4.650.2
PCA_ROOT = ./libpca-1.2.11

INC =-Ivl/
INC +=-I$(OPENCV_ROOT)
INC +=-I$(OPENCV_ROOT)/modules/core/include
#INC +=-I/usr/local/include 
#INC +=-I/usr/local/openssl/include/ -I/usr/lib/x86_64-linux-gnu/gcc/x86_64-linux-gnu/4.5/include 
INC +=-I$(ARMADILLO_ROOT)/include -I$(PCA_ROOT)/include 

LIB     = -lm -lrt -lpthread -ldl -pipe -lpca -L$(PCA_ROOT)/build -larmadillo -L$(ARMADILLO_ROOT) -std=c++0x -pthread
LIB     += -L$(CUDA_ROOT)/lib64/ -lcudart -lcufft 
LIB     += -lopencv_highgui -lopencv_calib3d -lopencv_contrib -lopencv_core -lopencv_features2d -lopencv_flann -fopenmp -lcublas 

NVCC	= $(CUDA_ROOT)/bin/nvcc -arch=sm_20 -m 64 $(INC)

CXX = g++ -pipe
CC = gcc -pipe
LINK	= -O3 -pthread 
CFLAGS	= -O3 -c -pthread -fopenmp -DVL_DISABLE_AVX
CXXFLAGS= -O3 -c -pthread -std=c++0x -fopenmp -DVL_DISABLE_AVX

gpu_fv: $(COBJECTS) $(CXXOBJECTS) cuda_files.o
	$(CXX) $(LINK) -o $@ $(COBJECTS) $(CXXOBJECTS) cuda_files.o $(LIB) 
#svm/tron.o svm/linear.o svm/blas.a 

cuda_files.o: cuda_files.cu 
	$(NVCC) -I$(CUDA_ROOT)/include/  -O3 -c cuda_files.cu -lcublas

$(COBJECTS): $(CSRCS) $(CHEADERS) Makefile
	$(CC) $(INC) $(CFLAGS) $(CSRCS)

$(CXXOBJECTS): $(CXXSRCS) $(CXXHEADERS) Makefile
	$(CXX) $(INC) $(CXXFLAGS) $(CXXSRCS)

clean:
	rm -f gpu_fv $(COBJECTS) $(CXXOBJECTS) cuda_files.o

.DEFAULT:
	@echo The target \"$@\" does not exist in Makefile.

