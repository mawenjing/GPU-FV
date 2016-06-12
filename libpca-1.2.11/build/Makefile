PROG = libpca.so

UNAME := $(shell uname)

ifeq ($(UNAME), Darwin)
CXX = clang++ -stdlib=libc++
else
CXX = g++
endif

FLAGS = -O2 -Wall -std=c++0x -pthread -shared -fPIC

INCS = -I"../include" -I../../armadillo-4.650.2/include
LIBS = -larmadillo -L../../armadillo-4.650.2/ -L/usr/lib64
SRCS = ../src/*.cpp

RM = rm -f

all :
	$(CXX) $(FLAGS) $(INCS) $(SRCS) $(LIBS) -o $(PROG)

clean :
	$(RM) $(PROG)
