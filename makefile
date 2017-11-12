HOST := $(shell hostname)
SHELL = /bin/bash
CC = mpicc
CFLAGS = -fPIC -Wall -Wextra -O3 -msse -msse2 -msse3 -mtune=native -march=native -ftree-vectorizer-verbose=7
RM = rm -vf
NAME = LLR
LIBS = -lm -lgfortran
BLAS = $(HOME)/OpenBLAS/$(HOST)/lib/libopenblas.a
VERSION = 0.1

SRCS = _LLR_prox_mpi_complex_acc.c _LLR_prox_complex.c _LLR_prox_mpi_complex.c mat_complex.c  mat.c  param.c
OBJS = $(SRCS:.c=.o)
TARGET = lib$(NAME).so

.PHONY : clean

test: $(TARGET)
	$(CC) -I. -L. -g -o $@ test.c -l$(NAME)

$(TARGET) : $(OBJS)
	$(CC) -shared -Wl,-soname,$@ -o $@ $^ $(BLAS) $(LIBS)

$(SRCS:.c=.d):%.d:%.c
	$(CC) $(CFLAGS) -MM $< >$@

include $(SRCS:.c=.d)

clean :
	-$(RM) lib$(NAME).so $(OBJS) $(SRCS:.c=.d) test
