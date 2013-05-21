CC = gcc

SOURCES = qvm.c gates.c qureg_dense.c cfloat.c

TARGETS = qvm

VPATH = sexp/lib
INCPATH = -I./sexp/include -I /usr/local/cuda-5.0/include -I./
LIBPATH = -L./sexp/lib
LIBS = -lsexp -lm -lOpenCL -lpapi
OFLAGS = #-O2 -Wall #-O2
DFLAGS = -g
CFLAGS = $(OFLAGS) $(DFLAGS) $(INCPATH) $(LIBPATH) -std=c99

DEST_OBJS=$(SOURCES:.c=.o)

all:  qvm

qvm: $(DEST_OBJS)
	$(CC) $(CFLAGS) -o $@ $(DEST_OBJS) $(LIBS)

%.o: %.c %.h
	$(CC) $(CFLAGS) -c -o $@ $<

clean:
	rm -f $(TARGETS) $(DEST_OBJS)
