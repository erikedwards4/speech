#@author Erik Edwards
#@date 2018-present
#@license BSD 3-clause

#speech is my own library of functions for speech processing in C and C++.
#This is the makefile for the C++ command-line tools.

#For general-purpose audio, including music, etc., see the aud repo.
#Higher-level functions meant only for speech and voice are here (speech).

SHELL=/bin/bash
ss=../util/bin/srci2src
CC=clang++

ifeq ($(CC),clang++)
	STD=-std=c++11
	WFLAG=-Weverything -Wno-c++98-compat -Wno-old-style-cast -Wno-gnu-imaginary-constant
else
	STD=-std=gnu++14
	WFLAG=-Wall -Wextra
endif

INCLS=-Ic -I../util
CFLAGS=$(WFLAG) $(STD) -O2 -ffast-math -march=native $(INCLS)


All: all
all: Dirs F0 SAD VAD Clean
	rm -f 7 obj/*.o

Dirs:
	mkdir -pm 777 bin obj


F0: f0_ccs
f0_ccs: srci/f0_ccs.cpp c/f0_ccs.c
	$(ss) -vd srci/$@.cpp > src/$@.cpp; $(CC) -c src/$@.cpp -oobj/$@.o $(CFLAGS); $(CC) obj/$@.o -obin/$@ -largtable2


SAD: sad_thresh
sad_thresh: srci/sad_thresh.cpp c/sad_thresh.c
	$(ss) -vd srci/$@.cpp > src/$@.cpp; $(CC) -c src/$@.cpp -oobj/$@.o $(CFLAGS); $(CC) obj/$@.o -obin/$@ -largtable2


VAD: vad_ccs
vad_ccs: srci/vad_ccs.cpp c/vad_ccs.c
	$(ss) -vd srci/$@.cpp > src/$@.cpp; $(CC) -c src/$@.cpp -oobj/$@.o $(CFLAGS); $(CC) obj/$@.o -obin/$@ -largtable2


#make clean
Clean: clean
clean:
	find ./obj -type f -name *.o | xargs rm -f
	rm -f 7 X* Y* x* y*
