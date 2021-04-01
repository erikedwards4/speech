#@author Erik Edwards
#@date 2018-present
#@license BSD 3-clause

#speech is my own library of C functions for speech processing.

#This is the Makefile used during development of the C functions.
#Since these are not stand-alone programs (no main functions), the only outputs are object files to obj dir.
#It is not necessary to run this when making the C++ command-line tools.

SHELL=/bin/bash
CC=clang

ifeq ($(CC),$(filter $(CC),clang gcc))
	STD=-std=c11
else
	STD=-std=gnu++14
endif

ifeq ($(CC),clang++)
	STD=-std=c++11
	WFLAG=-Weverything -Wno-c++98-compat -Wno-padded -Wno-gnu-imaginary-constant -Wno-deprecated -Wno-old-style-cast
else
	WFLAG=-Wall -Wextra
endif

CFLAGS=$(WFLAG) $(STD) -O3 -march=native -fPIC


All: all
all: Dirs F0 SAD VAD Clean

Dirs:
	mkdir -pm 777 obj


#F0: pitch
Pre: rms_scale preemph dither
rms_scale: rms_scale.c; $(CC) -c $@.c -o obj/$@.o $(CFLAGS)
preemph: preemph.c; $(CC) -c $@.c -o obj/$@.o $(CFLAGS)
dither: dither.c; if [ $(CC) = "clang" ]; then gcc -c $@.c -o obj/$@.o $(CFLAGS); else $(CC) -c $@.c -o obj/$@.o $(CFLAGS); fi


#SAD: speech activity detection
SAD: sad_thresh
sad_thresh: sad_thresh.c; $(CC) -c $@.c -o obj/$@.o $(CFLAGS)


#VAD: voiced activity detection
VAD: vad_ccs
vad_ccs: vad_ccs.c; $(CC) -c $@.c -o obj/$@.o $(CFLAGS)


#If want to make a library archive for C++ use, this works with clang++
#ar crs ../lib/libaud.a obj/*.o


#make clean
Clean: clean
clean:
	find ./obj -type f -name *.o | xargs rm -f
	rm -f *.s tmp.*