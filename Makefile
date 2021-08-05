#@author Erik Edwards
#@date 2018-present
#@license BSD 3-clause

#speech is my own library of functions for speech processing in C and C++.
#This is the makefile for the C++ command-line tools.

#For general-purpose audio, including music, etc., see the aud repo.
#Higher-level functions meant only for speech and voice are here (speech).

SHELL=/bin/bash
ss=../util/bin/srci2src
CC=g++-7

ifeq ($(CC),clang++)
	STD=-std=c++11
	WFLAG=-Weverything -Wno-c++98-compat -Wno-old-style-cast -Wno-gnu-imaginary-constant
else
	STD=-std=gnu++14
	WFLAG=-Wall -Wextra
endif

INCLS=-Ic -I../util
CFLAGS=$(WFLAG) $(STD) -O2 -ffast-math -march=native $(INCLS)
KINCLS=-I../util -I/opt/kaldi/src -I/opt/kaldi/src/bin -I/opt/kaldi/src/base -I/opt/kaldi/src/matrix -I/opt/kaldi/src/util -I/opt/kaldi/src/makefiles -I/opt/kaldi/src/feat -I/opt/kaldi/src/featbin -I/opt/kaldi/src/transform
KFLAGS=-DKALDI_DOUBLEPRECISION=0

All: all
all: Dirs Kaldi Kaldi_compat F0 SAD VAD Clean
	rm -f 7 obj/*.o

Dirs:
	mkdir -pm 777 bin obj


#Use Kaldi C++ code directly (but usual commmand-line interface)
Kaldi: kaldi.spectrogram
kaldi.spectrogram: src/kaldi.spectrogram.cpp
	$(CC) -c src/$@.cpp -oobj/$@.o $(KINCLS); $(CC) obj/$@.o -obin/$@ -rdynamic -largtable2 -lopenblas -lm -lpthread -ldl


#My own re-implementation of Kaldi feats
Kaldi_compat: kaldi_spectrogram kaldi_fbank kaldi_mfcc #kaldi_plp kaldi_pitch
kaldi_spectrogram: srci/kaldi_spectrogram.cpp c/kaldi_spectrogram.c
	$(ss) -vd srci/$@.cpp > src/$@.cpp; $(CC) -c src/$@.cpp -oobj/$@.o $(CFLAGS); $(CC) obj/$@.o -obin/$@ -largtable2 -lfftw3f -lfftw3 -lm
kaldi_fbank: srci/kaldi_fbank.cpp c/kaldi_fbank.c
	$(ss) -vd srci/$@.cpp > src/$@.cpp; $(CC) -c src/$@.cpp -oobj/$@.o $(CFLAGS); $(CC) obj/$@.o -obin/$@ -largtable2 -lfftw3f -lfftw3 -lm
kaldi_mfcc: srci/kaldi_mfcc.cpp c/kaldi_mfcc.c
	$(ss) -vd srci/$@.cpp > src/$@.cpp; $(CC) -c src/$@.cpp -oobj/$@.o $(CFLAGS); $(CC) obj/$@.o -obin/$@ -largtable2 -lfftw3f -lfftw3 -lm
kaldi_plp: srci/kaldi_plp.cpp c/kaldi_plp.c
	$(ss) -vd srci/$@.cpp > src/$@.cpp; $(CC) -c src/$@.cpp -oobj/$@.o $(CFLAGS); $(CC) obj/$@.o -obin/$@ -largtable2 -lfftw3f -lfftw3 -lm


F0: #f0_ccs
f0_ccs: srci/f0_ccs.cpp c/f0_ccs.c
	$(ss) -vd srci/$@.cpp > src/$@.cpp; $(CC) -c src/$@.cpp -oobj/$@.o $(CFLAGS); $(CC) obj/$@.o -obin/$@ -largtable2


SAD: #sad_thresh
sad_thresh: srci/sad_thresh.cpp c/sad_thresh.c
	$(ss) -vd srci/$@.cpp > src/$@.cpp; $(CC) -c src/$@.cpp -oobj/$@.o $(CFLAGS); $(CC) obj/$@.o -obin/$@ -largtable2


VAD: #vad_ccs
vad_ccs: srci/vad_ccs.cpp c/vad_ccs.c
	$(ss) -vd srci/$@.cpp > src/$@.cpp; $(CC) -c src/$@.cpp -oobj/$@.o $(CFLAGS); $(CC) obj/$@.o -obin/$@ -largtable2


#make clean
Clean: clean
clean:
	find ./obj -type f -name *.o | xargs rm -f
	rm -f 7 X* Y* x* y*
