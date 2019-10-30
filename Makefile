
all:


EXTRA_CXXFLAGS = -Wno-sign-compare
include ../kaldi.mk
OBJFILES = fft.o complex.o mels_utils.o
LIBNAME = ext_mels
BINFILES = mels_a_file
ADDLIBS = ../feat/kaldi-feat.a ../util/kaldi-util.a ../matrix/kaldi-matrix.a ../base/kaldi-base.a

include ../makefiles/default_rules.mk
