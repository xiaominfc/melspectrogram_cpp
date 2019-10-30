/*
 * mfcc_a_file.cc
 * Copyright (C) 2019 xiaominfc(武汉鸣鸾信息科技有限公司) <xiaominfc@gmail.com>
 *
 * Distributed under terms of the MIT license.
 */
#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "feat/wave-reader.h"
#include "feat/feature-spectrogram.h"
#include "feat/feature-fbank.h"
#include "matrix/matrix-lib.h"
#include "matrix/matrix-functions.h"
#include "fft.h"
#include "mels_utils.h"
using namespace kaldi;


int main(int argc, char* argv[]){
	try{
		using namespace kaldi;

		//test_hanning_w();
		const char *usage = "compute  mels feature for a wav file\n"
			"Usage: mels_a_file [options...] <wav-path>\n";
		ParseOptions po(usage);
		po.Read(argc, argv);
		if (po.NumArgs() != 1) {
			po.PrintUsage();
			exit(1);
		}
		size_t N = 1024;
		std::string wavpath = po.GetArg(1);
		std::ifstream is(wavpath, std::ios_base::binary);
		WaveData wave;
		wave.Read(is);
		SubVector<BaseFloat> waveform(wave.Data(), 0);
		float sr = wave.SampFreq();
    Matrix<double64> result;
		melspectrogram(waveform,sr,N,512,80,result);
	  
    //只打印第一行
		for(int i = 0; i < 80; i++) {
			printf("%0.10e\n",result(0,i));
		}


	}catch(const std::exception &e) {

	}
}

