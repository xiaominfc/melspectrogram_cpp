/*
 * mels_util.cc
 * Copyright (C) 2019 xiaominfc(武汉鸣鸾信息科技有限公司) <xiaominfc@gmail.com>
 *
 * Distributed under terms of the MIT license.
 */

#include "mels_utils.h"


double hz_to_mel(double frequencies){
	double mels = (frequencies - MIN_F) / FSP;
	if(frequencies >= MIN_LOG_HZ) {
	  mels = (MIN_LOG_HZ - MIN_F)/FSP + log(frequencies / MIN_LOG_HZ) / LOG_STEP;
	}
	return mels;
}

double mel_to_hz(double mels) {
	double freqs = MIN_F + FSP * mels;
	if(mels >= MIN_LOG_MEL) {
		freqs = MIN_LOG_HZ * exp(LOG_STEP * (mels - MIN_LOG_MEL));
	}
	return freqs;
}


//mel 单元
void filters_mel(float sr,uint n_fft,uint n_mels,float fmin,float fmax, Matrix<double64> &mel_basis) {
	if(fmax == 0.0) {
		fmax = sr / 2.0;
	}
	uint rows = 1+n_fft/2;
	mel_basis.Resize(n_mels,rows);
	// gen mel_frequencies
	double min_mel = hz_to_mel(fmin);
	double max_mel = hz_to_mel(fmax);
	Vector<double64> mels;
	mels.Resize(n_mels + 2);
	double offset_mel = (max_mel - min_mel)/(n_mels + 1);
	double offset_hz = sr/n_fft; 
	mels(0) = mel_to_hz(min_mel);
	mels(1) = mel_to_hz(min_mel + offset_mel);
	mels(mels.Dim() - 1) = mel_to_hz(max_mel);
	for(int i = 2; i < mels.Dim(); i ++) {
		double down_hz= mels(i - 2);
		double up_hz = mel_to_hz(min_mel + offset_mel*i);
		mels(i) = up_hz;
		for(int j = 0; j < rows; j++ ) {
			double lower = -1.0 * (down_hz - offset_hz*j) / (mels(i-1)-down_hz);
			double upper = (up_hz -offset_hz*j) / (up_hz - mels(i-1));
			double min  = lower < upper? lower:upper;
			min = min > 0?min:0.0;
      		mel_basis(i-2,j) = min*2.0/ (up_hz-down_hz);
		}
	}
}



//hanning窗
void hanning_w(int frame_length,Vector<double64>& window) {
	window.Resize(frame_length);
	double PI = 3.141592653589793;
	double PI_2 = PI * 2.0;
	double a = PI_2 / (frame_length);
	//printf("a:%.10lf\n",a);
	for (int32 i = 0; i < frame_length; i++) {
		double i_fl = static_cast<double>(i);
		window(i) = 0.5  - 0.5*cos(a * i_fl);
	}
}



//数据分帧
void frame_data(Vector<double64> &data,int frame_length,int hop_length, Matrix<double64> &frames){
	int n_frames = (data.Dim() - frame_length ) / hop_length + 1;
	frames.Resize(frame_length,n_frames);
	for(int i=0; i < frame_length; i ++) {
		for(int j = 0; j < n_frames; j ++) {
			int index = i + j * hop_length;
			if(index < data.Dim()) {
				frames(i,j) = data.Data()[index]; 
			}else {
				printf("index to large:%d\n",index);
			}
		}
	}
}



//补数据 顺便把数据缩到[-1.0 , 1.0]之间
void pad_data(SubVector<BaseFloat> &data, int hop_length,Vector<double64>& outData) {   
	size_t len = data.Dim();
	outData.Resize(len + hop_length*2);
	int end = len - hop_length - 2;
	for(int i = 0 ; i < len; i++) {
		double v = static_cast<double>(data.Data()[i])/32767.0;// convert to float  [-1.0 , 1.0] 
		outData.Data()[hop_length + i] = v;
		if(i > 0 && i <=hop_length) {
			outData.Data()[hop_length-i] = v;
		}
		if(i > end && i < len - 1) {
			int index = len * 2 - i - 2 + hop_length;
			outData.Data()[index] = v;
		} 
	}
}


void melspectrogram(SubVector<BaseFloat> &data,float sr,uint n_fft,uint hop_length,uint n_mels,Matrix<double64> &result){
	uint half_n_fft = n_fft / 2;
	Matrix<double64> mel_basis;
	filters_mel(sr,n_fft,n_mels,0.0,MAXFRE,mel_basis);
	Vector<double64> padData;
	pad_data(data,half_n_fft,padData);
	Matrix<double64> frames;
	frame_data(padData,n_fft , hop_length ,frames);
	Vector<double64> window;
	hanning_w(n_fft,window);
	frames.MulRowsVec(window);
	Matrix<double64> fft_result;
	fft_result.Resize(half_n_fft + 1,frames.NumCols());
	complex *pSignal = new complex[frames.NumRows()];
	for(int i = 0; i < frames.NumCols(); i ++) {
		for(int j = 0; j < frames.NumRows();j++) {
			pSignal[j] = frames(j,i);
		}
		CFFT::Forward(pSignal,n_fft);
		for(int j = 0; j < fft_result.NumRows(); j ++) {
			double v = sqrt(pSignal[j].re()*pSignal[j].re()+pSignal[j].im()*pSignal[j].im());
			fft_result(j,i) = pow(v,2.0);
		}
	}

	result.Resize(fft_result.NumCols(),mel_basis.NumRows());
	for(int i = 0; i < result.NumRows(); i ++) {
		for(int j = 0; j < result.NumCols(); j ++) {
			double sum = 0;
			for(int index = 0; index < half_n_fft + 1; index++ ){
				sum += fft_result(index,i)*mel_basis(j,index);
			}
			result(i,j) = sum;
		}
	}
}
