/*
 * mels_utils.h
 * Copyright (C) 2019 xiaominfc(武汉鸣鸾信息科技有限公司)
 * Email: xiaominfc@gmail.com
 * Distributed under terms of the MIT license.
 */

#ifndef MELS_UTILS_H
#define MELS_UTILS_H

#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "matrix/matrix-lib.h"
#include "matrix/matrix-functions.h"
#include "fft.h"
using namespace kaldi;

#define MIN_LOG_HZ 1000.0
#define MIN_LOG_MEL 15.0
#define FSP 66.6666666667
#define MIN_F 0.0
#define LOG_STEP 0.06875177742094912
#define MAXFRE 8000.0


typedef unsigned int uint;


double hz_to_mel(double frequencies);
double mel_to_hz(double mels);
//mel 单元
void filters_mel(float sr,uint n_fft,uint n_mels,float fmin,float fmax, Matrix<double64> &mel_basis);


//hanning窗
void hanning_w(int frame_length,Vector<double64>& window);

//数据分帧
void frame_data(Vector<double64> &data,int frame_length,int hop_length, Matrix<double64> &frames);


//补数据 顺便把数据缩到[-1.0 , 1.0]之间
void pad_data(SubVector<BaseFloat> &data, int hop_length,Vector<double64>& outData);

void melspectrogram(SubVector<BaseFloat> &data,float sr,uint n_fft,uint hop_length,uint n_mels,Matrix<double64> &result);


#endif /* !MELS_UTILS_H */
