# C/C++实现Python音频处理库librosa中melspectrogram的计算过程


## 编译

因为有矩阵计算图省事所以依赖[kaldi](https://github.com/kaldi-asr/kaldi)
先要准备好一个kaldi的编译好的环境

```

cd ${kaldi_project_path}/src
git clone https://github.com/xiaominfc/melspectrogram_cpp
cd melspectrogram_cpp

#为了兼容 任意长度的fft 所以依赖fftw

wget http://www.fftw.org/fftw-3.3.9.tar.gz
tar -xf fftw-3.3.9.tar.gz
cd fftw-3.3.9
./configure
make -j 4

cd ..
#编译 fftw结束

make -j 4

```




## 测试

```
python3 ./melspectrogram_util.py
# 看到输出

./mels_a_file ./6983123609037.wav
# 对比输出(cpp程序只输出了第一行)

```
