import matplotlib.pyplot as plt
import math
import numpy as np
import pandas as pd
from scipy.io import wavfile
from scipy.fft import fftfreq,rfftfreq,rfft
from scipy import fftpack
import scipy
import os
import time
from scipy import stats
import noisereduce as nr
import librosa


path = "C:\\Users\\rkggp\\OneDrive\\Desktop\\Dataset\\"
dir_list = os.listdir(path)
pearson1 = []
pearson2 = []
pearson3 = []
pearson4 = []
pearson5 = []
x = []
# Assuming the recorded data E_1.wav is noise free and the actual breathing sound. We will use this for noise reduction. 
audio_no_noise,rate=librosa.load("C:\\Users\\rkggp\\OneDrive\\Desktop\\Dataset\\E_1.wav")
N = audio_no_noise.shape[0]
xf_no_noise = rfftfreq(N,1/rate)
spectrum1 = rfft(audio_no_noise)
spectrum1 = np.abs(spectrum1)
spec1 = []
for i in range(7):
    spec1.append((spectrum1[i]))
for i in range(len(spectrum1)-7):
    spec1.append(sum(spectrum1[i:i+7])/7)

high1 = 0
spec11 = []
for i in range(len(xf_no_noise)):
        if xf_no_noise[i] >6000:
            high1= i
            break
for  i in range(len(xf_no_noise)):
    if i <=60:
        spec11.append(0)
    elif i>60 and i< high1:
        spec11.append(spec1[i])
    else:
        spec11.append(0) 

b = 0
for f_name in dir_list:
    b+=1
    audio,rate=librosa.load(path+ str(f_name))
    N = audio.shape[0]
    xf=rfftfreq(N,1/rate)
    L = N / rate
    spectrum3 = rfft(audio)
    spectrum3 = np.abs(spectrum3)
    spec3=[]
    for i in range(7):
        spec3.append((spectrum3[i]))
    for i in range(len(spectrum3)-7):
        spec3.append(sum(spectrum3[i:i+7])/7) 
    high3 = 0
    spec33 = []
    for i in range(len(xf)):
            if xf[i] >6000:
                high3 = i
                break
    for  i in range(len(xf)):
        if i <=60:
            spec33.append(0)
        elif i>60 and i< high3:
            spec33.append(spec3[i])
        else:
            spec33.append(0) 
    reduced_noise = nr.reduce_noise(audio_clip=audio, noise_clip=audio_no_noise, verbose=False)
    reduced_noise2 = nr.reduce_noise(audio_clip=audio, noise_clip=reduced_noise, verbose=False)
    spectrum = rfft(reduced_noise2)
    spectrum = np.abs(spectrum)
    spec2=[]
    for i in range(7):
        spec2.append((spectrum[i]))
    for i in range(len(spectrum)-7):
        spec2.append(sum(spectrum[i:i+7])/7)
    high2 = 0
    spec22 = []
    for i in range(len(xf)):
            if xf[i] >6000:
                high2= i
                break
    for  i in range(len(xf)):
        if i <=60:
            spec22.append(0)
        elif i>60 and i< high2:
            spec22.append(spec2[i])
        else:
            spec22.append(0) 
    
    m = min(len(spectrum1),len(spectrum3))
    pearson1.append((stats.pearsonr(spectrum1[:m], spectrum3[:m]))[0])
    pearson2.append((stats.pearsonr(spec1[:m], spec3[:m]))[0])
    pearson3.append((stats.pearsonr(spec11[:m], spec33[:m]))[0])
    pearson4.append((stats.pearsonr(spec1[:m], spec2[:m]))[0])
    pearson5.append((stats.pearsonr(spec11[:m], spec22[:m]))[0])
    x.append(b)
    print(b)

plt.plot(x,pearson1)
plt.xlabel("Sample Number")
plt.ylabel("For Raw Samples")
plt.show()
print(sum(pearson1)/len(pearson1))

plt.plot(x,pearson2)
plt.show()
plt.xlabel("Sample Number")
plt.ylabel("For Moving Averages")
print(sum(pearson2)/len(pearson2))

plt.plot(x,pearson3)
plt.show()
plt.xlabel("Sample Number")
plt.ylabel("For Moving Average + HPF + LPF ")
print(sum(pearson3)/len(pearson3))

plt.plot(x,pearson4)
plt.show()
plt.xlabel("Sample Number")
plt.ylabel("For Noise Reduce + MOving Average")
print(sum(pearson4)/len(pearson4))

plt.plot(x,pearson5)
plt.show()
plt.xlabel("Sample Number")
plt.ylabel("For Noise Reduce + Moving Average + HPF + LPF")
print(sum(pearson5)/len(pearson5))






