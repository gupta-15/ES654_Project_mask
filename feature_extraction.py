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

path = "C:\\Users\\rkggp\\OneDrive\\Desktop\\Dataset\\"
dir_list = os.listdir(path)
F1 = []
F2 = []
F3 = []
F4 = []
F5 = []
F6 = []
F7 = []
F8 = []
F9 = []
F10 = []
F11 = []
F12 = []

for f_name in dir_list:
    print(f_name)
    rate,audio=wavfile.read(path + str(f_name))
    N = audio.shape[0]
    ##----Distinction of low frequency and high frequency-----###
    xf=rfftfreq(N,1/rate)
    for i in range(len(xf)):
        if xf[i] >7000:
            index = i
            break

    L = N / rate

    #print(str(L) +" seconds")
    ##----FFT and separation in low spectrum and high spectrum-----###
    spectrum = rfft(audio)
    spectrum = np.abs(spectrum)
    low_spec = spectrum[:index]
    high_spec = spectrum[index:]

    ######__Feature-1___#######
    #######----Sum power of low frequencies-----###
    p_low = np.sum(np.square(np.array(low_spec)))
    F1.append(p_low)
    #print(p_low)


    ######__Feature-2___#######
    #######----Sum power of high frequencies-----###
    p_high = np.sum(np.square(np.array(high_spec)))
    F2.append(p_high)
    #print(p_high)


    ######__Feature-3___#######
    #######----Ration of the high vs low-----###
    p_ratio = p_low/p_high
    F3.append(p_ratio)
    #print(p_ratio)


    ######__Feature-4,5,6,7___#######
    ###----3 degree polynomial fit to a fft-----###
    coef_3D_poly = np.polyfit(xf, spectrum, 3)
    d,c,b,a = coef_3D_poly
    #y_3D = d + c*xf + b*xf*2 + a*xf*3
    #plt.plot(xf,y_3D)
    #print(a,b,c,d)
    # plt.show()
    F4.append(a)
    F5.append(b)
    F6.append(c)
    F7.append(d)


    ######__Feature-8,9___#######
    ###----linear degree polynomial fit to a fft-----###
    coef_linear_poly = np.polyfit(xf, spectrum, 1)
    x1, x0 = coef_linear_poly
    F8.append(x0)
    F9.append(x1)


    ######__Feature-10___#######
    ######---Ratio of maimum to next 9 peaks-----####
    new = list(spectrum.copy())
    tmp = []
    for i in range(10):
        temp = max(new)
        new.remove(temp)
        tmp.append(temp)
    peak_ratio = tmp[0]/(sum(tmp[1:])*len(tmp[1:]))
    #print(peak_ratio)
    F10.append(peak_ratio)

    
    ######__Feature-11___#######
    ######---Standard Deviation-----####
    std = np.std(spectrum)
    #print(std)
    F11.append(std)


    ######__Feature-12___#######
    #####----Area-----#####
    area  = 0
    spec = list(spectrum)
    x = list(xf)
    #print(len(spec))
    #print(len(x))
    for i in range(len(spec)-1):
        area += abs((spec[i+1]-spec[i])*(x[i+1]-x[i]))
    #print(area)
    F12.append(area)
data = {"file name":dir_list, "1":F1, "2":F2, "3":F3, "4":F4, "5":F5, "6":F6, "7":F7, "8":F8, "9":F9, "10":F10, "11":F11, "12":F12}
df_dataset = pd.DataFrame(data)

df_dataset.to_csv('project_dataset_mask_final.csv')
