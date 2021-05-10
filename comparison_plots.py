import matplotlib.pyplot as plt
import math
import numpy as np
import pandas as pd
from scipy.io import wavfile
from scipy.fft import fftfreq,rfftfreq,rfft


path = "C:\\Users\\rkggp\\OneDrive\\Desktop\\Dataset\\"
f_name = "D_3.wav"
rate,audio=wavfile.read(path + (f_name))
N = audio.shape[0]
##----Distinction of low frequency and high frequency-----###
xf=rfftfreq(N,1/rate)
spectrum = rfft(audio)
spectrum = np.abs(spectrum)
plt.plot(xf,spectrum)
path = "C:\\Users\\rkggp\\OneDrive\\Desktop\\Dataset\\"
f_name = "E_1.wav"
rate,audio=wavfile.read(path + (f_name))
N = audio.shape[0]
##----Distinction of low frequency and high frequency-----###
xf=rfftfreq(N,1/rate)
spectrum = rfft(audio)
spectrum = np.abs(spectrum)
plt.plot(xf,spectrum)
plt.legend(['Class 2', 'Class 1'],prop={'size': 16})
plt.show()