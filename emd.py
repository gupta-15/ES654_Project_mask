import numpy as np
from PyEMD import EMD
import librosa
import os
from scipy import stats
import matplotlib.pyplot as plt

audio1,rate1=librosa.load("C:\\Users\\rkggp\\OneDrive\\Desktop\\Dataset\\E_1.wav")
path = "C:\\Users\\rkggp\\OneDrive\\Desktop\\Dataset\\"
aud1 = []
for i in range(7):
    aud1.append((audio1[i]))
for i in range(len(audio1)-7):
    aud1.append(sum(audio1[i:i+7])/7)
dir_list = os.listdir(path)
result = []
result2 = []
x= []
b = 0
for f_name in dir_list:
    b+=1
    audio2,rate2=librosa.load(path + str(f_name))
    emd = EMD()
    rate = min(rate1,rate2)
    aud2 = []
    for i in range(7):
        aud2.append((audio2[i]))
    for i in range(len(audio2)-7):
        aud2.append(sum(audio2[i:i+7])/7)
    aud2 = np.array(aud2)
    aud1 = np.array(aud1)
    IMFs1 = emd(audio1[:rate//2])
    IMFs2 = emd(audio2[:rate//2])
    IMFs3 = emd(aud1[:rate//2])
    IMFs4 = emd(aud2[:rate//2])

    pearson  = []
    for i in range(IMFs1.shape[0]):
        for j in range(IMFs2.shape[0]):
            pearson.append((stats.pearsonr(IMFs1[i], IMFs2[j]))[0])
    result.append(max(pearson))

    pearson2  = []
    for i in range(IMFs3.shape[0]):
        for j in range(IMFs4.shape[0]):
            pearson2.append((stats.pearsonr(IMFs3[i], IMFs4[j]))[0])
    result2.append(max(pearson2))
    x.append(b)
    print(b)
plt.plot(x,result)
plt.xlabel("Sample Number")
plt.ylabel("Max Pearson Coefficient")
plt.show()
print(sum(result)/len(result))

plt.plot(x,result2)
plt.xlabel("Sample Number")
plt.ylabel("Max Pearson Coefficient")
plt.show()
print(sum(result2)/len(result2))


