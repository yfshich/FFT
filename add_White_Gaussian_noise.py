# -*- coding: utf-8 -*-
"""
Created on Mon Mar  5 13:41:09 2018

"""

'''
SNR = [-4,-2,0,2,4,6,8,10](dB)
SNR = 10*log10( sum(x**2) / sum(n**2))
程序中用hist()检查噪声是否是高斯分布，psd()检查功率谱密度是否为常数。
'''
import numpy as np
import pylab as plt

def wgn(x, snr):
    snr = 10**(snr/10.0)
    xpower = np.sum(x**2)/len(x)
    npower = xpower / snr
    return np.random.randn(len(x)) * np.sqrt(npower)

t = np.arange(0, 1000000) * 0.1
x = np.sin(t)
snr = 10
n = wgn(x, snr)
xn = x+n # 增加了6dBz信噪比噪声的信号

plt.figure(figsize=(6,8))
plt.subplot(411)
plt.title('Gauss Distribution')
plt.hist(n, bins=100, normed=True)
plt.subplot(412)
plt.psd(n)
plt.subplot(413)
plt.plot(t[0:100],x[0:100])
plt.title('The Original Sin Signal')
plt.subplot(414)
plt.plot(t[0:100],xn[0:100])
plt.title('The Noisy Sin Signal')
plt.show()
plt.tight_layout()
