#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 10 06:26:16 2021

@author: jan sima
"""

#import torch
import matplotlib
import onnx
import scipy.signal as signal
import scipy.stats as stats
import numpy as np
import onnxruntime as rt
import tansorflow as tf

#model = torch.load('/media/sf_NoiseDetectionCNN-python/model')

## ------------------------------------------------------------------
from pymef.mef_session import MefSession

# nacteni
text = '/media/sf_vb_shared/Easrec-1310090913.mefd'
ms = MefSession(text, 'bemena')
dr = ms.read_ts_channel_basic_info()

channel_names = [x['name'] for x in dr]
# data 3s
data = ms.read_ts_channels_sample(channel_names[5], [0,15000]) 
#data = ms.read_ts_channels_sample(["A'1"], [100,200])
#matplotlib.pyplot.plot(data)

# Preproces dat spectrogram
_,_, data = signal.spectrogram(data,fs=5000,nperseg=256,noverlap=128,nfft=1024)
data = data[:200,:]
data = stats.zscore(data,axis=1)
data = np.expand_dims(data,axis=0)

# data convecrt array to tensor
tf.convert_to_tensor(data, dtype=None, dtype_hint=None, name=None)

sess = rt.InferenceSession('Model_Noise_IEEG.onnx')

out = sess.run(None, {'input': data})


# Model
model = onnx.load('Model_Noise_IEEG.onnx')
# out = model(data) 

onnx.checker.check_model(model)

