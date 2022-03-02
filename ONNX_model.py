#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 15 19:15:18 2022

@author: jan
"""

import onnx
import onnxruntime as rt
import netron



sess = rt.InferenceSession('Model_Noise_IEEG.onnx')

out = sess.run(None, {'input': x.numpy()})


model = onnx.load('Model_Noise_IEEG.onnx')

print(model.graph)
onnx.checker.check_model(model)
n = netron('Model_Noise_IEEG.onnx')