from imghdr import tests
import os
import sys
from cvUtils import loadDataset, loadModel, parallelModel, saveModel, testAccuracy
from pretrain import pretrain
from torch.utils.data import DataLoader
import torch.nn as nn
import gluoncv
from gluon2pytorch import gluon2pytorch

"""
Hacer un main que llame a pretrain
"""
dsdirr = "/home/calculon/aprendiz/datasets/indoorCVPR"
mdirr = "/home/calculon/aprendiz/models/testFullDS.pth"
model = pretrain(dsdirr)
saveModel(model, mdirr)

"""
Cargar modelo y comparar valores de rendimiento
"""
dataset, trainSet, testSet = loadDataset(dsdirr)
model = loadModel(mdirr)
print(model)
tloss, acc = testAccuracy(model, DataLoader(testSet, batch_size=100, num_workers=12), nn.CrossEntropyLoss().cuda("cuda:0"))
print(tloss, acc)