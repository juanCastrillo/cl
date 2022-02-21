import os
import sys
import torch
from avalanche.benchmarks.utils import ImageFolder
from torchvision import transforms
from torch.utils import data
import numpy as np
import gluoncv
from cvUtils import loadDataset, parallelModel, saveModel, trainModel
from gluon2pytorch import gluon2pytorch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models


'''
Dado un dataset, preentrenar un modelo con todas las clases
y guardarlo.
'''
def pretrain(dirrLoadDataset):
    """ LOAD DATA """
    dataset, trainSet, testSet = loadDataset(dirrLoadDataset, trainRatio=0.7)
    # Modelos de clasificación https://cv.gluon.ai/model_zoo/classification.html
    # classifModels = [ # He cogido los mejores a ojo, 1 de cada tipo https://kobiso.github.io/Computer-Vision-Leaderboard/imagenet.html
    #     "ResNet101_v1d", 
    #     "SE_ResNext101_32x4d",
    #     "ResNeSt269",
    #     "MobileNetV3_Large",
    #     "VGG19_bn",
    #     "SqueezeNet1.0",
    #     "DenseNet161",
    #     "resnet101_v1d_0.76", 
    #     "SENet_154"
    # ]
    # EfficientNet-B7 al 5, GPipe-AmoebaNet-B, Oct-ResNet-152+SE
    
    """ LOAD MODEL """
    # model = gluoncv.model_zoo.get_model("ResNet101_v1d", pretrained=False) # TODO
    # model.hybridize()
    # model.collect_params().initialize()
    # # Remove net output from console
    # old_stdout = sys.stdout # backup current stdout
    # sys.stdout = open(os.devnull, "w")
    # # Transform model
    # model: nn.Module = gluon2pytorch(model, [(100, 3, 256, 256)], dst_dir=None, pytorch_module_name='idk')
    # sys.stdout = old_stdout
    model = models.MobileNetV2(num_classes=67) #models.mobilenet_v2()
    # for param in model.parameters():
    #     param.requires_grad = False
    
    model = parallelModel(model)
    model.cuda()

    """ TRAIN MODEL """
    trainDataloader = data.DataLoader(trainSet, batch_size=100, num_workers=12, shuffle=True) # IMPORTANTE 12 workers, si no se quedan esperando las GPU
    testDataloader = data.DataLoader(testSet, batch_size=400, num_workers=12, shuffle=True)
    loss = nn.CrossEntropyLoss().cuda("cuda:0")
    # 0.0001
    optimizer = torch.optim.Adam(model.parameters(), lr=0.00005, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False) # optim.SGD(model.parameters(), lr=0.005, momentum=0.9)
    model = trainModel(model, 200, trainDataloader, testDataloader, optimizer, loss)
    print(model)
    return model
