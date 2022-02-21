from datetime import datetime
import os
import sys
import torch
from avalanche.benchmarks.utils import ImageFolder
from torchvision import transforms
from torch.utils import data
from torch import save
import torch.nn as nn
from torch import load
import numpy as np
from torch.autograd import Variable
from gluon2pytorch import gluon2pytorch
from torch.utils.tensorboard import SummaryWriter

"""
UTILIDADES de computer vision o ML en general (pytorch)
"""


"""
Given a directory, load the dataset
and split it into train and test
    dir: Directorio en el que está el dataset
    trainRatio: Ratio del dataset para training >= 0 & <= 1; 
"""
def loadDataset(dirr, trainRatio=0.6, imageRes=None):
    if trainRatio < 0: trainRatio = 0
    elif trainRatio > 1: trainRatio = 1

    """ LOAD DATA """
    dataset = ImageFolder(dirr, 
        transform=transforms.Compose([
            transforms.Resize((256,256)),
            transforms.ToTensor()
        ])
    )
    labels = dataset.classes

    lengths = [int(np.ceil(0.7*len(dataset))),
            int(np.floor(0.3*len(dataset)))]
    
    diff = len(dataset)-sum(lengths)
    lengths[0] += diff
    trainSet, testSet = data.random_split(dataset, lengths)

    return dataset, trainSet, testSet


"""
MxNet model to pytorch
    inputSize: numBatch, canales, res.  ej: [(120, 3, 256, 256)]
"""
def convertModel(model, inputSize):
    # Remove net output from console
    old_stdout = sys.stdout # backup current stdout
    sys.stdout = open(os.devnull, "w")
    model.hybridize()
    model.collect_params().initialize()
    model: nn.Module = gluon2pytorch(model, inputSize, dst_dir=None, pytorch_module_name='idk')
    sys.stdout = old_stdout
    return model


"""
Preparar el modelo para procesado multiGPU
    model: Modelo de Pytorch
{returns modelo paralelizado}
"""
def parallelModel(model):
    model = nn.DataParallel(model)
    gpu = "cuda:0"
    torch.cuda.set_device(gpu)
    model.cuda(gpu)
    return model


"""
Guarda el modelo en el directorio indicado
    model: Modelo pytorch a guardar
    dirr: Directorio donde guardar el modelo
"""
def saveModel(model, dirr):
    torch.save(model, dirr)


"""
Carga el modelo guardado en el directorio indicado
"""
def loadModel(dirr):
    model = torch.load(dirr)
    return model


"""
Metodo de entrenamiento del modelo (optimizado para multiGPU)
    model: Modelo a entrenar
    epochs: Entero con el numero de epocas a entrenar
    trainLoader: Cargador de datos de entrenamiento
    testLoader: Cargador de datos de test
    optimizer: Optimizador a aplicar al modelo
    lossFunction: Funcion de coste o perdida a utilizar
    plotDensity: Number of points to plot each epoch

returns {modelo con la mejor precisión}
"""
def trainModel(model, epochs: int, trainLoader, testLoader, optimizer, lossFunction, plotDensity = 10):
    print(f"Training: {epochs} epochs")
    print(f"-----------------------")
    now = datetime.now()
    writer = SummaryWriter(f'runs/{now}')
    model.train()
    best_accuracy = 0
    nExamples = len(trainLoader) # Arreglar, el i va de 1 en 1 y este va en bloque de imagenes o algo asi TODO
    jPlot = nExamples//plotDensity
    print(jPlot)
    for epoch in range(epochs):  # loop over the dataset multiple times
        running_loss = 0.0
        running_acc = 0.0
        print(f"Epoch {epoch}")
       
        for i, (images, labels) in enumerate(trainLoader, 0):
            
            #print(f"Batch {i}")
            # get the inputs
            images = images.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)

            # zero the parameter gradients
            optimizer.zero_grad()
            # predict classes using images from the training set
            outputs = model(images)
            # compute the loss based on model output and real labels
            loss = lossFunction(outputs, labels)
            # backpropagate the loss
            loss.backward()
            # adjust parameters based on the calculated gradients
            optimizer.step()
            
            # Let's print statistics for every X images
            _, predicted = torch.max(outputs.data, 1)
            running_loss += loss.item() # extract the loss value
            running_acc += predicted.eq(labels).sum()
            # print(f"Loss: {loss.item()}")
            if i % jPlot == 0 and i != 0:
                # print every 1000 (twice per epoch) 
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / jPlot))
                # ...log the running loss
                writer.add_scalar('Training loss',
                                running_loss / jPlot,
                                epoch * len(trainLoader) + i)

                writer.add_scalar('Training Accuracy',
                                running_acc / (len(labels)*jPlot) * 100,
                                epoch * len(trainLoader) + i)

                # ...log a Matplotlib Figure showing the model's predictions on a
                # random mini-batch
                # writer.add_figure('predictions vs. actuals',
                #                 plot_classes_preds(model, images.detach().cpu().numpy(), labels),
                #                 global_step=epoch * len(trainLoader) + i)
                
                # zero the loss
                if i != nExamples:
                    running_loss = 0.0
                    running_acc = 0.0
            
            printProgressBar(i+1, nExamples)

        # Compute and print the average accuracy fo this epoch when tested over all 10000 test images
        testLoss, accuracy = testAccuracy(model, testLoader, lossFunction)
        print(' For epoch', epoch+1,'the test accuracy over the whole test set is %d %%' % (accuracy))
        
        # we want to save the model if the accuracy is the best
        if accuracy > best_accuracy:
            nModel = model
            best_accuracy = accuracy
        
        writer.add_scalar('Test Accuracy',
                                accuracy,
                                epoch)
        
        writer.add_scalar('Test Loss',
                                testLoss,
                                epoch)
    
    return nModel


"""
Evalua el modelo
"""
def testAccuracy(model, testLoader, lossFunction):
    model.eval()
    testLoss = 0
    accuracy = 0
    nET = 0
    nETC = 0
    with torch.no_grad():

        for inputs, labels in testLoader:
            # inputs = inputs.to('cuda')
            inputs = inputs.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)
            output = model.forward(inputs)
            testLoss += lossFunction(output, labels).item()

            ps = torch.exp(output)
            equality = (labels.data == ps.max(dim=1)[1])
            accuracy += equality.type(torch.FloatTensor).mean()

            nET += len(labels)
            nETC += equality.sum()

    print(f" Test ACC: {nETC}/{nET}")
    return testLoss, nETC/nET*100 # accuracy
    

"""
Muestra una barra de progreso que se actualiza
    progress: progreso actual
    total: valor final del 100%
"""
def printProgressBar(progress, total):
    size = 30 #size of progress bar
    j = progress/total
    sys.stdout.write('\r')
    sys.stdout.write(f"[{'=' * int(size * j):{size}s}] {int(100 * j)}%  {progress}/{total}")
    sys.stdout.flush()



def images_to_probs(net, images):
    '''
    Generates predictions and corresponding probabilities from a trained
    network and a list of images
    '''
    output = net(images)
    # convert output probabilities to predicted class
    _, preds_tensor = torch.max(output, 1)
    preds = np.squeeze(preds_tensor.numpy())
    return preds, [F.softmax(el, dim=0)[i].item() for i, el in zip(preds, output)]


def plot_classes_preds(net, images, labels):
    '''
    Generates matplotlib Figure using a trained network, along with images
    and labels from a batch, that shows the network's top prediction along
    with its probability, alongside the actual label, coloring this
    information based on whether the prediction was correct or not.
    Uses the "images_to_probs" function.
    '''
    preds, probs = images_to_probs(net, images)
    # plot the images in the batch, along with predicted and true labels
    fig = plt.figure(figsize=(12, 48))
    for idx in np.arange(4):
        ax = fig.add_subplot(1, 4, idx+1, xticks=[], yticks=[])
        matplotlib_imshow(images[idx], one_channel=True)
        ax.set_title("{0}, {1:.1f}%\n(label: {2})".format(
            classes[preds[idx]],
            probs[idx] * 100.0,
            classes[labels[idx]]),
                    color=("green" if preds[idx]==labels[idx].item() else "red"))
    return fig