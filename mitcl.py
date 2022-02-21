
from cProfile import label
from avalanche.benchmarks.generators import nc_benchmark
from avalanche.benchmarks.generators import dataset_benchmark
from avalanche.evaluation.metrics import forgetting_metrics, accuracy_metrics,\
    loss_metrics, timing_metrics, cpu_usage_metrics, StreamConfusionMatrix,\
    disk_usage_metrics, gpu_usage_metrics
from avalanche.models import SimpleMLP
from avalanche.models import SimpleCNN
from avalanche.logging import InteractiveLogger, TextLogger, TensorboardLogger
from avalanche.training.plugins import EvaluationPlugin
from avalanche.training.strategies import Naive
from avalanche.benchmarks.utils import AvalancheDataset
from avalanche.benchmarks.utils import ImageFolder

from torch.optim import SGD
from torch.nn import CrossEntropyLoss
from torch.utils import data
from torchvision import transforms
from torchvision import datasets

import numpy as np


'''
Intentar coger un dataset en formato de directorios e imagenes (no los precargados)
en este caso,
    Indoor Scene Recognition
    http://web.mit.edu/torralba/www/indoor.html
y convertirlo a un dataset de CL (por experiencias)
'''

""" LOAD DATA """
dataset = ImageFolder('../datasets/indoorCVPR/', 
    transform=transforms.Compose([
        transforms.Resize((256,256)),
        transforms.ToTensor()
    ])
)
labels = dataset.classes
# dataset = AvalancheDataset(dataset)

lengths = [int(np.ceil(0.7*len(dataset))),
           int(np.floor(0.3*len(dataset)))]
train_set, test_set = data.random_split(dataset, lengths)

train_dataloader = data.DataLoader(train_set, batch_size=10)
test_dataloader = data.DataLoader(test_set, batch_size=10)

print(len(train_set))
print(len(test_set))

# nc_benchmark(train_dataset=train_set, test_dataset=test_set, n_experiences=1, task_labels=labels)
n = 5
for x, y in train_dataloader:
    print(y)
    print(np.array(x).shape)
    if n == 0: break;
    n-=1
""""""

# scenario = AvalancheDataset(dataset)
# for k in scenario:
#     k


nclases = len(labels)
print(labels)

"""
Hacer función que tome un modelo ya entrenado y poder enseñarle una nueva clase.
"""

bnch = nc_benchmark(train_dataset=train_set, test_dataset=test_set, n_experiences=1, task_labels=True)

# MODEL CREATION
model = SimpleCNN(nclases) #SimpleMLP(num_classes=nclases, input_size=256*256)

# DEFINE THE EVALUATION PLUGIN and LOGGERS
# The evaluation plugin manages the metrics computation.
# It takes as argument a list of metrics, collectes their results and returns
# them to the strategy it is attached to.
device = f'cuda:0'


""" LOGGING """
# log to Tensorboard
tb_logger = TensorboardLogger()

# log to text file
text_logger = TextLogger(open('log.txt', 'a'))

# print to stdout
interactive_logger = InteractiveLogger()

# Stats to log
eval_plugin = EvaluationPlugin(
    accuracy_metrics(minibatch=True, epoch=True, experience=True, stream=True),
    loss_metrics(minibatch=True, epoch=True, experience=True, stream=True),
    timing_metrics(epoch=True),
    cpu_usage_metrics(experience=True),
    forgetting_metrics(experience=True, stream=True),
    StreamConfusionMatrix(num_classes=nclases, save_image=False),
    disk_usage_metrics(minibatch=True, epoch=True, experience=True, stream=True),
    loggers=[interactive_logger, text_logger, tb_logger]
)
""""""


# CREATE THE STRATEGY INSTANCE (NAIVE)
cl_strategy = Naive(
    model, SGD(model.parameters(), lr=0.001, momentum=0.9),
    CrossEntropyLoss(), train_mb_size=32, train_epochs=1, eval_mb_size=100,
    device=device,
    evaluator=eval_plugin)

# TRAINING LOOP
print('Entrenando modelo')
results = []
for experience in bnch.train_stream:
    # print(len(experience[0]))
    print("Start of experience: ", experience.current_experience)
    print("Current Classes: ", experience.classes_in_this_experience)

    # train returns a dictionary which contains all the metric values
    # for k in experience.dataset:
    #     print(k)
    #     print(k[0].size())
    #     break

    res = cl_strategy.train(experience, num_workers=12)
    print('Training completed')

    print('Computing accuracy on the whole test set')
    # eval also returns a dictionary which contains all the metric values
    results.append(cl_strategy.eval(bnch.test_stream, num_workers=12))