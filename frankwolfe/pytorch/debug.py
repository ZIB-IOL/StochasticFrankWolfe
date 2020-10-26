from barbar import Bar
import math
import sys

import torch
from torch import nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms

import frankwolfe.pytorch as fw
import warnings

class Utilities:

    @staticmethod
    @torch.no_grad()
    def categorical_accuracy(y_true, output, topk=1):
        """Computes the precision@k for the specified values of k"""
        prediction = output.topk(topk, dim=1, largest=True, sorted=False).indices.t()
        n_labels = float(len(y_true))
        return prediction.eq(y_true.expand_as(prediction)).sum().item() / n_labels

class RetractionLR(torch.optim.lr_scheduler._LRScheduler):
    """
    Retracts the learning rate as follows. Two running averages are kept, one of length n_close, one of n_far. Adjust
    the learning_rate depending on the relation of far_average and close_average. Decrease by 1-retraction_factor.
    Increase by 1/(1 - retraction_factor*growth_factor)
    """
    def __init__(self, optimizer, retraction_factor=0.3, n_close=5, n_far=10, lowerBound=1e-5, upperBound=1, growth_factor=0.2, last_epoch=-1):
        self.retraction_factor = retraction_factor
        self.n_close = n_close
        self.n_far = n_far
        self.lowerBound = lowerBound
        self.upperBound = upperBound
        self.growth_factor = growth_factor

        assert (0 <= self.retraction_factor < 1), "Retraction factor must be in [0, 1[."
        assert (0 <= self.lowerBound < self.upperBound <= 1), "Bounds must be in [0, 1]"
        assert (0 < self.growth_factor <= 1), "Growth factor must be in ]0, 1]"

        self.closeAverage = RunningAverage(self.n_close)
        self.farAverage = RunningAverage(self.n_far)

        super(RetractionLR, self).__init__(optimizer, last_epoch)

    def update_averages(self, loss):
        self.closeAverage(loss)
        self.farAverage(loss)

    def get_lr(self):
        if not self._get_lr_called_within_step:
            warnings.warn("To get the last learning rate computed by the scheduler, "
                          "please use `get_last_lr()`.", UserWarning)

        factor = 1
        if self.farAverage.is_complete() and self.closeAverage.is_complete():
            if self.closeAverage.result() > self.farAverage.result():
                # Decrease the learning rate
                factor = 1 - self.retraction_factor
            elif self.farAverage.result() > self.closeAverage.result():
                # Increase the learning rate
                factor = 1./(1 - self.retraction_factor*self.growth_factor)

        return [max(self.lowerBound, min(factor * group['lr'], self.upperBound)) for group in self.optimizer.param_groups]

class RunningAverage(object):
    """Tracks the running average of n numbers"""
    def __init__(self, n):
        self.n = n
        self.reset()

    def reset(self):
        self.sum = 0
        self.avg = 0
        self.entries = []

    def result(self):
        return self.avg

    def get_count(self):
        return len(self.entries)

    def is_complete(self):
        return len(self.entries) == self.n

    def __call__(self, val):
        if len(self.entries) == self.n:
            l = self.entries.pop(0)
            self.sum -= l
        self.entries.append(val)
        self.sum += val
        self.avg = self.sum / len(self.entries)

    def __str__(self):
        return str(self.avg)

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.sum = 0
        self.count = 0
        self.avg = 0

    def result(self):
        return self.avg

    def __call__(self, val, n=1):
        """val is an average over n samples. To compute the overall average, add val*n to sum and increase count by n"""
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        return str(self.avg)

means = {
    'cifar10': (0.4914, 0.4822, 0.4465),
    'cifar100': (0.5071, 0.4867, 0.4408),
    'imagenet': (0.485, 0.456, 0.406),
}

stds = {
    'cifar10': (0.2023, 0.1994, 0.2010),
    'cifar100': (0.2675, 0.2565, 0.2761),
    'imagenet': (0.229, 0.224, 0.225),
}


datasetDict = {  # Links dataset names to actual torch datasets
    'mnist': getattr(torchvision.datasets, 'MNIST'),
    'cifar10': getattr(torchvision.datasets, 'CIFAR10'),
    'cifar100': getattr(torchvision.datasets, 'CIFAR100'),
    #'imagenet': getattr(torchvision.datasets, 'ImageNet'),
}

# Note: previously, these were dependent on the model name, now they are on the dataset name only
trainTransformDict = {  # Links dataset names to train dataset transformers
    'mnist': transforms.Compose([transforms.ToTensor()]),
    'cifar10': transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=means['cifar10'], std=stds['cifar10']), ]),
    'cifar100': transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(mean=means['cifar100'], std=stds['cifar100']), ]),
    'imagenet': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=means['imagenet'], std=stds['imagenet']),]),
}
testTransformDict = {  # Links dataset names to test dataset transformers
    'mnist': transforms.Compose([transforms.ToTensor()]),
    'cifar10': transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=means['cifar10'], std=stds['cifar10']), ]),
    'cifar100': transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=means['cifar100'], std=stds['cifar100']), ]),
    'imagenet': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=means['imagenet'], std=stds['imagenet']),]),
}

dataset_name = 'mnist' #@param ['cifar10', 'cifar100']
model_type = 'Simple' #@param ['DenseNet', 'WideResNet', 'GoogLeNet', 'ResNeXt']

root = f"{dataset_name}-dataset"
trainData = datasetDict[dataset_name](root=root, train=True, download=True,
                                            transform=trainTransformDict[dataset_name])
testData = datasetDict[dataset_name](root=root, train=False,
                                        transform=testTransformDict[dataset_name])

# initialize model
if model_type == 'Simple':
    class SimpleCNN(torch.nn.Module):
        """Translated from https://github.com/Davidnet/TensorFlow_Examples/blob/master/keras/mnist_cnn.py"""

        def __init__(self):
            super(SimpleCNN, self).__init__()
            self.conv_layer = torch.nn.Sequential(
                # Conv Layer block 1
                torch.nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=0),
                torch.nn.ReLU(inplace=True),
                torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=0),
                torch.nn.ReLU(inplace=True),
                torch.nn.MaxPool2d(kernel_size=2),
                torch.nn.Dropout(p=0.25)
            )

            self.fc_layer = torch.nn.Sequential(
                torch.nn.Linear(9216, 128),
                torch.nn.ReLU(inplace=True),
                torch.nn.Dropout(p=0.5),
                torch.nn.Linear(128, 10)
            )

        def forward(self, x):
            """Perform forward."""
            x = self.conv_layer(x)
            x = x.view(x.size(0), -1)
            x = self.fc_layer(x)
            return x
    model = SimpleCNN()
elif model_type == 'DenseNet':
    model = torchvision.models.densenet121(pretrained=False)
elif model_type == 'WideResNet':
    class WideResNet(nn.Module):
        def __init__(self, depth=28, widen_factor=10, dropout_rate=0.3, num_classes=10):
            super().__init__()
            self.in_planes = 16

            assert ((depth-4)%6 ==0), 'Wide-resnet depth should be 6n+4'
            n = (depth-4)/6
            k = widen_factor

            nStages = [16, 16*k, 32*k, 64*k]

            class wide_basic(nn.Module):
                def __init__(self, in_planes, planes, dropout_rate, stride=1):
                    super(wide_basic, self).__init__()
                    self.bn1 = nn.BatchNorm2d(in_planes)
                    self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, padding=1, bias=True)
                    self.dropout = nn.Dropout(p=dropout_rate)
                    self.bn2 = nn.BatchNorm2d(planes)
                    self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=True)

                    self.shortcut = nn.Sequential()
                    if stride != 1 or in_planes != planes:
                        self.shortcut = nn.Sequential(
                            nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=True),
                        )

                def forward(self, x):
                    out = self.dropout(self.conv1(F.relu(self.bn1(x))))
                    out = self.conv2(F.relu(self.bn2(out)))
                    out += self.shortcut(x)

                    return out

            self.conv1 = self.conv3x3(3,nStages[0])
            self.layer1 = self._wide_layer(wide_basic, nStages[1], n, dropout_rate, stride=1)
            self.layer2 = self._wide_layer(wide_basic, nStages[2], n, dropout_rate, stride=2)
            self.layer3 = self._wide_layer(wide_basic, nStages[3], n, dropout_rate, stride=2)
            self.bn1 = nn.BatchNorm2d(nStages[3], momentum=0.9)
            self.linear = nn.Linear(nStages[3], num_classes)

        def conv3x3(self, in_planes, out_planes, stride=1):
            return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=True)

        def _wide_layer(self, block, planes, num_blocks, dropout_rate, stride):
            strides = [stride] + [1]*(int(num_blocks)-1)
            layers = []

            for stride in strides:
                layers.append(block(self.in_planes, planes, dropout_rate, stride))
                self.in_planes = planes

            return nn.Sequential(*layers)

        def forward(self, x):
            out = self.conv1(x)
            out = self.layer1(out)
            out = self.layer2(out)
            out = self.layer3(out)
            out = F.relu(self.bn1(out))
            out = F.avg_pool2d(out, 8)
            out = out.view(out.size(0), -1)
            out = self.linear(out)

            return out
    model = WideResNet(num_classes=10 if dataset_name == 'cifar10' else 100)
elif model_type == 'ResNeXt':
    model = torchvision.models.resnext50_32x4d(pretrained=False)
elif model_type == 'GoogLeNet':
    class GoogleNet(torch.nn.Module):
        def __init__(self, num_class=100):
            super().__init__()

            class Inception(torch.nn.Module):
                def __init__(self, input_channels, n1x1, n3x3_reduce, n3x3, n5x5_reduce, n5x5, pool_proj):
                    super().__init__()

                    # 1x1conv branch
                    self.b1 = nn.Sequential(
                        nn.Conv2d(input_channels, n1x1, kernel_size=1),
                        nn.BatchNorm2d(n1x1),
                        nn.ReLU(inplace=True)
                    )

                    # 1x1conv -> 3x3conv branch
                    self.b2 = nn.Sequential(
                        nn.Conv2d(input_channels, n3x3_reduce, kernel_size=1),
                        nn.BatchNorm2d(n3x3_reduce),
                        nn.ReLU(inplace=True),
                        nn.Conv2d(n3x3_reduce, n3x3, kernel_size=3, padding=1),
                        nn.BatchNorm2d(n3x3),
                        nn.ReLU(inplace=True)
                    )

                    # 1x1conv -> 5x5conv branch
                    # we use 2 3x3 conv filters stacked instead
                    # of 1 5x5 filters to obtain the same receptive
                    # field with fewer parameters
                    self.b3 = nn.Sequential(
                        nn.Conv2d(input_channels, n5x5_reduce, kernel_size=1),
                        nn.BatchNorm2d(n5x5_reduce),
                        nn.ReLU(inplace=True),
                        nn.Conv2d(n5x5_reduce, n5x5, kernel_size=3, padding=1),
                        nn.BatchNorm2d(n5x5, n5x5),
                        nn.ReLU(inplace=True),
                        nn.Conv2d(n5x5, n5x5, kernel_size=3, padding=1),
                        nn.BatchNorm2d(n5x5),
                        nn.ReLU(inplace=True)
                    )

                    # 3x3pooling -> 1x1conv
                    # same conv
                    self.b4 = nn.Sequential(
                        nn.MaxPool2d(3, stride=1, padding=1),
                        nn.Conv2d(input_channels, pool_proj, kernel_size=1),
                        nn.BatchNorm2d(pool_proj),
                        nn.ReLU(inplace=True)
                    )

                def forward(self, x):
                    return torch.cat([self.b1(x), self.b2(x), self.b3(x), self.b4(x)], dim=1)


            self.prelayer = nn.Sequential(
                nn.Conv2d(3, 192, kernel_size=3, padding=1),
                nn.BatchNorm2d(192),
                nn.ReLU(inplace=True)
            )

            #although we only use 1 conv layer as prelayer,
            #we still use name a3, b3.......
            self.a3 = Inception(192, 64, 96, 128, 16, 32, 32)
            self.b3 = Inception(256, 128, 128, 192, 32, 96, 64)

            #"""In general, an Inception network is a network consisting of
            #modules of the above type stacked upon each other, with occasional
            #max-pooling layers with stride 2 to halve the resolution of the
            #grid"""
            self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)

            self.a4 = Inception(480, 192, 96, 208, 16, 48, 64)
            self.b4 = Inception(512, 160, 112, 224, 24, 64, 64)
            self.c4 = Inception(512, 128, 128, 256, 24, 64, 64)
            self.d4 = Inception(512, 112, 144, 288, 32, 64, 64)
            self.e4 = Inception(528, 256, 160, 320, 32, 128, 128)

            self.a5 = Inception(832, 256, 160, 320, 32, 128, 128)
            self.b5 = Inception(832, 384, 192, 384, 48, 128, 128)

            #input feature size: 8*8*1024
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            self.dropout = nn.Dropout2d(p=0.4)
            self.linear = nn.Linear(1024, num_class)

        def forward(self, x):
            output = self.prelayer(x)
            output = self.a3(output)
            output = self.b3(output)

            output = self.maxpool(output)

            output = self.a4(output)
            output = self.b4(output)
            output = self.c4(output)
            output = self.d4(output)
            output = self.e4(output)

            output = self.maxpool(output)

            output = self.a5(output)
            output = self.b5(output)

            #"""It was found that a move from fully connected layers to
            #average pooling improved the top-1 accuracy by about 0.6%,
            #however the use of dropout remained essential even after
            #removing the fully connected layers."""
            output = self.avgpool(output)
            output = self.dropout(output)
            output = output.view(output.size()[0], -1)
            output = self.linear(output)

            return output
    model = GoogleNet(num_class=10 if dataset_name == 'cifar10' else 100)


#@title Choosing Lp-Norm constraints
#@markdown To constrain the parameters we have to specify the order (ord) of the Lp-Ball (which is equal to p, here). Furthermore, we distinguish between three different modes:
#@markdown - radius: value just specifies the radius of the LpBall. Each single layer is constrained to lie in an LpBall of equal radius.
#@markdown - diameter: value just specifies the radius of the LpBall. Each single layer is constrained to lie in an LpBall of equal diameter.
#@markdown - initialization: Each layer is constrained by an LpBall whose radius is determined by the average initialization norm of that layer times the factor value.
ord =  "2" #@param [1, 2, 5, 'inf']
ord = float(ord)
value = 1 #@param {type:"number"}
mode = 'initialization' #@param ['initialization', 'radius', 'diameter']

assert value > 0

# Select constraints
fw.constraints.set_lp_constraints(model, ord=ord, value=value, mode=mode)
for p in model.parameters():
    print(p.shape, p.constraint)

#@title Configuring the Frank-Wolfe Algorithm
#@markdown Select the momentum parameter between 0 and 1. Furthermore, the learning rate can be decoupled from the size of the feasible region, using three different modi: 'gradient', 'diameter' and 'None'. For an in-depth discussion see Section 3.1 of [arXiv:2010.07243](https://arxiv.org/pdf/2010.07243.pdf).
momentum = 0.9 #@param {type:"number"}
rescale = 'diameter' #@param ['gradient', 'diameter', 'None']
rescale = None if rescale == 'None' else rescale
#@markdown ---
#@markdown #### Adjusting the learning rate
#@markdown We choose the initial learning rate and can activate the learning rate scheduler, which automatically multiplies the current learning rate by lr_decrease_factor every lr_step_size epochs.
learning_rate = 0.1 #@param {type:"number"}
lr_scheduler_active = True #@param {type:"boolean"}
lr_decrease_factor = 0.1 #@param {type:"number"}
lr_step_size = 60 #@param {type:"integer"}

#@markdown Further, it is of use to enable a retraction of the learning rate, i.e. if enabled it is increased/decreased automatically depending only on two indicators, namely two moving averages (of different lengths) of the train loss over the epochs.

retraction = True #@param {type:"boolean"}

assert learning_rate > 0
assert 0 <= momentum <= 1
assert lr_decrease_factor > 0
assert lr_step_size > 0


# Select optimizer
optimizer = fw.optimizers.SFW(params=model.parameters(), learning_rate=learning_rate, momentum=momentum, rescale=rescale)


#@title Training the network
#@markdown Last but not least, the number of epochs and the size of each batch have to be specified:
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
nepochs = 1 #@param {type:"integer"}
batch_size =  1028 #@param {type:"integer"}

fw.constraints.make_feasible(model)

# define the loss object
loss_criterion = torch.nn.CrossEntropyLoss().to(device=device)
model = model.to(device=device)

# Loaders
trainLoader = torch.utils.data.DataLoader(trainData, batch_size=batch_size, shuffle=True,
                        pin_memory=torch.cuda.is_available(), num_workers=2)
testLoader = torch.utils.data.DataLoader(testData, batch_size=batch_size, shuffle=False,
                        pin_memory=torch.cuda.is_available(), num_workers=2)

# initialize some necessary metrics objects
train_loss, train_accuracy = AverageMeter(), AverageMeter()
test_loss, test_accuracy = AverageMeter(), AverageMeter()

if lr_scheduler_active:
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=lr_step_size, gamma=lr_decrease_factor)

if retraction:
    retractionScheduler = RetractionLR(optimizer=optimizer)

# function to reset metrics
def reset_metrics():
    train_loss.reset()
    train_accuracy.reset()

    test_loss.reset()
    test_accuracy.reset()

@torch.no_grad()
def evaluate_model(data='train'):
    if data == 'train':
        loader = trainLoader
        mean_loss, mean_accuracy = train_loss, train_accuracy
    elif data == 'test':
        loader = testLoader
        mean_loss, mean_accuracy = test_loss, test_accuracy

    sys.stdout.write(f"Evaluation of {data} data:\n")
    for x_input, y_target in Bar(loader):
        x_input, y_target = x_input.to(device), y_target.to(device)  # Move to CUDA if possible
        output = model.eval()(x_input)
        loss = loss_criterion(output, y_target)
        mean_loss(loss.item(), len(y_target))
        mean_accuracy(Utilities.categorical_accuracy(y_true=y_target, output=output), len(y_target))

for epoch in range(nepochs + 1):
    reset_metrics()
    sys.stdout.write(f"\n\nEpoch {epoch}/{nepochs}\n")
    if epoch == 0:
        # Just evaluate the model once to get the metrics
        continue
        evaluate_model(data='train')
    else:
        # Train
        sys.stdout.write(f"Training:\n")
        for x_input, y_target in Bar(trainLoader):
            x_input, y_target = x_input.to(device), y_target.to(device)  # Move to CUDA if possible
            optimizer.zero_grad()  # Zero the gradient buffers
            output = model.train()(x_input)
            loss = loss_criterion(output, y_target)
            loss.backward()  # Backpropagation
            optimizer.step()
            train_loss(loss.item(), len(y_target))
            train_accuracy(Utilities.categorical_accuracy(y_true=y_target, output=output), len(y_target))
        if lr_scheduler_active:
            scheduler.step()
        if retraction:
            # Learning rate retraction
            retractionScheduler.update_averages(train_loss.result())
            retractionScheduler.step()

    evaluate_model(data='test')
    sys.stdout.write(f"\n Finished epoch {epoch}/{nepochs}: Train Loss {train_loss.result()} | Test Loss {test_loss.result()} | Train Acc {train_accuracy.result()} | Test Acc {test_accuracy.result()}\n")
