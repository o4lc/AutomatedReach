import torch.nn

from packages import *


class NeuralNetworkReachability(nn.Module):
    def __init__(self, path, lowerBound, A=None, B=None, c=None, numberOfTimeSteps=1, multiStepSingleHorizon=True):
        super().__init__()
        cpuDevice = torch.device("cpu")
        stateDictionary = torch.load(path, map_location=cpuDevice)
        # if A is None:
        #     layers = []
        #     for keyEntry in stateDictionary:
        #         if "weight" in keyEntry:
        #             layers.append(nn.Linear(stateDictionary[keyEntry].shape[1], stateDictionary[keyEntry].shape[0]))
        #             layers.append(nn.ReLU())
        #     layers.pop()
        #     self.Linear = nn.Sequential(
        #         *layers
        #     )
        #     self.load_state_dict(stateDictionary)
        # else:

        inputDimension = lowerBound.shape[0]
        layers = [nn.Linear(inputDimension, 2 * inputDimension)]

        layers[0].weight = torch.nn.Parameter(torch.vstack([torch.eye(inputDimension), torch.eye(inputDimension)]))
        layers[0].bias = torch.nn.Parameter(torch.hstack([-lowerBound.to(cpuDevice), torch.zeros(inputDimension)]))
        # layers[0].bias = torch.nn.Parameter(torch.hstack([-lowerBound, -lowerBound]))
        # layers.append(nn.ReLU())
        # layers.append(nn.Identity())
        count = 1
        for keyEntry in stateDictionary:
            if "weight" in keyEntry:
                weightKeyEntry = keyEntry
                layers.append(nn.Linear(inputDimension + stateDictionary[keyEntry].shape[1],
                                        inputDimension + stateDictionary[keyEntry].shape[0]))

                layers[-1].weight = torch.nn.Parameter(torch.block_diag(torch.eye(inputDimension),
                                                                         stateDictionary[keyEntry]))
                if count == 0:
                    count += 1
                    layers[-1].bias =\
                        torch.nn.Parameter(torch.hstack([torch.zeros(inputDimension),
                                                         stateDictionary[keyEntry[:-6] + "bias"] +
                                                         stateDictionary[keyEntry] @ lowerBound.to(cpuDevice)]))
                else:
                    layers[-1].bias = torch.nn.Parameter(torch.hstack([torch.zeros(inputDimension),
                                                                       stateDictionary[keyEntry[:-6] + "bias"]]))
                layers.append(nn.ReLU())
        layers.pop()
        outputDimension = layers[-1].bias.shape[0] - inputDimension
        # layers[-1].bias = torch.nn.Parameter(torch.hstack([lowerBound,
        #                                                    stateDictionary[weightKeyEntry[:-6] + "bias"]]) + 100 * torch.ones(inputDimension + outputDimension))
        layers[-1].bias = torch.nn.Parameter(torch.hstack([lowerBound.to(cpuDevice),
                                                           stateDictionary[
                                                               weightKeyEntry[:-6] + "bias"]]))

        flag = False
        if A is None:
            flag = True
            A = torch.zeros((outputDimension, inputDimension))
            B = torch.eye(outputDimension)
            c = torch.zeros(outputDimension)
        c = c.squeeze() if len(c.shape) > 1 else c
        # layers.append(nn.ReLU())
        # layers.append(nn.Identity())
        layers.append(nn.Linear(layers[-1].bias.shape[0],
                                outputDimension if flag else inputDimension))
        layers[-1].weight = torch.nn.Parameter(torch.hstack([A, B]))
        # layers[-1].bias = torch.nn.Parameter(c - (100 * A @ torch.ones((inputDimension, 1)) +
        #                                                        100 * B @ torch.ones((outputDimension, 1))).squeeze())
        layers[-1].bias = torch.nn.Parameter(c)
        # for l in layers:
        #     try:
        #         print(l.weight.shape, l.bias.shape)
        #     except:
        #         continue
        self.layers = layers
        if multiStepSingleHorizon:
            originalLength = len(layers)
            for i in range(numberOfTimeSteps - 1):
                for j in range(originalLength):
                    layers.append(layers[j])
        self.Linear = nn.Sequential(
            *layers
        )

    def setRotation(self, rotationLayer):
        layers = [rotationLayer] + self.layers
        self.Linear = nn.Sequential(*layers)

    def forward(self, x):
        x = self.Linear(x)
        return x


