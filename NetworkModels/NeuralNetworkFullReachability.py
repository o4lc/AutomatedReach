import torch.nn
import copy

from packages import *


class NeuralNetworkReachability(nn.Module):
    def __init__(self, path=None,
                 A=None, B=None, c=None,
                 numberOfTimeSteps=1, multiStepSingleHorizon=True,
                 layers=None):
        super().__init__()
        if layers:
            self.layers = layers
        else:
            cpuDevice = torch.device("cpu")
            stateDictionary = torch.load(path, map_location=cpuDevice)
            for key in stateDictionary:
                inputDimension = stateDictionary[key].shape[1]
                break
            for key in reversed(stateDictionary):
                outputDimension = stateDictionary[key].shape[0]
                break
            flag = False
            if A is None:
                flag = True
                A = torch.zeros((outputDimension, inputDimension))
                B = torch.eye(outputDimension)
                c = torch.zeros(outputDimension)
            c = c.squeeze() if len(c.shape) > 1 else c
            A = A.to(cpuDevice)
            B = B.to(cpuDevice)
            c = c.to(cpuDevice)
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

            self.layers, self.originalWeights =\
                createLayersForSingleStep(A, B, c, stateDictionary, inputDimension, flag)

            for i in range(numberOfTimeSteps - 1):

                layers, _ = createLayersForSingleStep(A, B, c, stateDictionary, inputDimension, flag)
                self.layers += layers

            # self.layers = layers
            # originalLayers = copy.deepcopy(layers)
            # originalSize = len(originalLayers)
            # if multiStepSingleHorizon:
            #     for i in range(numberOfTimeSteps - 1):
            #         for j in range(originalSize):
            #             if type(originalLayers[j]) == nn.modules.linear.Linear:
            #                 w0 = originalLayers[j].weight
            #                 b0 = originalLayers[j].bias
            #                 layers.append(nn.Linear(w0.shape[1], w0.shape[0]))
            #                 layers[-1].weight = nn.Parameter(w0)
            #                 layers[-1].bias = nn.Parameter(b0)
            #             elif type(originalLayers[j]) == nn.modules.activation.ReLU:
            #                 layers.append(nn.ReLU())
            #             elif type(originalLayers[j]) == nn.modules.linear.Identity:
            #                 layers.append(nn.Identity())
            #             else:
            #                 raise ValueError
        self.Linear = nn.Sequential(
            *self.layers
        )

    def setRotation(self, rotationLayer):
        layers = [rotationLayer] + self.layers
        self.Linear = nn.Sequential(*layers)

    def forward(self, x):
        x = self.Linear(x)
        return x


def createLayersForSingleStep(A, B, c, stateDictionary, inputDimension,
                              fullReachabilityFlag=True):
    cpuDevice = torch.device("cpu")
    layers = [nn.Linear(inputDimension, 3 * inputDimension)]
    originalWeights = []
    layers[0].weight = torch.nn.Parameter(torch.vstack([torch.eye(inputDimension),
                                                        -torch.eye(inputDimension),
                                                        torch.eye(inputDimension)]))
    layers[0].bias = torch.nn.Parameter(torch.zeros(int(3 * inputDimension)))
    # layers[0].bias = torch.nn.Parameter(torch.hstack([-lowerBound, -lowerBound]))
    # layers.append(nn.ReLU())
    # layers.append(nn.Identity())
    count = 1
    for keyEntry in stateDictionary:
        if "weight" in keyEntry:
            originalWeights.append(stateDictionary[keyEntry])
            weightKeyEntry = keyEntry
            layers.append(nn.Linear(int(2 * inputDimension) + stateDictionary[keyEntry].shape[1],
                                    int(2 * inputDimension) + stateDictionary[keyEntry].shape[0]))

            layers[-1].weight = torch.nn.Parameter(torch.block_diag(torch.eye(inputDimension),
                                                                    torch.eye(inputDimension),
                                                                    stateDictionary[keyEntry]))

            layers[-1].bias = torch.nn.Parameter(torch.hstack([torch.zeros(int(2 * inputDimension)),
                                                               stateDictionary[keyEntry[:-6] + "bias"]]))
            layers.append(nn.ReLU())
    layers.pop()
    outputDimension = layers[-1].bias.shape[0] - int(2 * inputDimension)
    # layers[-1].bias = torch.nn.Parameter(torch.hstack([lowerBound,
    #                                                    stateDictionary[weightKeyEntry[:-6] + "bias"]])
    #                                      + 100 * torch.ones(inputDimension + outputDimension))
    # layers[-1].bias = torch.nn.Parameter(torch.hstack([lowerBound.to(cpuDevice),
    #                                                    stateDictionary[
    #                                                        weightKeyEntry[:-6] + "bias"]]))

    # layers.append(nn.ReLU())
    # layers.append(nn.Identity())
    layers.append(nn.Linear(layers[-1].bias.shape[0],
                            outputDimension if fullReachabilityFlag else inputDimension))
    layers[-1].weight = torch.nn.Parameter(torch.hstack([A, -A, B]))
    # layers[-1].bias = torch.nn.Parameter(c - (100 * A @ torch.ones((inputDimension, 1)) +
    #                                                        100 * B @ torch.ones((outputDimension, 1))).squeeze())
    layers[-1].bias = torch.nn.Parameter(c)
    return layers, originalWeights


def calculateBoundsAfterLinearTransformation(weight, bias, lowerBound, upperBound):
    """
    :param weight:
    :param bias: A (n * 1) matrix
    :param lowerBound: A vector and not an (n * 1) matrix
    :param upperBound: A vector and not an (n * 1) matrix
    :return:
    """

    outputLowerBound = (np.maximum(weight, 0) @ (lowerBound[np.newaxis].transpose())
                        + np.minimum(weight, 0) @ (upperBound[np.newaxis].transpose()) + bias).squeeze()
    outputUpperBound = (np.maximum(weight, 0) @ (upperBound[np.newaxis].transpose())
                        + np.minimum(weight, 0) @ (lowerBound[np.newaxis].transpose()) + bias).squeeze()

    return outputLowerBound, outputUpperBound


def propagateBoundsInNetwork(l, u, layers):
    """
    NOTE: This function does not take into consideration that there are any ReLUs in the network and only considers
    linear layers. Using information of existence of ReLUs can help in achieving better bounds which is not needed
    for our purposes
    """
    weights = []
    biases = []
    for layer in layers:
        if type(layer) == nn.modules.linear.Linear:
            weights.append(layer.weight.detach().numpy())
            biases.append(layer.bias.unsqueeze(1).detach().numpy())
    s, t = [l.numpy()], [u.numpy()]
    for i, (W, b) in enumerate(zip(weights, biases)):
        val1, val2 = s[-1], t[-1]
        if i > 0 and type(layers[i - 1]) == nn.modules.activation.ReLU:
            val1, val2 = np.maximum(val1, 0), np.maximum(val2, 0)
        if val1.shape == ():
            val1 = np.array([val1])
            val2 = np.array([val2])
        sTemp, tTemp = calculateBoundsAfterLinearTransformation(W, b, val1, val2)
        if sTemp.shape == ():
            sTemp = np.array([sTemp])
            tTemp = np.array([tTemp])
        s.append(sTemp)
        t.append(tTemp)
    return torch.from_numpy(s[-1]), torch.from_numpy(t[-1])
