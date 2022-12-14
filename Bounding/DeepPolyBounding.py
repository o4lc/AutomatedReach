import torch

from NetworkModels.NeuralNetwork import NeuralNetwork
from src.abstract_layers.abstract_network import AbstractNetwork
from src.mn_bab_verifier import MNBaBVerifier
from src.utilities.loading.network import freeze_network
from copy import deepcopy


class DeepPolyBounding:
    def __init__(self, network: NeuralNetwork, device, extraInfo=None):
        self.providesUpperBound = True
        originalNetwork = deepcopy(network)
        originalNetwork.to(device)
        originalNetwork.eval()
        originalNetwork = AbstractNetwork.from_concrete_module(originalNetwork.Linear,
                                                               originalNetwork.Linear[0].weight.shape[1])
        freeze_network(originalNetwork)
        self.network = originalNetwork

        verifier = MNBaBVerifier(
            originalNetwork,
            device,
            False,
            0,
            0,
            False,
            0,
            0,
            {},
            0,
            0,
            0,
            [],
            {},
            False,
            False,
            False,
        )


        self.verifier = verifier
        self.babOptimizer = verifier.bab.optimizer
        self.device = device

    def lowerBound(self,
                   queryCoefficient: torch.Tensor,
                   inputLowerBound: torch.Tensor,
                   inputUpperBound: torch.Tensor,
                   timer=None
                   ):
        lbs = []
        ubs = []
        for batch in range(inputLowerBound.shape[0]):
            lb, ub = self.babOptimizer.bound_minimum_with_deep_poly(queryCoefficient.unsqueeze(0).unsqueeze(0),
                                                                    self.network,
                                                                    inputLowerBound[batch, :],
                                                                    inputUpperBound[batch, :])


            lbs.append(lb[0])
            ubs.append(ub[0])

        return torch.Tensor(lbs).to(self.device), torch.Tensor(ubs).to(self.device)