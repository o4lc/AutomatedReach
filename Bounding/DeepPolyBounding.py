import torch

from NetworkModels.NeuralNetwork import NeuralNetwork
from src.abstract_layers.abstract_network import AbstractNetwork
from src.mn_bab_verifier import MNBaBVerifier
from src.utilities.loading.network import freeze_network
from copy import deepcopy


class DeepPolyBounding:
    def __init__(self, network: NeuralNetwork, device, extraInfo=None):
        originalNetwork = deepcopy(network)
        originalNetwork.to(device)
        originalNetwork.eval()
        originalNetwork = AbstractNetwork.from_concrete_module(originalNetwork.Linear,
                                                               originalNetwork.Linear[0].weight.shape[1])
        freeze_network(originalNetwork)
        self.network = originalNetwork
        from src.utilities.argument_parsing import get_args, get_config_from_json
        config = get_config_from_json("configs/2dRandomNetwork1.json")

        # verifier = MNBaBVerifier(
        #     network,
        #     device,
        #     False,
        #     0,
        #     0,
        #     False,
        #     0,
        #     0,
        #     {},
        #     0,
        #     0,
        #     0,
        #     [],
        #     {},
        #     False,
        #     False,
        #     False,
        # )
        verifier = MNBaBVerifier(
            originalNetwork,
            device,
            config.optimize_alpha,
            config.alpha_lr,
            config.alpha_opt_iterations,
            config.optimize_prima,
            config.prima_lr,
            config.prima_opt_iterations,
            config.prima_hyperparameters,
            config.peak_lr_scaling_factor,
            config.final_lr_div_factor,
            config.beta_lr,
            config.bab_batch_sizes,
            config.branching,
            config.recompute_intermediate_bounds_after_branching,
            config.use_dependence_sets,
            config.use_early_termination,
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
        for batch in range(inputLowerBound.shape[0]):
            lb, ub = self.babOptimizer.bound_minimum_with_deep_poly(queryCoefficient.unsqueeze(0).unsqueeze(0),
                                                                    self.network,
                                                                    inputLowerBound[batch, :],
                                                                    inputUpperBound[batch, :])

            # boundedSubProblem, intermediateBound =\
            #     self.verifier.bab.optimizer.bound_root_subproblem(-queryCoefficient.unsqueeze(0).unsqueeze(0),
            #                                                       self.network,
            #                                                       inputLowerBound, inputUpperBound,
            #                                                       float("inf"), 300,
            #                                                       self.verifier.bab.device)
            #
            # lb = [boundedSubProblem.lower_bound]

            lbs.append(lb[0])

        return torch.Tensor(lbs).to(self.device)