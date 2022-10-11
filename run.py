from tabnanny import verbose
import torch

from packages import *

from BranchAndBound import BranchAndBound
from NeuralNetwork import NeuralNetwork
import pandas as pd
from sklearn.decomposition import PCA
import copy
import json

torch.set_printoptions(precision=8)


def calculateDirectionsOfOptimization(onlyPcaDirections, imageData):
    data_mean = 0
    inputData = None
    if onlyPcaDirections:
        pca = PCA()
        pcaData = pca.fit_transform(imageData)

        data_mean = pca.mean_
        data_comp = pca.components_
        data_sd = np.sqrt(pca.explained_variance_)

        inputData = torch.from_numpy(data_comp @ (imageData.cpu().numpy() - data_mean).T).T.float()
        # print(np.linalg.norm(data_comp, 2, 1))

        pcaDirections = []
        for direction in data_comp:
            pcaDirections.append(-direction)
            pcaDirections.append(direction)

    else:
        pcaDirections = []
        numDirections = 30

        data_comp = np.array(
            [[np.cos(i * np.pi / numDirections), np.sin(i * np.pi / numDirections)] for i in range(numDirections)])
        for direction in data_comp:
            pcaDirections.append(-direction)
            pcaDirections.append(direction)
    return pcaDirections, data_comp, data_mean, inputData


def calculateDirectionsOfHigherDimProjections(currentPcaDirections, imageData):
    indexToStartReadingBoundsForPlotting = len(currentPcaDirections)
    projectedImageData = imageData.clone()
    projectedImageData[:, 2:] = 0
    pca2 = PCA()
    _ = pca2.fit_transform(projectedImageData)
    plottingDirections = pca2.components_
    for direction in plottingDirections[:2]:
        currentPcaDirections.append(-direction)
        currentPcaDirections.append(direction)
    return indexToStartReadingBoundsForPlotting


def solveSingleStepReachability(pcaDirections, imageData, config, iteration, device, network,
                                plottingConstants, calculatedLowerBoundsforpcaDirections,
                                originalNetwork, horizonForLipschitz, lowerCoordinate, upperCoordinate):
    eps = config['eps']
    verbose = config['verbose']
    verboseEssential = config['verboseEssential']
    scoreFunction = config['scoreFunction']
    virtualBranching = config['virtualBranching']
    numberOfVirtualBranches = config['numberOfVirtualBranches']
    maxSearchDepthLipschitzBound = config['maxSearchDepthLipschitzBound']
    normToUseLipschitz = config['normToUseLipschitz']
    useTwoNormDilation = config['useTwoNormDilation']
    useSdpForLipschitzCalculation = config['useSdpForLipschitzCalculation']
    lipschitzSdpSolverVerbose = config['lipschitzSdpSolverVerbose']
    initialGD = config['initialGD']
    dim = network.Linear[0].weight.shape[1]

    for i in range(len(pcaDirections)):
        previousLipschitzCalculations = []
        if i % 2 == 1 and torch.allclose(pcaDirections[i], -pcaDirections[i - 1]):
            previousLipschitzCalculations = BB.lowerBoundClass.calculatedLipschitzConstants
        c = pcaDirections[i]
        if True:
            print('** Solving Horizon: ', iteration, 'dimension: ', i)
        initialBub = torch.min(imageData @ c)
        # initialBub = None
        BB = BranchAndBound(upperCoordinate, lowerCoordinate, verbose=verbose, verboseEssential=verboseEssential,
                            inputDimension=dim,
                            eps=eps, network=network, queryCoefficient=c, currDim=i, device=device,
                            nodeBranchingFactor=2, branchNodeNum=512,
                            scoreFunction=scoreFunction,
                            pgdIterNum=0, pgdNumberOfInitializations=1, pgdStepSize=0.5,
                            virtualBranching=virtualBranching,
                            numberOfVirtualBranches=numberOfVirtualBranches,
                            maxSearchDepthLipschitzBound=maxSearchDepthLipschitzBound,
                            normToUseLipschitz=normToUseLipschitz, useTwoNormDilation=useTwoNormDilation,
                            useSdpForLipschitzCalculation=useSdpForLipschitzCalculation,
                            lipschitzSdpSolverVerbose=lipschitzSdpSolverVerbose,
                            initialGD=initialGD,
                            previousLipschitzCalculations=previousLipschitzCalculations,
                            originalNetwork=originalNetwork,
                            horizonForLipschitz=horizonForLipschitz,
                            initialBub=initialBub
                            )
        lowerBound, upperBound, space_left = BB.run()
        plottingConstants[i] = -lowerBound
        calculatedLowerBoundsforpcaDirections[i] = lowerBound

        print('Best lower/upper bounds are:', lowerBound, '->', upperBound)


def main():
    configFileToLoad = "Config/quadRotor.json"
    with open(configFileToLoad, 'r') as file:
        config = json.load(file)

    eps = config['eps']
    verboseMultiHorizon = config['verboseMultiHorizon']
    normToUseLipschitz = config['normToUseLipschitz']
    useSdpForLipschitzCalculation = config['useSdpForLipschitzCalculation']
    finalHorizon = config['finalHorizon']
    performMultiStepSingleHorizon = config['performMultiStepSingleHorizon']
    plotProjectionsOfHigherDims = config['plotProjectionsOfHigherDims']
    onlyPcaDirections = config['onlyPcaDirections']
    pathToStateDictionary = config['pathToStateDictionary']
    fullLoop = config['fullLoop']
    if config['A']:
        A = torch.Tensor(config['A'])
        B = torch.Tensor(config['B'])
        c = torch.Tensor(config['c'])
    else:
        A = B = c = None
    lowerCoordinate = torch.Tensor(config['lowerCoordinate'])
    upperCoordinate = torch.Tensor(config['upperCoordinate'])

    if not verboseMultiHorizon:
        plotProjectionsOfHigherDims = False

    if finalHorizon > 1 and performMultiStepSingleHorizon and\
            (normToUseLipschitz != 2 or not useSdpForLipschitzCalculation):
        raise ValueError

    if torch.cuda.is_available():
        device = torch.device("cuda", 0)
    else:
        device = torch.device("cpu")

    # Temporary
    device = torch.device("cpu")
    print(device)
    print(' ')

    # fileName = "randomNetwork.pth"
    # fileName = "randomNetwork2.pth"
    # fileName = "randomNetwork3.pth"
    # fileName = "trainedNetwork1.pth"
    # fileName = "doubleIntegrator.pth"
    # fileName = "doubleIntegrator_reachlp.pth"
    # fileName = "quadRotor5.pth"
    # fileName = "quadRotorv2.0.pth"
    # fileName = "RobotArmStateDict2-50-2.pth"
    # fileName = "Test3-5-3.pth"
    # fileName = "ACASXU.pth"
    # fileName = "mnist_3_50.pth"
    # fileName = "quadRotorFullLoopV1.8.pth"
    fileName = "quadRotorNormalV1.2.pth"

    lowerCoordinate = lowerCoordinate.to(device)
    upperCoordinate = upperCoordinate.to(device)

    network = NeuralNetwork(pathToStateDictionary, A, B, c)
    horizonForLipschitz = 1
    originalNetwork = None
    if performMultiStepSingleHorizon:
        originalNetwork = copy.deepcopy(network)
        horizonForLipschitz = finalHorizon
        network.setRepetition(finalHorizon)
        # repeatNetwork(network, finalHorizon)
        finalHorizon = 1

    dim = network.Linear[0].weight.shape[1]
    outputDim = network.Linear[-1].weight.shape[0]
    network.to(device)

    if dim < 3:
        plotProjectionsOfHigherDims = False

    plottingData = {}

    inputData = (upperCoordinate - lowerCoordinate) * torch.rand(1000, dim, device=device) \
                                                        + lowerCoordinate
    if verboseMultiHorizon:
        # fig = plt.figure()
        fig, ax = plt.subplots()
        if "robotarm" not in configFileToLoad.lower():
            plt.scatter(inputData[:, 0], inputData[:, 1], marker='.', label='Initial', alpha=0.5)
    plottingData[0] = {"exactSet": inputData}

    startTime = time.time()

    for iteration in range(finalHorizon):
        inputDataVariable = Variable(inputData, requires_grad=False)
        with no_grad():
            imageData = network.forward(inputDataVariable)
        plottingData[iteration + 1] = {"exactSet": imageData}
        pcaDirections, data_comp, data_mean, inputData = calculateDirectionsOfOptimization(onlyPcaDirections, imageData)

        if verboseMultiHorizon:
            # plt.figure()
            plt.scatter(imageData[:, 0], imageData[:, 1], marker='.', label='Horizon ' + str(iteration + 1), alpha=0.5)


        numberOfInitialDirections = len(pcaDirections)
        indexToStartReadingBoundsForPlotting = 0
        plottingDirections = pcaDirections
        if plotProjectionsOfHigherDims:
            indexToStartReadingBoundsForPlotting = calculateDirectionsOfHigherDimProjections(pcaDirections, imageData)

        plottingData[iteration + 1]["A"] = pcaDirections
        plottingConstants = np.zeros((len(pcaDirections), 1))
        plottingData[iteration + 1]['d'] = plottingConstants
        pcaDirections = torch.Tensor(np.array(pcaDirections))
        calculatedLowerBoundsforpcaDirections = torch.Tensor(np.zeros(len(pcaDirections)))

        solveSingleStepReachability(pcaDirections, imageData, config, iteration, device, network,
                                    plottingConstants, calculatedLowerBoundsforpcaDirections,
                                    originalNetwork, horizonForLipschitz, lowerCoordinate, upperCoordinate)

        if finalHorizon > 1:
            rotation = nn.Linear(dim, dim)
            rotation.weight = torch.nn.parameter.Parameter(torch.linalg.inv(torch.from_numpy(data_comp).float().to(device)))
            rotation.bias = torch.nn.parameter.Parameter(torch.from_numpy(data_mean).float().to(device))
            network.rotation = rotation

            centers = []
            for i, component in enumerate(data_comp):
                u = -calculatedLowerBoundsforpcaDirections[2 * i]
                l = calculatedLowerBoundsforpcaDirections[2 * i + 1]
                # center = (u + l) / 2
                center = component @ data_mean
                centers.append(center)
                upperCoordinate[i] = u - center
                lowerCoordinate[i] = l - center

        if verboseMultiHorizon:
            AA = -np.array(pcaDirections[indexToStartReadingBoundsForPlotting:])
            AA = AA[:, :2]
            bb = []
            for i in range(indexToStartReadingBoundsForPlotting, len(calculatedLowerBoundsforpcaDirections)):
                bb.append(-calculatedLowerBoundsforpcaDirections[i])

            bb = np.array(bb)
            pltp = polytope.Polytope(AA, bb)
            ax = pltp.plot(ax, alpha = 0.1, color='grey', edgecolor='black')
            ax.set_xlim([0, 5])
            ax.set_ylim([-4, 5])

            plt.axis("equal")
            if fileName != "RobotArmStateDict2-50-2.pth":
                leg1 = plt.legend()
            plt.xlabel('$x_0$')
            plt.ylabel('$x_1$')

    if verboseMultiHorizon:
        if fileName == "doubleIntegrator_reachlp.pth":
            reachlp = np.array([
                # [[2.5, 3], [-0.25, 0.25]],
            [[ 1.90837383, 2.75 ],
            [-1.125, -0.70422709]],

            [[1.0081799, 1.8305043],
            [-1.10589671, -0.80364925]],

            [[ 0.33328745,  0.94537741],
            [-0.76938218, -0.41314635]],

            [[-0.06750171, 0.46302059],
            [-0.47266394, -0.07047667]],

            [[-0.32873616,  0.38155359],
            [-0.30535603,  0.09282264]]
            ])
            plottingData["reachlp"] = reachlp
            for i in range(len(reachlp)):
                currHorizon = reachlp[i]
                rectangle = patches.Rectangle((currHorizon[0][0], currHorizon[1][0]),
                                currHorizon[0][1] - currHorizon[0][0],
                                currHorizon[1][1] - currHorizon[1][0],
                                edgecolor='b', facecolor='none', linewidth=2, alpha=1)
                x = ax.add_patch(rectangle)

            custom_lines = [Line2D([0], [0], color='b', lw=2),
                                Line2D([0], [0], color='red', lw=2, linestyle='--')]
            ax.legend(custom_lines, ['ReachLP', 'ReachLipSDP'], loc=4)
            

        # plt.gca().add_artist(leg1)
        plt.savefig("reachabilityPics/" + fileName + "Iteration" + str(iteration) + ".png")
        # plt.show()

    endTime = time.time()

    print('The algorithm took (s):', endTime - startTime, 'with eps =', eps)

    torch.save(plottingData, "Output/reachLip" + fileName)
    return endTime - startTime


if __name__ == '__main__':
    runTimes = []
    for i in range(1):
        runTimes.append(main())
    print(np.mean(runTimes))
    plt.show()