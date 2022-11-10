from packages import *

from BranchAndBound import BranchAndBound
from NetworkModels.NeuralNetwork import NeuralNetwork
from sklearn.decomposition import PCA
import copy
import json
from NetworkModels.NeuralNetworkFullReachability import NeuralNetworkReachability
from Utilities.directionsOfOptimizationFunctions import getDirectionsToOptimize, createStandardBasisDirection

torch.set_printoptions(precision=20)
defaultDtype = torch.float64
torch.set_default_dtype(defaultDtype)


def calculateDirectionsOfOptimization(onlyPcaDirections, imageData):
    data_mean = 0
    inputData = None
    if onlyPcaDirections:
        pca = PCA()
        pcaData = pca.fit_transform(imageData)

        data_mean = pca.mean_
        data_comp = pca.components_
        data_sd = np.sqrt(pca.explained_variance_)

        inputData = torch.from_numpy(data_comp @ (imageData - data_mean).T).T.to(defaultDtype)
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
    projectedImageData = np.copy(imageData)
    projectedImageData[:, 2:] = 0
    pca2 = PCA()
    _ = pca2.fit_transform(projectedImageData)
    plottingDirections = pca2.components_
    for direction in plottingDirections[:2]:
        currentPcaDirections.append(-direction)
        currentPcaDirections.append(direction)
    return indexToStartReadingBoundsForPlotting


def solveSingleStepReachability(lowerCoordinate, upperCoordinate, pcaDirections, imageData, config, iteration, device, network,
                                plottingConstants, calculatedLowerBoundsforpcaDirections,
                                originalNetwork=None, horizonForLipschitz=1, looseLowerBound=False):
    dim = network.Linear[0].weight.shape[1]
    lowerBoundExtraInfo = {}
    for i in range(len(pcaDirections)):
        if config['lowerBoundMethod'] == "lipschitz":
            previousLipschitzCalculations = []
            if i % 2 == 1 and torch.allclose(pcaDirections[i], -pcaDirections[i - 1]):
                previousLipschitzCalculations = BB.lowerBoundClass.calculatedLipschitzConstants
            lowerBoundExtraInfo = {"calculatedLipschitzConstants": previousLipschitzCalculations,
                                   "horizon": horizonForLipschitz,
                                   "originalNetwork": originalNetwork}
        c = pcaDirections[i].to(device)
        if True:
            print('** Solving Horizon: ', iteration, 'dimension: ', i)
        initialBub = torch.min(imageData @ c)
        # print("initialBuB", initialBub)
        # initialBub = None
        BB = BranchAndBound(upperCoordinate, lowerCoordinate, config,
                            inputDimension=dim, network=network, queryCoefficient=c, currDim=i, device=device,
                            initialBub=initialBub,
                            lowerBoundExtraInfo=lowerBoundExtraInfo
                            )
        if looseLowerBound:
            BB.spaceOutThreshold = 0
        lowerBound, upperBound, space_left = BB.run()
        plottingConstants[i] = -lowerBound
        calculatedLowerBoundsforpcaDirections[i] = lowerBound

        # print('Best lower/upper bounds are:', lowerBound, '->', upperBound)


def main():
    configFileToLoad = "Config/doubleIntegratorDeepPoly.json"
    # configFileToLoad = "Config/quadRotorDeepPoly.json"
    # configFileToLoad = "Config/fourDimDeepPoly.json"
    with open(configFileToLoad, 'r') as file:
        config = json.load(file)
    # config['A'] = None
    # config['B'] = None
    # config['c'] = None
    lowerBoundMethod = config['lowerBoundMethod']
    eps = config['eps']
    verboseMultiHorizon = config['verboseMultiHorizon']

    finalHorizon = config['finalHorizon']
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
    if lowerBoundMethod == "lipschitz":
        if finalHorizon > 1 and config['performMultiStepSingleHorizon'] and\
                (config['normToUseLipschitz'] != 2 or not config['useSdpForLipschitzCalculation']):
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
    lowerCoordinates = [lowerCoordinate.detach().clone()]
    upperCoordinates = [upperCoordinate.detach().clone()]
    network = NeuralNetworkReachability(pathToStateDictionary, lowerCoordinate, upperCoordinate, A, B, c,
                                        1, config['performMultiStepSingleHorizon'])
    originalWeights = network.originalWeights
    horizonForLipschitz = 1
    originalNetwork = None

    dim = network.Linear[0].weight.shape[1]
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
            plottingCoords = inputData.cpu().numpy()
            plt.scatter(plottingCoords[:, 0], plottingCoords[:, 1], marker='.', label='Initial', alpha=0.5)
    plottingData[0] = {"exactSet": inputData}

    startTime = time.time()

    initialPolytope = np.vstack([np.eye(dim), -np.eye(dim)])

    for iteration in range(finalHorizon):
        if iteration > 0:
            lowerCoordinates.append(lowerCoordinate.detach().clone())
            upperCoordinates.append(upperCoordinate.detach().clone())
            layers = []
            for j in range(iteration + 1):
                layers += NeuralNetworkReachability(pathToStateDictionary, lowerCoordinates[j], upperCoordinates[j],
                                                    A, B, c,
                                                    1, config['performMultiStepSingleHorizon']).layers
            network = NeuralNetworkReachability(layers=layers)
        directions = initialPolytope

        with no_grad():
            imageData = network(inputData)
        imageDataCpu = imageData.detach().clone().cpu().numpy()
        plottingData[iteration + 1] = {"exactSet": imageDataCpu}
        indexToStartReadingBoundsForPlotting = 0
        if config['onlyPcaDirections']:
            directions, data_comp, data_mean, _ = calculateDirectionsOfOptimization(onlyPcaDirections,
                                                                                            imageDataCpu)
            baseDirections, (upperBoundIndices, lowerBoundIndices) = createStandardBasisDirection(dim, len(directions))
            directions = directions + baseDirections.tolist()
            if plotProjectionsOfHigherDims:
                indexToStartReadingBoundsForPlotting = calculateDirectionsOfHigherDimProjections(directions,
                                                                                                 imageDataCpu)
        else:
            for j in range(iteration + 1):
                directions, (lowerBoundIndices, upperBoundIndices), higherDimensionPlottingDirections \
                    = getDirectionsToOptimize(A.cpu().numpy(), B.cpu().numpy(), directions, originalWeights,
                                              higherDimensionPlottingDirections=plotProjectionsOfHigherDims,
                                              addStandardBasisDirectionsAtEnd=j == iteration,
                                              similarityTolerance=0.02)
            if plotProjectionsOfHigherDims:
                indexToStartReadingBoundsForPlotting = directions.shape[0]
                directions = np.vstack([directions, higherDimensionPlottingDirections])
                print(directions.shape, higherDimensionPlottingDirections.shape)

        if verboseMultiHorizon:
            # plt.figure()
            plt.scatter(imageDataCpu[:, 0], imageDataCpu[:, 1], marker='.', label='Horizon ' + str(iteration + 1), alpha=0.5)




        plottingData[iteration + 1]["A"] = directions
        plottingConstants = np.zeros((len(directions), 1))
        plottingData[iteration + 1]['d'] = plottingConstants
        directions = torch.Tensor(np.array(directions))
        calculatedLowerBoundsForDirections = torch.Tensor(np.zeros(len(directions)))

        solveSingleStepReachability(lowerCoordinates[0], upperCoordinates[0], directions, imageData, config, iteration, device, network,
                                    plottingConstants, calculatedLowerBoundsForDirections,
                                    originalNetwork, horizonForLipschitz)

        if finalHorizon > 1:
            for i, index in enumerate(lowerBoundIndices):
                upperCoordinate[i] = -calculatedLowerBoundsForDirections[index]
            for i, index in enumerate(upperBoundIndices):
                lowerCoordinate[i] = calculatedLowerBoundsForDirections[index]
        if verboseMultiHorizon:
            AA = -np.array(directions[indexToStartReadingBoundsForPlotting:])
            AA = AA[:, :2]
            bb = []
            for i in range(indexToStartReadingBoundsForPlotting, len(calculatedLowerBoundsForDirections)):
                bb.append(-calculatedLowerBoundsForDirections[i])

            bb = np.array(bb)
            pltp = polytope.Polytope(AA, bb)
            ax = pltp.plot(ax, alpha = 1, color='grey', edgecolor='red')
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