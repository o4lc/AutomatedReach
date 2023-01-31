import torch

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
                                originalNetwork=None, horizonForLipschitz=1, looseLowerBound=False,
                                lowerBoundIndices=None, upperBoundIndices=None):
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
        if looseLowerBound \
                or (lowerBoundIndices is not None and i in lowerBoundIndices) \
                or (upperBoundIndices is not None and i in upperBoundIndices):
            BB.spaceOutThreshold = 1
        lowerBound, upperBound, space_left = BB.run()
        plottingConstants[i] = -lowerBound
        calculatedLowerBoundsforpcaDirections[i] = lowerBound

        # print('Best lower/upper bounds are:', lowerBound, '->', upperBound)


def main():
    configBaseLocation = "Config/"
    configFileToLoad = "doubleIntegratorDeepPoly.json"
    # configFileToLoad = "quadRotorDeepPoly.json"
    # configFileToLoad = "fourDimDeepPoly.json"
    with open(configBaseLocation + configFileToLoad, 'r') as file:
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
    # device = torch.device("cpu", 0)
    print(device)
    print(' ')

    lowerCoordinate = lowerCoordinate.to(device)
    upperCoordinate = upperCoordinate.to(device)
    lowerCoordinates = [lowerCoordinate.detach().clone()]
    upperCoordinates = [upperCoordinate.detach().clone()]
    network = NeuralNetworkReachability(pathToStateDictionary, A, B, c,
                                        1, config['performMultiStepSingleHorizon'])
    originalWeights = network.originalWeights
    horizonForLipschitz = 1
    originalNetwork = None

    dim = network.Linear[0].weight.shape[1]
    network.to(device)
    if dim < 3:
        plotProjectionsOfHigherDims = False
    print(network)
    plottingData = {}
    inputData = (upperCoordinate - lowerCoordinate) * torch.rand(1000, dim, device=device) \
                                                        + lowerCoordinate
    if verboseMultiHorizon:
        # fig = plt.figure()
        fig, ax = plt.subplots()
        if "robotarm" not in configFileToLoad.lower() and "doubleIntegrator" not in configFileToLoad:
            plottingCoords = inputData.cpu().numpy()
            plt.scatter(plottingCoords[:, 0], plottingCoords[:, 1], marker='.', label='Initial', alpha=0.5)
    plottingData[0] = {"exactSet": inputData}

    startTime = time.time()

    initialPolytope = np.vstack([np.eye(dim), -np.eye(dim)])
    numberOfTotalDirections = 0
    for iteration in range(finalHorizon):
        # if iteration in [2, 4]:
        if iteration in range(5) and "doubleIntegrator" in configFileToLoad:
            fig, ax = plt.subplots()
        if iteration > 0:
            lowerCoordinates.append(lowerCoordinate.detach().clone())
            upperCoordinates.append(upperCoordinate.detach().clone())

            network = NeuralNetworkReachability(pathToStateDictionary,
                                                A, B, c,
                                                iteration + 1, config['performMultiStepSingleHorizon'])

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
                                              addStandardBasisDirectionsAtEnd=False,
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
        numberOfTotalDirections += directions.shape[0]
        solveSingleStepReachability(lowerCoordinates[0], upperCoordinates[0], directions, imageData, config, iteration, device, network,
                                    plottingConstants, calculatedLowerBoundsForDirections,
                                    originalNetwork, horizonForLipschitz,
                                    lowerBoundIndices=lowerBoundIndices, upperBoundIndices=upperBoundIndices)

        # if finalHorizon > 1:
            # for i, index in enumerate(lowerBoundIndices):
            #     upperCoordinate[i] = -calculatedLowerBoundsForDirections[index]
            # for i, index in enumerate(upperBoundIndices):
            #     lowerCoordinate[i] = calculatedLowerBoundsForDirections[index]
        if verboseMultiHorizon:
            AA = -np.array(directions[indexToStartReadingBoundsForPlotting:])
            AA = AA[:, :2]
            bb = []
            for i in range(indexToStartReadingBoundsForPlotting, len(calculatedLowerBoundsForDirections)):
                bb.append(-calculatedLowerBoundsForDirections[i])

            bb = np.array(bb)
            pltp = polytope.Polytope(AA, bb)
            ax = pltp.plot(ax, alpha = 1, color='None', edgecolor='red')


            if "doubleIntegrator" in configFileToLoad:
                reachlp = np.array([
                    # [[2.5, 3], [-0.25, 0.25]],
                    [[1.90837383, 2.75],
                     [-1.125, -0.70422709]],

                    [[1.0081799, 1.8305043],
                     [-1.10589671, -0.80364925]],

                    [[0.33328745, 0.94537741],
                     [-0.76938218, -0.41314635]],

                    [[-0.06750171, 0.46302059],
                     [-0.47266394, -0.07047667]],

                    [[-0.32873616, 0.38155359],
                     [-0.30535603, 0.09282264]]
                ])
                plottingData["reachlp"] = reachlp

                i = iteration
                currHorizon = reachlp[i]
                rectangle = patches.Rectangle((currHorizon[0][0], currHorizon[1][0]),
                                              currHorizon[0][1] - currHorizon[0][0],
                                              currHorizon[1][1] - currHorizon[1][0],
                                              edgecolor='b', facecolor='none', linewidth=2, alpha=1)
                x = ax.add_patch(rectangle)

                reachlp = np.array([
                 [[1.90837407,  2.75],
                   [-1.10949314, - 0.70422715]],
                 [[1.03360844,  1.83050406],
                   [-1.09321809, - 0.80526382]],
                 [[0.4081341,   0.91338146],
                   [-0.76728338, - 0.42616922]],
                 [[0.10232732,  0.35712636],
                   [-0.37809575, - 0.16485944]],
                 [[-0.03504939,  0.12998357],
                   [-0.1469591, - 0.03796272]]])

                currHorizon = reachlp[i]
                rectangle = patches.Rectangle((currHorizon[0][0], currHorizon[1][0]),
                                              currHorizon[0][1] - currHorizon[0][0],
                                              currHorizon[1][1] - currHorizon[1][0],
                                              edgecolor='g', facecolor='none', linewidth=2, alpha=1)
                x = ax.add_patch(rectangle)

                custom_lines = [Line2D([0], [0], color='b', lw=2),
                                Line2D([0], [0], color='g', lw=2),
                                Line2D([0], [0], color='red', lw=2, linestyle='--'),
                                Line2D([0], [0], color='black', lw=2, linestyle='--')]
                reachLipBnbPlottingData = torch.load("Output/reachLipdoubleIntegrator_reachlp.pth")

                AA = -np.array(reachLipBnbPlottingData[iteration + 1]["A"])
                AA = AA[:, :2]
                bb = []
                for i in range(len(reachLipBnbPlottingData[iteration + 1]['d'])):
                    bb.append(reachLipBnbPlottingData[iteration + 1]['d'])

                bb = np.array(bb)
                pltp = polytope.Polytope(AA, bb)
                ax = pltp.plot(ax, alpha=1, color='None', edgecolor='black')

                leg1 = plt.legend()
                ax.legend(custom_lines, ['ReachLP GSG', 'ReachLP Uniform', 'Our method', 'ReachLipBnB'], loc=4)
                plt.gca().add_artist(leg1)

            # ax.set_xlim([0, 5])
            # ax.set_ylim([-4, 5])

            # plt.axis("equal")
            # if fileName != "RobotArmStateDict2-50-2.pth":
            #     leg1 = plt.legend()
            plt.xlabel('$x_0$')
            plt.ylabel('$x_1$')
            # plt.show()
            plt.savefig("reachabilityPics/" + configFileToLoad[7:] + "Iteration" + str(iteration) + ".png")



    if verboseMultiHorizon:
        if "quadRotor" in configFileToLoad:
            reachlp = np.array([
             [[4.78338957,  4.80661011],
               [4.64838982 , 4.75161028],
            [2.9733901,
            3.02661014],
            [0.74101514 , 0.74490422],
            [-0.1412082, - 0.13706659],
            [-0.93384773, - 0.93015969]],
             [[4.85589075,  4.88270044],
               [4.63270712,  4.73946762],
            [2.87841702,
            2.93517137],
            [0.54030085 , 0.54791057],
            [-0.27151978, - 0.26335227],
            [-1.86192393, - 1.85471177]],
             [[4.9083209 ,  4.93909168],
               [4.60403156 , 4.7146616],
            [2.6906476,
            2.7512548],
            [0.34828603 , 0.35965163],
            [-0.39033422, - 0.37805188],
            [-2.78396106 ,- 2.77318573]],
             [[4.9415493,   4.97665691],
               [4.56351233 , 4.67835093],
            [2.41068554,
            2.4754684],
            [0.16548158,  0.18064192],
            [-0.49697527, - 0.48048425],
            [-3.69960403 ,- 3.68522429]],
             [[4.95649719 , 4.99632168],
               [4.51236582,  4.63176346],
            [2.03917027,
            2.1084559],
            [-0.00762022 , 0.01137753],
            [-0.59079105 ,- 0.56999248],
            [-4.60851097, - 4.59048223]],
             [[4.95413494 , 4.99905968],
               [4.45187473 , 4.5761919],
            [1.57677507,
            1.65089548],
            [-0.17054564, - 0.14766392],
            [-0.67115438, - 0.64594424],
            [-5.51035261, - 5.48862743]],
             [[4.93548059 , 4.98589325],
               [4.38338423  ,4.51299286],
            [1.02420616,
            1.10349882],
            [-0.32283968 ,- 0.29602349],
            [-0.7374627 ,- 0.7077319],
            [-6.40481329, - 6.37934113]],
             [[4.90159655 , 4.95789099],
               [4.30829906 , 4.44358301],
            [0.38220155,
            0.46700904],
            [-0.46406615 ,- 0.43326095],
            [-0.78913879 ,- 0.75477242],
            [-7.29158926 ,- 7.26231718]],
             [[4.85359001 , 4.91616488],
               [4.22808218  ,4.36943817],
            [-0.34847036 ,- 0.25780004],
            [-0.59380782 ,- 0.55895495],
            [-0.82563019 ,- 0.78650802],
            [-8.17039108 ,- 8.13726234]],
             [[4.79260921 , 4.86186981],
               [4.14425182 , 4.29208946],
            [-1.16701245 ,- 1.07012486],
            [-0.71166641 ,- 0.67270303],
            [-0.8464098, - 0.80240554],
            [-9.04094219 ,- 9.00389671]],
             [[4.71984243  ,4.7961998],
               [4.05837917 , 4.21312094],
            [-2.07260013 ,- 1.96913421],
            [-0.81726295 ,- 0.77412158],
            [-0.85097575, - 0.80195713],
            [-9.9029789 ,- 9.86195374]],
             [[4.63651609 ,  4.72038746],
               [3.972085  ,   4.13416815],
            [-3.06438184, - 2.95396948],
            [-0.91023737 ,- 0.86284614],
            [-0.83885151 ,- 0.78468013],
            [-10.75625229 ,- 10.71117973]]])
            plottingData["reachlp"] = reachlp
            for i in range(len(reachlp)):
                currHorizon = reachlp[i]
                rectangle = patches.Rectangle((currHorizon[0][0], currHorizon[1][0]),
                                              currHorizon[0][1] - currHorizon[0][0],
                                              currHorizon[1][1] - currHorizon[1][0],
                                              edgecolor='b', facecolor='none', linewidth=2, alpha=1)
                x = ax.add_patch(rectangle)

            reachLipBnbPlottingData = torch.load("Output/reachLipquadRotorv2.0.pth")
            indexToStartReadingBoundsForPlotting = 12
            for iteration in range(finalHorizon):
                AA = -np.array(reachLipBnbPlottingData[iteration + 1]["A"][indexToStartReadingBoundsForPlotting:])
                AA = AA[:, :2]
                bb = []
                for i in range(indexToStartReadingBoundsForPlotting, len(reachLipBnbPlottingData[iteration + 1]['d'])):
                    bb.append(reachLipBnbPlottingData[iteration + 1]['d'][i])

                bb = np.array(bb)
                pltp = polytope.Polytope(AA, bb)
                ax = pltp.plot(ax, alpha=1, color='None', edgecolor='black')


            custom_lines = [Line2D([0], [0], color='b', lw=2),
                            Line2D([0], [0], color='red', lw=2, linestyle='--'),
                            Line2D([0], [0], color='black', lw=2, linestyle='--')]
            plt.axis("equal")
            leg1 = plt.legend()
            ax.legend(custom_lines, ['ReachLP', 'Our method', 'ReachLipBnB'], loc=4)

            plt.gca().add_artist(leg1)

            

        # plt.gca().add_artist(leg1)

    plt.savefig("reachabilityPics/" + configFileToLoad[7:] + "Iteration" + str(iteration) + ".png")
    print(numberOfTotalDirections)

    endTime = time.time()

    print('The algorithm took (s):', endTime - startTime, 'with eps =', eps)

    torch.save(plottingData, "Output/reachDeep" + configFileToLoad[:-4] + "pth")
    return endTime - startTime


if __name__ == '__main__':
    runTimes = []
    for i in range(1):
        runTimes.append(main())
    print(np.mean(runTimes))
    plt.show()