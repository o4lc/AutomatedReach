import numpy as np


def calculateDirectionsToOptimize(baseDirections, addReluZeroPlanes, weights, addStandardBasisDirectionsAtEnd,
                                  removeSimilarDirections=False, similarityTolerance=0.005):
    directionsToOptimize = baseDirections.copy()
    for i, weight in enumerate(weights):
        directionsToOptimize = np.delete(directionsToOptimize, np.linalg.norm(directionsToOptimize, axis=1) == 0, 0)
        directionsToOptimize = directionsToOptimize / np.linalg.norm(directionsToOptimize, axis=1, keepdims=True)
        directionsToOptimize = directionsToOptimize @ np.linalg.pinv(weight)
        if weight.shape[0] > weight.shape[1]:
            uMatrix, _, _ = np.linalg.svd(weight)
            for j in range(weight.shape[1], weight.shape[0]):
                directionsToOptimize = np.vstack([directionsToOptimize, uMatrix[:, j].T, -uMatrix[:, j].T])
        if addReluZeroPlanes and i < len(weights) - 1:
            for j in range(weight.shape[0]):
                ei = np.zeros((1, weight.shape[0]))
                ei[0, j] = -1
                directionsToOptimize = np.vstack([directionsToOptimize, ei.copy()])

    if removeSimilarDirections:
        directionsToOptimize = removeSimilarRows(directionsToOptimize, similarityTolerance)

    directionsToOptimize = np.delete(directionsToOptimize, np.linalg.norm(directionsToOptimize, axis=1) == 0, 0)
    directionsToOptimize = directionsToOptimize / np.linalg.norm(directionsToOptimize, axis=1, keepdims=True)

    if addStandardBasisDirectionsAtEnd:
        outputBaseDirections, [upperBoundIndices, lowerBoundIndices] = createStandardBasisDirection(
            directionsToOptimize.shape[1], directionsToOptimize.shape[0])
        directionsToOptimize = np.vstack([directionsToOptimize, outputBaseDirections])

    if addStandardBasisDirectionsAtEnd:
        return directionsToOptimize, lowerBoundIndices, upperBoundIndices
    return directionsToOptimize


def calculateDirectionsOfMinkowskiSumLayer(firstMultiplierMatrix, firstPolytopeMatrix, secondMultiplierMatrix,
                                           secondPolytopeMatrix,
                                           addStandardBasisDirectionsAtEnd,
                                           removeSimilarDirections=False, similarityTolerance=0.005):
    """
    y = firstMultiplierMatrix * x1 + secondMultiplierMatrix * x2
    firstPolytopeMatrix * x1 <= d1
    secondPolytopeMatrix * x2 <= d2

    :param firstMultiplierMatrix:
    :param firstPolytopeMatrix:
    :param secondMultiplierMatrix:
    :param secondPolytopeMatrix:
    :return:
    """

    directions = np.vstack(
        [np.hstack([firstPolytopeMatrix, np.zeros((firstPolytopeMatrix.shape[0], secondPolytopeMatrix.shape[1]))]),
         np.hstack([np.zeros((secondPolytopeMatrix.shape[0], firstPolytopeMatrix.shape[1])),
                    secondPolytopeMatrix])]) @ np.linalg.pinv(
        np.hstack([firstMultiplierMatrix, secondMultiplierMatrix]))

    if removeSimilarDirections:
        directions = removeSimilarRows(directions, similarityTolerance)

    if addStandardBasisDirectionsAtEnd:
        outputBaseDirections, [upperBoundIndices, lowerBoundIndices] = createStandardBasisDirection(directions.shape[1],
                                                                                                    directions.shape[0])
        directions = np.vstack([directions, outputBaseDirections])

    directions = np.delete(directions, np.linalg.norm(directions, axis=1) == 0, 0)
    directions = directions / np.linalg.norm(directions, axis=1, keepdims=True)
    if addStandardBasisDirectionsAtEnd:
        return directions, lowerBoundIndices, upperBoundIndices
    return directions


def calculateDirectionsOfMinkowskiSum(firstMultiplierMatrix, firstPolytopeMatrix, secondMultiplierMatrix,
                                      secondPolytopeMatrix,
                                      addStandardBasisDirectionsAtEnd,
                                      removeSimilarDirections=False, similarityTolerance=0.005):
    """
    y = firstMultiplierMatrix * x1 + secondMultiplierMatrix * x2
    firstPolytopeMatrix * x1 <= d1
    secondPolytopeMatrix * x2 <= d2

    :param firstMultiplierMatrix:
    :param firstPolytopeMatrix:
    :param secondMultiplierMatrix:
    :param secondPolytopeMatrix:
    :return:
    """
    firstSetDirection = firstPolytopeMatrix @ np.linalg.pinv(firstMultiplierMatrix)
    if firstMultiplierMatrix.shape[0] > firstMultiplierMatrix.shape[1]:
        uMatrix, _, _ = np.linalg.svd(firstMultiplierMatrix)
        for j in range(firstMultiplierMatrix.shape[1], firstMultiplierMatrix.shape[0]):
            firstSetDirection = np.vstack([firstSetDirection, uMatrix[:, j].T, -uMatrix[:, j].T])
    secondSetDirection = secondPolytopeMatrix @ np.linalg.pinv(secondMultiplierMatrix)
    if secondMultiplierMatrix.shape[0] > secondMultiplierMatrix.shape[1]:
        uMatrix, _, _ = np.linalg.svd(secondMultiplierMatrix)
        for j in range(secondMultiplierMatrix.shape[1], secondMultiplierMatrix.shape[0]):
            secondSetDirection = np.vstack([secondSetDirection, uMatrix[:, j].T, -uMatrix[:, j].T])
    directions = np.vstack([firstSetDirection, secondSetDirection])

    if removeSimilarDirections:
        directions = removeSimilarRows(directions, similarityTolerance)

    if addStandardBasisDirectionsAtEnd:
        outputBaseDirections, [upperBoundIndices, lowerBoundIndices] = createStandardBasisDirection(directions.shape[1],
                                                                                                    directions.shape[0])
        directions = np.vstack([directions, outputBaseDirections])

    directions = np.delete(directions, np.linalg.norm(directions, axis=1) == 0, 0)
    directions = directions / np.linalg.norm(directions, axis=1, keepdims=True)
    if addStandardBasisDirectionsAtEnd:
        return directions, lowerBoundIndices, upperBoundIndices
    return directions


def createStandardBasisDirection(dimension, returnIndexOffset=0):
    positiveDirectionIndices = returnIndexOffset + np.arange(dimension)
    negativeDirectionIndices = returnIndexOffset + dimension + np.arange(dimension)

    baseDirections = np.zeros((2 * dimension, dimension))
    for i in range(dimension):
        baseDirections[i, i] = 1
    for i in range(dimension):
        baseDirections[dimension + i, i] = -1
    return baseDirections, [positiveDirectionIndices, negativeDirectionIndices]


def removeSimilarRows(directions, similarityTolerance=0.005):
    directions = np.delete(directions, np.linalg.norm(directions, axis=1) == 0, 0)
    directions = directions / np.linalg.norm(directions, axis=1, keepdims=True)
    indexesToDelete = []
    for i in range(directions.shape[0]):
        if i in indexesToDelete:
            continue
        for j in range(i + 1, directions.shape[0]):
            if j in indexesToDelete:
                continue
            if 1 - directions[i:i + 1, :] @ directions[j:j + 1, :].T <= similarityTolerance:
                indexesToDelete.append(j)
    directions = np.delete(directions, indexesToDelete, axis=0)
    return directions


def getDirectionsToOptimize(A, B, currentPolytope, Ws,
                            higherDimensionPlottingDirections=False,
                            firstPlottingDimension=0, secondPlottingDimension=1,
                            optimizationDirectionChoice="inverseMethod",
                            addReluZeroPlanes=True,
                            removeSimilarDirections=True,
                            similarityTolerance=0.001,
                            numberOfPlanesToOptimize=100,
                            addStandardBasisDirectionsAtEnd=True
                            ):
    higherDimensionD = []
    problemDimension = currentPolytope.shape[1]
    if optimizationDirectionChoice == "hyperCube":
        directionsToOptimize, [upperBoundIndices, lowerBoundIndices] = createStandardBasisDirection(
            currentPolytope.shape[1])
        if higherDimensionPlottingDirections:
            higherDimensionD = np.zeros((problemDimension, 4))
            higherDimensionD[0, firstPlottingDimension] = 1
            higherDimensionD[1, secondPlottingDimension] = 1
            higherDimensionD[2, firstPlottingDimension] = -1
            higherDimensionD[3, secondPlottingDimension] = -1
    elif optimizationDirectionChoice == "inverseMethod":
        directionsToOptimize = calculateDirectionsToOptimize(currentPolytope, addReluZeroPlanes, Ws, False,
                                                             removeSimilarDirections=removeSimilarDirections,
                                                             similarityTolerance=similarityTolerance)

        if addStandardBasisDirectionsAtEnd:
            directionsToOptimize, lowerBoundIndices, upperBoundIndices = \
                calculateDirectionsOfMinkowskiSum(A, currentPolytope, B, directionsToOptimize,
                                                       addStandardBasisDirectionsAtEnd,
                                                       removeSimilarDirections=removeSimilarDirections,
                                                       similarityTolerance=similarityTolerance)
        else:
            lowerBoundIndices = upperBoundIndices = None
            directionsToOptimize = \
                calculateDirectionsOfMinkowskiSum(A, currentPolytope, B, directionsToOptimize,
                                                       addStandardBasisDirectionsAtEnd,
                                                       removeSimilarDirections=removeSimilarDirections,
                                                       similarityTolerance=similarityTolerance)
        if higherDimensionPlottingDirections:
            # projection = np.zeros((2, problemDimension))
            # projection[0, firstPlottingDimension] = 1
            # projection[1, secondPlottingDimension] = 1
            # print((projection @ A).shape, currentPolytope.shape, (currentPolytope @ np.linalg.pinv(projection @ A)).shape)
            # higherDimensionD = calculateDirectionsOfMinkowskiSum(projection @ A, currentPolytope,
            #                                                      projection @ B, directionsToOptimize,
            #                                                      False,
            #                                                      removeSimilarDirections=removeSimilarDirections,
            #                                                      similarityTolerance=similarityTolerance)
            higherDimensionD = directionsToOptimize.copy()
            higherDimensionD[:, 2:] = 0
            higherDimensionD = removeSimilarRows(higherDimensionD, similarityTolerance)
    elif optimizationDirectionChoice == "uniform" and problemDimension == 2:
        directionsToOptimize = np.zeros((numberOfPlanesToOptimize, 2))
        assert numberOfPlanesToOptimize % 4 == 0  # or else the basis directions won't be correct
        for i in range(numberOfPlanesToOptimize):
            directionsToOptimize[i, :] = np.array([np.cos(2 * np.pi * i / numberOfPlanesToOptimize),
                                                   np.sin(2 * np.pi * i / numberOfPlanesToOptimize)])
        lowerBoundIndices = [int(numberOfPlanesToOptimize / 2), int(3 * numberOfPlanesToOptimize / 4)]
        upperBoundIndices = [0, int(numberOfPlanesToOptimize / 4)]
    else:
        raise ValueError
    return directionsToOptimize, [lowerBoundIndices, upperBoundIndices], higherDimensionD
