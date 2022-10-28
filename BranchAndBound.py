from packages import *
from Utilities.Plotter import Plotter
from BranchAndBoundNode import BB_node
from Bounding.LipschitzBound import LipschitzBounding
from Bounding.PgdUpperBound import PgdUpperBound
from Utilities.Timer import Timers
from Bounding.DeepPolyBounding import DeepPolyBounding


class BranchAndBound:
    def __init__(self, coordUp, coordLow, config,
                 inputDimension=2, network=None, queryCoefficient=None, currDim = 0,
                 device=torch.device("cuda", 0),
                 maximumBatchSize=1024,
                 initialBub=None,
                 lowerBoundExtraInfo=None
                 ):

        self.spaceNodes = [BB_node(np.infty, -np.infty, coordUp, coordLow, scoreFunction=config['scoreFunction'])]
        self.bestUpperBound = initialBub
        self.bestLowerBound = None
        self.initCoordUp = coordUp
        self.initCoordLow = coordLow
        self.verbose = config['verbose']
        self.verboseEssential = config['verboseEssential']
        self.pgdIterNum = config['pgdIterNum']
        self.pgdNumberOfInitializations = config['pgdNumberOfInitializations']
        self.inputDimension = inputDimension
        self.eps = config['eps']
        self.network = network
        self.currDim = currDim
        self.queryCoefficient = queryCoefficient
        self.lowerBoundMethod = config['lowerBoundMethod']
        if config['lowerBoundMethod'] == "lipschitz":
            self.lowerBoundClass = LipschitzBounding(network, config, device=device, extraInfo=lowerBoundExtraInfo)
        elif config['lowerBoundMethod'] == "deepPoly":
            self.lowerBoundClass = DeepPolyBounding(network, device)
        self.upperBoundClass = PgdUpperBound(network, config['pgdNumberOfInitializations'],
                                             config['pgdIterNum'], config['pgdStepSize'],
                                             inputDimension, device, maximumBatchSize)
        self.nodeBranchingFactor = config['nodeBranchingFactor']
        self.scoreFunction = config['scoreFunction']
        self.branchNodeNum = config['branchNodeNum']
        self.device = device
        self.maximumBatchSize = 1024
        self.initialGD = config['initialGD']
        self.timers = Timers(["lowerBound",
                              "lowerBound:lipschitzForwardPass", "lowerBound:lipschitzCalc",
                              "lowerBound:lipschitzSearch",
                              "lowerBound:virtualBranchPreparation", "lowerBound:virtualBranchMin",
                              "upperBound",
                              "bestBound",
                              "branch", "branch:prune", "branch:maxFind", "branch:nodeCreation",
                              ])
        self.numberOfBranches = 0
        self.spaceOutThreshold = 40000

    def prune(self):
        for i in range(len(self.spaceNodes) - 1, -1, -1):
            if self.spaceNodes[i].lower > self.bestUpperBound:
                self.spaceNodes.pop(i)

    def lowerBound(self, indices):
        lowerBounds = torch.vstack([self.spaceNodes[index].coordLower for index in indices])
        upperBounds = torch.vstack([self.spaceNodes[index].coordUpper for index in indices])
        return self.lowerBoundClass.lowerBound(self.queryCoefficient, lowerBounds, upperBounds, timer=self.timers)

    def upperBound(self, indices):
        return self.upperBoundClass.upperBound(indices, self.spaceNodes, self.queryCoefficient)

    def branch(self):
        # Prunning Function
        self.timers.start("branch:prune")
        self.prune()
        self.timers.pause("branch:prune")
        numNodesAfterPrune = len(self.spaceNodes)


        self.timers.start("branch:maxFind")
        scoreArray = torch.Tensor([self.spaceNodes[i].score for i in range(len(self.spaceNodes))])
        scoreArraySorted = torch.argsort(scoreArray)
        if len(self.spaceNodes) > self.branchNodeNum:
            maxIndices = scoreArraySorted[len(scoreArraySorted) - self.branchNodeNum : len(scoreArraySorted)]
        else:
            maxIndices = scoreArraySorted[:]

        deletedUpperBounds = []
        deletedLowerBounds = []
        nodes = []
        maxIndices, __ = torch.sort(maxIndices, descending=True)
        for maxIndex in maxIndices:
            node = self.spaceNodes.pop(maxIndex)
            nodes.append(node)
            for i in range(self.nodeBranchingFactor):
                deletedUpperBounds.append(node.upper)
                deletedLowerBounds.append(node.lower)
        deletedLowerBounds = torch.Tensor(deletedLowerBounds).to(self.device)
        deletedUpperBounds = torch.Tensor(deletedUpperBounds).to(self.device)
        self.timers.pause("branch:maxFind")
        for j in range(len(nodes) - 1, -1, -1):
            self.timers.start("branch:nodeCreation")
            coordToSplitSorted = torch.argsort(nodes[j].coordUpper - nodes[j].coordLower)
            coordToSplit = coordToSplitSorted[len(coordToSplitSorted) - 1]
            node = nodes[j]
            parentNodeUpperBound = node.coordUpper
            parentNodeLowerBound = node.coordLower

            newIntervals = torch.linspace(parentNodeLowerBound[coordToSplit],
                                                    parentNodeUpperBound[coordToSplit],
                                                    self.nodeBranchingFactor + 1)
            for i in range(self.nodeBranchingFactor):
                tempLow = parentNodeLowerBound.clone()
                tempHigh = parentNodeUpperBound.clone()

                tempLow[coordToSplit] = newIntervals[i]
                tempHigh[coordToSplit] = newIntervals[i+1]
                self.spaceNodes.append(BB_node(np.infty, -np.infty, tempHigh, tempLow, scoreFunction=self.scoreFunction))

                if torch.any(tempHigh - tempLow < 1e-8):
                    self.spaceNodes[-1].score = -1
            self.timers.pause("branch:nodeCreation")
        
        numNodesAfterBranch = len(self.spaceNodes)
        numNodesAdded = numNodesAfterBranch - numNodesAfterPrune + len(maxIndices)
        self.numberOfBranches += numNodesAdded

        return [len(self.spaceNodes) - j for j in range(1, numNodesAdded + 1)], deletedUpperBounds, deletedLowerBounds

    def bound(self, indices, parent_lb):
        self.timers.start("lowerBound")
        lowerBounds = torch.maximum(self.lowerBound(indices), parent_lb)
        self.timers.pause("lowerBound")
        self.timers.start("upperBound")
        upperBounds = self.upperBound(indices)
        self.timers.pause("upperBound")
        for i, index in enumerate(indices):
            self.spaceNodes[index].upper = upperBounds[i]
            self.spaceNodes[index].lower = lowerBounds[i]

    def run(self):
        if self.initialGD:
            initUpperBoundClass = PgdUpperBound(self.network, 10, 1000, 0.001,
                                                self.inputDimension, self.device, self.maximumBatchSize)

            if self.bestUpperBound:
                self.bestUpperBound =\
                    torch.minimum(self.bestUpperBound,
                                  torch.Tensor(initUpperBoundClass.upperBound([0], self.spaceNodes,
                                                                              self.queryCoefficient)))
            if self.verboseEssential:
                print(self.bestUpperBound)
        elif self.bestUpperBound is None:
            self.bestUpperBound = torch.Tensor([torch.inf]).to(self.device)
        self.bestLowerBound = torch.Tensor([-torch.inf]).to(self.device)

        if self.verbose:
            plotter = Plotter()
        self.bound([0], self.bestLowerBound)
        if self.scoreFunction in ["worstLowerBound", "bestLowerBound", "bestUpperBound", "worstUpperBound",
                                  "averageBounds", "weightedGap"]:
            self.spaceNodes[0].score = self.spaceNodes[0].calc_score()

        while self.bestUpperBound - self.bestLowerBound >= self.eps:
            if len(self.spaceNodes) > self.spaceOutThreshold:
                break
            if self.verboseEssential:
                print(len(self.spaceNodes))
            self.timers.start("branch")
            indices, deletedUb, deletedLb = self.branch()
            self.timers.pause("branch")
            self.bound(indices, deletedLb)

            if self.scoreFunction in ["worstLowerBound", "bestLowerBound", "bestUpperBound", "worstUpperBound",
                                      "averageBounds", "weightedGap"]:
                minimumIndex = len(self.spaceNodes) - self.branchNodeNum * self.nodeBranchingFactor
                if minimumIndex < 0:
                    minimumIndex = 0
                maximumIndex = len(self.spaceNodes)
                for i in range(minimumIndex, maximumIndex):
                    self.spaceNodes[i].score = self.spaceNodes[i].calc_score()

            self.timers.start("bestBound")

            self.bestUpperBound =\
                torch.minimum(self.bestUpperBound,
                              torch.min(torch.Tensor([self.spaceNodes[i].upper for i in range(len(self.spaceNodes))])))
            self.bestLowerBound = torch.min(
                torch.Tensor([self.spaceNodes[i].lower for i in range(len(self.spaceNodes))]))
            self.timers.pause("bestBound")
            if self.verboseEssential:
                print('Best LB', self.bestLowerBound, 'Best UB', self.bestUpperBound, "diff", self.bestUpperBound - self.bestLowerBound)

            if self.verbose:
                print('Best LB', self.bestLowerBound, 'Best UB', self.bestUpperBound)
                plotter.plotSpace(self.spaceNodes, self.initCoordLow, self.initCoordUp)
                print('----------' * 10)

        if self.verbose:
            print("Number of created nodes: {}".format(self.numberOfBranches))
            plotter.showAnimation(self.spaceNodes, self.currDim)
        self.timers.pauseAll()
        if self.verboseEssential:
            self.timers.print()
            if self.lowerBoundMethod == "lipschitz":
                print(self.lowerBoundClass.calculatedLipschitzConstants)
                print("number of calculated lipschitz constants ", len(self.lowerBoundClass.calculatedLipschitzConstants))

        return self.bestLowerBound, self.bestUpperBound, self.spaceNodes

    def __repr__(self):
        string = 'These are the remaining nodes: \n'
        for i in range(len(self.spaceNodes)):
            string += self.spaceNodes[i].__repr__() 
            string += "\n"

        return string


        