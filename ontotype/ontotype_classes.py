import pickle

class GoData:
    def __init__(self, go, relatedGenesNumber):
        self.go = go
        self.relatedGenes = relatedGenesNumber

class Go:
    def __init__(self, goNumber, goName):
        self.goNumber = goNumber
        self.goName = goName

    def __eq__(self, other):
        return self.goNumber == other.goNumber

class Ontotype:

    def __init__(self):
        self.goDataFile = 'mart_export.txt'
        self.initializationMapFileName = 'initializationMap.pkl'
        self.loadInitializationMap()

    def getGoListForGeneId(self, geneId):
        return self.initializationMap.get(geneId)

    def createGoDataMap(self, geneIdList):
        geneIdUndouplicationsList = self.undouplicatingList(geneIdList)
        goDataMap = dict()
        for geneId in geneIdUndouplicationsList:
            if geneId not in self.initializationMap:
                continue
            goListForGene = self.getGoListForGeneId(geneId)
            for go in goListForGene:
                if go.goNumber in goDataMap:
                    goDataMap[go.goNumber].relatedGenes += 1
                else:
                    goDataMap[go.goNumber] = GoData(go, 1)
        self.goDataMap = goDataMap
        return goDataMap

    def undouplicatingList(self, inputList):
        output = []
        for x in inputList:
            if x not in output:
                output.append(x)
        return output

    def loadInitializationMap(self):
        initializationMap = open(self.initializationMapFileName,'r')
        self.initializationMap = pickle.load(initializationMap)

    def printGoDataMap(self, goDataMap):
        print 'ONTOTYPE:', len(goDataMap), 'Gos found'
        print
        for key, value in goDataMap.items():
            print '-----key:------'
            print key
            print '-----value:----'
            print 'goName: ', value.go.goName
            print 'goNumber: ', value.go.goNumber
            print 'related genes number: ', value.relatedGenes
            print
