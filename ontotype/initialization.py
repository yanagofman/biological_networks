from ontotype_classes import Go
import pickle
import csv

def parseLine(line):
    row = line.split(",")
    geneId = row[0]
    goNumber = row[1]
    goName = row[2]
    return (geneId, goNumber, goName)


def isLineValid(line):
    row = line.split(",")
    if len(row) == 3:
        geneId = row[0]
        goNumber = row[1]
        goName = row[2]
        if (geneId != '' and geneId != '\n'
            and goNumber != '' and goNumber != '\n'
            and goName != '' and goName != '\n'):
            return True
        else:
            return False
    else:
        return False


def parseGoName(goName):
    if goName.endswith('\n'):
        return goName[0:len(goName) - 1]
    return goName

def createInitializationMap(goDataFile):
    initializationMap = dict()
    firstLine = True
    for line in open(goDataFile):
        if (firstLine):
            firstLine = False
            continue
        if (isLineValid(line)):
            (geneId, goNumber, goName) = parseLine(line)
            goName = parseGoName(goName)
            go = Go(goNumber, goName)
            if geneId in initializationMap:
                if go not in initializationMap[geneId]:
                    initializationMap[geneId].append(go)
            else:
                initializationMap[geneId] = [go]
    file_Name = "initializationMap.pkl"
    initializationMapFileObj = open(file_Name, 'wb')
    pickle.dump(initializationMap, initializationMapFileObj)
    initializationMapFileObj.close()
    print 'did it'


#addition
def getAllGenes(fileName):
    result = set()
    with open(fileName, "r") as f:
        reader = csv.reader(f, delimiter=',')
        next(reader)
        for row in reader:
            result.add(row[0])
    return result
