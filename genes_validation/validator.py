
class Validator(object):

    def __init__(self, data_file):
        self.data_file = data_file
        self.pearson_bound = 0.8
        self.mean_bound = 0.1
        self.variance_bound = 0.2

    def getGeneCsProfileVector(self, gene):
        return (0,0,0,0,0)

    def coumputeVarianceOfVector(self, vector):
        return 0

    def coumputeMeanOfVector(self, vector):
        return 0

    def isGeneValid(self, gene):
        vector = self.getGeneCsProfileVector(gene)
        mean = self.coumputeMeanOfVector(vector)
        variance = self.coumputeVarianceOfVector(vector)
        if mean > self.mean_bound and variance > self.variance_bound:
            return True
        return False


