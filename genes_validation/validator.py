import csv
import numpy

class Validator(object):

    def __init__(self, data_file):

        self.data_file = data_file

        #self.mean_bound = -0.812928571429
        # self.pearson_bound = 0.8
        #self.getMeanBound()

        self.variance_bound = 0.101202882653

        #self.variance_bound = self.getVarianceBound()

    # def getVarianceBound(self):
    #     result = []
    #     with open(self.data_file, "rb") as f:
    #         reader = csv.reader(f, delimiter=',')
    #         next(reader)
    #         for line in reader:
    #             vector = self.getGeneCsProfileVector(line)
    #             result.append(self.coumputeVarianceOfVector(vector))
    #         result.sort()
    #         bound = result[-2000]
    #     return bound

    def getGeneCsProfileVector(self, line):
        vector = line[1:]
        return [float(i) for i in vector]

    # def getMeanBound(self):
    #     result = []
    #     with open(self.data_file, "r+") as f:
    #         reader = csv.reader(f, delimiter=',')
    #         next(reader)
    #         lines = 0
    #         for line in reader:
    #             vector = self.getGeneCsProfileVector(line)
    #             mean = self.coumputeMeanOfVector(vector)
    #             result.append(mean)
    #             lines += 1
    #         result.sort()
    #         amountToDelete = (15.0/100.0) * lines
    #         #increasingly
    #         bound = result[int(amountToDelete)]
    #     return bound

    def coumputeVarianceOfVector(self, vector):
        return numpy.var(vector)

    # def coumputeMeanOfVector(self, vector):
    #     return numpy.mean(vector)

    # def isSecondLowerCsSmallerThenMinusOne(self, vector):
    #     vector.sort()
    #     if vector[1] < -1:
    #         return True
    #     return False

    def isLineValid(self, line):
        vector = self.getGeneCsProfileVector(line)
        variance = self.coumputeVarianceOfVector(vector)
        bool1 = variance > self.variance_bound
        return bool1


    def eliminateLines(self):
        with open(self.data_file, 'rb') as infile, open('genes_data_after_second_elimination.csv', 'wb') as outfile:
            writer = csv.writer(outfile)
            reader = csv.reader(infile, delimiter=',')
            for line in reader:
                if self.isLineValid(line):
                    writer.writerow(line)

if __name__ == '__main__':
    validator = Validator("genes_data_after_first_elimination.csv")
    validator.eliminateLines()
