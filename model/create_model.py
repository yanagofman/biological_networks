import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor
from sklearn.metrics import roc_curve, auc
from sklearn.cross_validation import KFold


# Class to test the Random Forest
class RF_TESTS(object):
    # creates a new object of class, chooses training data and builds predictor
    def __init__(self, data, labels):
        self.dt = np.array(data)
        self.lbs = np.array(labels)
        self.size = np.size(self.lbs)
        

    # Runs the classifier with cross validation over num_of_folds folds of the data given on __init__
    #For each run, gets the roc results over test_data, and returns the roc result of all the runs together
    def clf_roc(self, num_of_folds=5):
        sizes = sorted(sizes)
        plt.figure(1)
        plt.subplot(211)
        plt.ylim(0, 1)
        plt.xlim(0, 1)
        plt.ylabel("True Positive rate")
        plt.xlabel("False Positive Rate")
        plt.plot([0, 1], [0, 1], 'r-')
        clf = RandomForestClassifier(n_jobs = -1)
        indexes = list(range(self.size))
        dat = dict()
        kf = KFold(n = self.size, n_folds = num_of_folds)
        for train_index,test_index  in kf:
            for j in range(n):
                np.random.shuffle(indexes)
                clf.fit(self.dt[train_index], self.lbs[train_index])
                testLbs = self.lbs[test_index]
                testPre = clf.predict(self.dt[test_index])
                fpr, tpr, thresh = roc_curve(testLbs, testPre, pos_label=1)
                if fpr[1] in dat:
                    dat[fpr[1]] += [tpr[1], 1]
                else:
                    dat[fpr[1]] = np.array([tpr[1], 1])
        fprs = sorted(list(dat.keys()))
        tprs = []
        for i in range(len(fprs)):
            tprs += [dat[fprs[i]][0] / dat[fprs[i]][1]]
        fprs = [0] + fprs + [1]
        tprs = [0] + tprs + [1]
        plt.plot(fprs, tprs, 'b-')
        plt.show()
        return fprs, tprs, auc(fprs, tprs)
    
    #gets average squared distance of the regressor's prediction from test labels, using cross validation over num_of_folds folds
    def reg_av_squard_distance(self,num_of_folds = 3):
        kf = KFold(n = self.size, n_folds = num_of_folds)
        reg = RandomForestRegressor(n_jobs = -1)
        ret = 0
        for train_index,test_index  in kf:
            reg.fit(self.dt[train_index], self.lbs[train_index])
            test_lbs = self.lbs[test_index]
            test_pre = reg.predict(self.dt[test_index])
            ret += np.linalg.norm(test_lbs-test_pre)**2
        return (ret/len(kf))
        
        