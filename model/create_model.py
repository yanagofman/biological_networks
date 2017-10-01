import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, auc
from sklearn.svm import SVC


# Class to test the Random Forest
class RF(object):
    # creates a new object of class, chooses training data and builds predictor
    def __init__(self, data, labels, train_size=900):
        self.dt = np.array(data)
        self.lbs = np.array(labels)
        self.size = np.size(self.lbs)
        self.classifiar = RandomForestClassifier(n_jobs = -1)
        indexes = list(range(self.size))
        np.random.shuffle(indexes)
        self.classifiar.fit(self.dt[indexes[:train_size]], self.lbs[indexes[:train_size]])

    # Gets size, creates n times a classifier for data, prints average stats
    def accuracy(self, trainsize, show=True, n=1):
        testsize = self.size - trainsize
        TP = 0
        TN = 0
        T = 0

        clf = RandomForestClassifier(n_jobs = -1)
        train_acc = 0
        indexes = list(range(self.size))
        for i in range(n):
            np.random.shuffle(indexes)
            train_data = self.dt[indexes[:trainsize]]
            train_labels = self.lbs[indexes[:trainsize]]
            test_data = self.dt[indexes[trainsize:]]
            test_labels = self.lbs[indexes[trainsize:]]
            clf.fit(train_data, train_labels)
            prediction = clf.predict(test_data)
            cor_true = np.sum(prediction == test_labels)
            T += cor_true / testsize
            tpr = np.sum(prediction + test_labels == 2)
            tnr = np.sum(prediction + test_labels == 0)
            if tpr != 0:
                TP += tpr / np.sum(test_labels == 1)
            if tnr != 0:
                TN += tnr / np.sum(test_labels == 0)
            prediction = clf.predict(train_data)
            train_acc += np.sum(prediction == train_labels) / trainsize
        T /= n
        TN /= n
        TP /= n
        train_acc /= n
        if show:
            print("average accuracy percentage on test data = " + str(T))
            print("average true positive percentage on test data = " + str(TP))
            print("average true negative percentage on test data = " + str(TN))
            print("average accuracy percentage on train data = " + str(train_acc))
            plt.figure(1)  # the first figure
            plt.subplot(211)
            plt.ylabel("Percentage")
            plt.bar([0, 1, 2, 3], [T, TP, TN, train_acc],
                    tick_label=["Avarage accuracy\non test data", "Avarage true positive\npercantage on test data",
                                "Avarage false positive\npercantage on test data", "Avarage accuracy\non train data"])
            plt.savefig("RF accuracy.png")
            plt.show()
        return [T, TP, TN, train_acc]

    # Get sizes list and make accuracies graph depended on those train sizes
    def accur_graphs(self, sizes, n = 1):
        vals = list()
        T = list()
        TP = list()
        TN = list()
        train = list()
        sizes = sorted(sizes)
        for i in sizes:
            ret = self.accuracy(i, False, n)
            T += [ret[0]]
            TP += [ret[1]]
            TN += [ret[2]]
            train += [ret[3]]
        plt.figure(1)
        plt.subplot(211)
        plt.ylabel("Percentage")
        plt.xlabel("Train data set size")
        acc_gr, = plt.plot(sizes, T, "b-", label="Average accuracy(test)")
        tp_gr, = plt.plot(sizes, TP, "g-", label="Average true positive accuracy(test)")
        tn_gr, = plt.plot(sizes, TN, "r-", label="Average true negative accuracy(test) ")
        train_gr, = plt.plot(sizes, train, "y-", label="Average accuracy(train)")
        plt.legend(handles=[acc_gr, tp_gr, tn_gr, train_gr])
        plt.legend(bbox_to_anchor=(0, -0.95), loc=2, borderaxespad=0.)
        plt.savefig("RF accuracy graph.png")
        plt.show()

    # Makes the Roc graph for classifier, using different trainsizes, each n times
    def ROC(self, sizes, n = 1):
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
        for size in sizes:
            for j in range(n):
                np.random.shuffle(indexes)
                clf.fit(self.dt[indexes[:size]], self.lbs[indexes[:size]])
                testLbs = self.lbs[indexes[size:]]
                testPre = clf.predict(self.dt[indexes[size:]])
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
        plt.savefig("RF ROC.png")
        plt.show()
        return fprs, tprs, auc(fprs, tprs)

     def pre_recall(sizes,n=2):
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
        for size in sizes:
            for j in range(n):
                np.random.shuffle(indexes)
                clf.fit(self.dt[indexes[:size]], self.lbs[indexes[:size]])
                testLbs = self.lbs[indexes[size:]]
                testPre = clf.predict(self.dt[indexes[size:]])
                tp = np.sum(testPre + testLbs == 2)
                re = tp/(np.sum(testLbs == 1))
                pre = tp/(np.sum(testPre == 1))
                if re in dat:
                    dat[re] += [pre, 1]
                else:
                    dat[re] = np.array([pre, 1])
        fprs = sorted(list(dat.keys()))
        tprs = []
        for i in range(len(fprs)):
            tprs += [dat[fprs[i]][0] / dat[fprs[i]][1]]
        fprs = [0] + fprs + [1]
        tprs = [1] + tprs + [0]
        plt.plot(fprs, tprs, 'b-')
        plt.savefig("RF precision-recall curve.png")
        plt.show()
        
    def re_class(self, train_size=900):
        indexes = list(range(self.size))
        np.random.shuffle(indexes)
        self.classifiar.fit(self.dt[indexes[:train_size]], self.lbs[indexes[:train_size]])

    def predict(self, x):
        return self.classifiar.predict(x)


class SVM(object):
    # creates a new object of class, chooses training data and builds predictor
    def __init__(self, data, labels, trainSize=900, ker='linear'):
        self.dt = np.array(data)
        self.lbs = np.array(labels)
        self.size = np.size(self.lbs)
        self.classifiar = SVC(kernel=ker)
        self.kernel = ker
        indexes = list(range(self.size))
        np.random.shuffle(indexes)
        self.classifiar.fit(self.dt[indexes[:trainSize]], self.lbs[indexes[:trainSize]])

    # Gets size, creates n times a classifier for data, prints average stats
    def accuracy(self, trainsize, show=True, n=10):
        testsize = self.size - trainsize
        TP = 0
        TN = 0
        T = 0
        clf = SVC(kernel=self.kernel)
        train_acc = 0
        indexes = list(range(self.size))
        for i in range(n):
            np.random.shuffle(indexes)
            train_data = self.dt[indexes[:trainsize]]
            train_labels = self.lbs[indexes[:trainsize]]
            test_data = self.dt[indexes[trainsize:]]
            test_labels = self.lbs[indexes[trainsize:]]
            clf.fit(train_data, train_labels)
            prediction = clf.predict(test_data)
            cor_true = np.sum(prediction == test_labels)
            T += cor_true / testsize
            tpr = np.sum(prediction + test_labels == 2)
            tnr = np.sum(prediction + test_labels == 0)
            if tpr != 0:
                TP += tpr / np.sum(test_labels == 1)
            if tnr != 0:
                TN += tnr / np.sum(test_labels == 0)
            prediction = clf.predict(train_data)
            train_acc += np.sum(prediction == train_labels) / trainsize
        T /= n
        TN /= n
        TP /= n
        train_acc /= n
        if show:
            print("avarage accuracy percentage on test data = " + str(T))
            print("avarage true positive percentage on test data = " + str(TP))
            print("avarage true negative percentage on test data = " + str(TN))
            print("avarage accuracy percentage on train data = " + str(train_acc))
            plt.figure(1)
            plt.subplot(211)
            plt.ylabel("Percentage")
            plt.bar([0, 1, 2, 3], [T, TP, TN, train_acc],
                    tick_label=["Avarage accuracy\non test data", "Avarage true positive\npercantage on test data",
                                "Avarage false positive\npercantage on test data", "Avarage accuracy\non train data"])
            plt.savefig("SVM accuracy.png")
            plt.show()
        return [T, TP, TN, train_acc]

    # Get sizes list and make accuracies graph depended on those train sizes
    def accur_graphs(self, sizes):
        T = list()
        TP = list()
        TN = list()
        train = list()
        sizes = sorted(sizes)
        for i in sizes:
            ret = self.accuracy(i, False)
            T += [ret[0]]
            TP += [ret[1]]
            TN += [ret[2]]
            train += [ret[3]]
        plt.figure(1)
        plt.subplot(211)
        plt.ylabel("Percentage")
        plt.xlabel("train data set size")
        acc_gr, = plt.plot(sizes, T, "b-", label="Average accuracy(test)")
        tp_gr, = plt.plot(sizes, TP, "g-", label="Average true positive accuracy(test)")
        tn_gr, = plt.plot(sizes, TN, "r-", label="Average true negative accuracy(test) ")
        train_gr, = plt.plot(sizes, train, "y-", label="Average accuracy(train)")
        plt.legend(handles=[acc_gr, tp_gr, tn_gr, train_gr])
        plt.legend(bbox_to_anchor=(0, -0.95), loc=2, borderaxespad=0.)
        plt.savefig("SVM accuracy graph.png")
        plt.show()

    # Makes the Roc graph for classifier, using different trainsizes, each n times
    def ROC(self, sizes):
        plt.figure(1)
        plt.subplot(211)
        plt.ylim(0, 1)
        plt.xlim(0, 1)
        plt.ylabel("True Positive rate")
        plt.xlabel("False Positive Rate")
        plt.plot([0, 1], [0, 1], 'r-')
        clf = SVC(kernel=self.kernel)
        indexes = list(range(self.size))
        dat = dict()
        for size in sizes:
            for j in range(10):

                np.random.shuffle(indexes)
                clf.fit(self.dt[indexes[:size]], self.lbs[indexes[:size]])
                testLbs = self.lbs[indexes[size:]]
                testPre = clf.predict(self.dt[indexes[size:]])
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
        plt.savefig("SVM ROC.png")
        plt.show()
        return fprs, tprs, auc(fprs, tprs)

    def re_class(self, train_size=900):
        indexes = list(range(self.size))
        np.random.shuffle(indexes)
        self.classifiar.fit(self.dt[indexes[:train_size]], self.lbs[indexes[:train_size]])

    def predict(self, x):
        return self.classifiar.predict(x)
