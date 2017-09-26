import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve,auc
from sklearn.svm import SVC
#Class to test the Random Forest 
class RF:
    #creates a new object of class, chooses training data and builds predictor
    def __init__(self,data, labels, trainSize=900):
        self.dt = np.array(data)
        self.lbs = np.array(labels)
        self.size = np.size(self.lbs)
        self.classifiar =  RandomForestClassifier()
        indexes = list(range(self.size))
        np.random.shuffle(indexes)
        self.classifiar.fit(self.dt[indexes[:trainSize]],self.lbs[indexes[:trainSize]])
    
    #Gets size, creates n times a classifier for data, prints average stats 
    def accuracy(self, trainsize, show = True, n = 10):
        testsize = self.size-trainsize
        TP = 0
        TN = 0
        T = 0
 
        clf = RandomForestClassifier()
        TrainAcc = 0
        indexes = list(range(self.size))
        for i in range(n):
            np.random.shuffle(indexes)
            train_data = self.dt[indexes[:trainsize]]
            train_labels = self.lbs[indexes[:trainsize]]
            test_data = self.dt[indexes[trainsize:]]
            test_labels = self.lbs[indexes[trainsize:]]
            clf.fit(train_data,train_labels)
            prediction = clf.predict(test_data)
            T += corTrue/testsize
            fpr,tpr,thresh = roc_curve(testLbs,testPre,pos_label = 1)
            TP += tpr[1]
            TN += (1-fpr[1])
            TrainAcc += np.sum(prediction == train_labels)/trainsize
        T /= n
        TN /= n
        TP /= n
        TrainAcc /= n
        if show:
            print( "avarage accuracy percentage on test data = "+str(T))
            print("avarage true positive percentage on test data = "+str(TP))
            print("avarage true negative percentage on test data = "+str(TN))
            print( "avarage accuracy percentage on train data = "+str(TrainAcc))
            plt.figure(1)                # the first figure
            plt.subplot(211)
            plt.ylabel("Percentage")
            plt.bar([0,1,2,3],[T,TP,TN,TrainAcc],tick_label=["Avarage accuracy\non test data","Avarage true positive\npercantage on test data","Avarage false positive\npercantage on test data","Avarage accuracy\non train data"])
            plt.show()
            plt.savefig("RF accuracy.png")
        return [T,TP,TN,TrainAcc]
    
    #Get sizes list and make accuracies graph depended on those train sizes
    def  accurGraphs(self, sizes):
        vals = list()
        T = list()
        TP = list()
        TN = list()
        Train = list()
        sizes = sorted(sizes)
        for i in sizes:
            ret = self.accuracy((self.size*(i-1)//i),False)
            vals += [self.size*(i-1)//i]
            T +=[ret[0]]
            TP +=[ret[1]]
            TN += [ret[2]]
            Train += [ret[3]]
        plt.figure(1)
        plt.subplot(211)
        plt.ylabel("Percentage")
        plt.xlabel("Train data set size")
        
        accGr, =plt.plot(vals,T,"b-",label = "Average accuracy(test)")
        tpGr, = plt.plot(vals,TP,"g-", label = "Average true positive accuracy(test)")
        tnGr, = plt.plot(vals,TN,"r-",label = "Average true negative accuracy(test) ")
        trainGr, = plt.plot(vals,Train,"y-",label = "Average accuracy(train)")
        plt.legend(handles = [accGr,tpGr,tnGr,trainGr])
        plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        plt.show()
        plt.savefig("RF accuracy graph.png")
    
    #Makes the Roc graph for classifier, using different trainsizes, each n times
    def ROC(self,sizes, n = 10):
        sizes = sorted(sizes)
        plt.figure(1)
        plt.subplot(211)
        plt.ylim(0,1)
        plt.xlim(0,1)
        plt.ylabel("True Positive rate")
        plt.xlabel("False Positive Rate")
        plt.plot([0,1],[0,1],'r-')
        clf = RandomForestClassifier()
        indexes = list(range(self.size))
        dat = dict()
        for i in sizes:
            for j in range(10):
                size = (i-1)*self.size//i
                np.random.shuffle(indexes)
                clf.fit(self.dt[indexes[:size]],self.lbs[indexes[:size]])
                testLbs=self.lbs[indexes[size:]]
                testPre = clf.predict(self.dt[indexes[size:]])
                fpr,tpr,thresh = roc_curve(testLbs,testPre,pos_label = 1)
                if fpr[1] in dat:
                    dat[fpr[1]] += [tpr[1],1]
                else:
                    dat[fpr[1]] = np.array([tpr[1],1])
        fprs = sorted(list(dat.keys()))
        tprs = []
        for i in range(len(fprs)):
            tprs+=[dat[fprs[i]][0]/dat[fprs[i]][1]]
        fprs = [0]+fprs+[1]
        tprs = [0]+tprs+[1]
        plt.plot(fprs,tprs,'b-')
        plt.show()
        plt.savefig("RF ROC.png")
        return fprs,tprs,auc(fprs,tprs)
    
    
    def reClass(self,trainSize = 900):
        indexes = list(range(self.size))
        np.random.shuffle(indexes)
        self.classifiar.fit(self.dt[indexes[:trainSize]],self.lbs[indexes[:trainSize]])
    
    
    def predict(self,x):
        return self.classifiar.predict(x)

class SVM:
    #creates a new object of class, chooses training data and builds predictor
    def __init__(self,data, labels, trainSize=900, ker = 'linear'):
        self.dt = np.array(data)
        self.lbs = np.array(labels)
        self.size = np.size(self.lbs)
        self.classifiar =  SVC(kernel = ker)
        self.kernel = ker
        indexes = list(range(self.size))
        np.random.shuffle(indexes)
        self.classifiar.fit(self.dt[indexes[:trainSize]],self.lbs[indexes[:trainSize]])
    
    #Gets size, creates n times a classifier for data, prints average stats
    def accuracy(self, trainsize, show = True, n = 10):
        testsize = self.size-trainsize
        TP = 0
        TN = 0
        T = 0
        clf = SVC(kernel = self.kernel)
        TrainAcc = 0
        indexes =list(range(self.size))
        for i in range(n):
            np.random.shuffle(indexes)
            train_data = self.dt[indexes[:trainsize]]
            train_labels = self.lbs[indexes[:trainsize]]
            test_data = self.dt[indexes[trainsize:]]
            test_labels = self.lbs[indexes[trainsize:]]
            clf.fit(train_data,train_labels)
            prediction = clf.predict(test_data)
            P = np.sum(test_labels == 1)
            F = np.sum(test_labels == 0)
            corTrue= np.sum(prediction == test_labels)
            fpr,tpr,thresh = roc_curve(testLbs,testPre,pos_label = 1)
            TP += tpr[1]
            TN += (1-fpr[1])
            TrainAcc += np.sum(prediction == train_labels)/trainsize
        T /= n
        TN /= n
        TP /= n
        TrainAcc /= n
        if show:
            print( "avarage accuracy percentage on test data = "+str(T))
            print("avarage true positive percentage on test data = "+str(TP))
            print("avarage true negative percentage on test data = "+str(TN))
            print( "avarage accuracy percentage on train data = "+str(TrainAcc))
            plt.figure(1)
            plt.subplot(211)
            plt.ylabel("Percentage")
            plt.bar([0,1,2,3],[T,TP,TN,TrainAcc],tick_label=["Avarage accuracy\non test data","Avarage true positive\npercantage on test data","Avarage false positive\npercantage on test data","Avarage accuracy\non train data"])
            plt.show()
            plt.savefig("SVM accuracy.png")
        return [T,TP,TN,TrainAcc]
    
    #Get sizes list and make accuracies graph depended on those train sizes
    def  accurGraphs(self, sizes):
        vals = list()
        T = list()
        TP = list()
        TN = list()
        Train = list()
        sizes = sorted(sizes)
        for i in sizes:
            ret = self.accuracy((self.size*(i-1)//i),False)
            vals += [self.size*(i-1)//i]
            T +=[ret[0]]
            TP +=[ret[1]]
            TN += [ret[2]]
            Train += [ret[3]]
        plt.figure(1)
        plt.subplot(211)
        plt.ylabel("Percentage")
        plt.xlabel("Train data set size")
        
        accGr, =plt.plot(vals,T,"b-",label = "Average accuracy(test)")
        tpGr, = plt.plot(vals,TP,"g-", label = "Average true positive accuracy(test)")
        tnGr, = plt.plot(vals,TN,"r-",label = "Average true negative accuracy(test) ")
        trainGr, = plt.plot(vals,Train,"y-",label = "Average accuracy(train)")
        plt.legend(handles = [accGr,tpGr,tnGr,trainGr])
        plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        plt.show()
        plt.savefig("SVM accuracy graph.png")
    
    #Makes the Roc graph for classifier, using different trainsizes, each n times
    def ROC(self,sizes, n = 10):
        plt.figure(1)
        plt.subplot(211)
        plt.ylim(0,1)
        plt.xlim(0,1)
        plt.ylabel("True Positive rate")
        plt.xlabel("False Positive Rate")
        plt.plot([0,1],[0,1],'r-')
        clf = SVC(kernel = self.kernel)
        indexes = list(range(self.size))
        dat = dict()
        for i in sizes:
            for j in range(10):
                size = (i-1)*self.size//i
                np.random.shuffle(indexes)
                clf.fit(self.dt[indexes[:size]],self.lbs[indexes[:size]])
                testLbs=self.lbs[indexes[size:]]
                testPre = clf.predict(self.dt[indexes[size:]])
                fpr,tpr,thresh = roc_curve(testLbs,testPre,pos_label = 1)
                if fpr[1] in dat:
                    dat[fpr[1]] += [tpr[1],1]
                else:
                    dat[fpr[1]] = np.array([tpr[1],1])
        fprs = sorted(list(dat.keys()))
        tprs = []
        for i in range(len(fprs)):
            tprs+=[dat[fprs[i]][0]/dat[fprs[i]][1]]
        fprs = [0]+fprs+[1]
        tprs = [0]+tprs+[1]
        plt.plot(fprs,tprs,'b-')
        plt.show()
        plt.savefig("SVM ROC.png")
        return fprs,tprs,auc(fprs,tprs)
    
    
    def reClass(self,trainSize = 900):
        indexes = list(range(self.size))
        np.random.shuffle(indexes)
        self.classifiar.fit(self.dt[indexes[:trainSize]],self.lbs[indexes[:trainSize]])
   
    
    def predict(self,x):
        return self.classifiar.predict(x)
