from create_model import *
import csv
import pickle as pkl
from math import *


to_float = lambda x: 0 if x =='' else float(x)

def build_classifier(traindata_filename):
    data = list()
    lbls = np.array([])
    
    with open(traindata_filename, newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        first_line = False
        j = 0
        for row in reader:
            #print('---',j,'---')
            #j+=1
            if not(first_line):
                first_line = True
                continue
            lbls = np.append(lbls, float(row[2]))
            vec = np.array(row[3:])
            vec[vec == ''] = '0'
            data.append(vec.astype(int))
       
    print("---finished reading data file---")
    data = np.array(data)        
    tresh = np.average(lbls)
    #print("max = ",max(lbls),",min = ",min(lbls), ",avg = ",tresh)
    lbls[lbls > tresh] = 1
    lbls[lbls <= tresh] = 0
    
    print("---finished building data set---")
    
    rfClf = RF(data,lbls,int(0.9*len(data)))
   # svmClf = SVM(data,lbls,int(0.9*len(data)))
    print("---finished building classifiers---")
    pkl.dump(rfClf,open('random_forest_classifier.p','wb'))
   # pkl.dump(svmClf,open('svm_classifier.p','wb'))
    print("---done---")
   
    
    return rfClf#, svmClf

def load_classifiers():
    rfClf = pkl.load(open('random_forest_classifier.p','rb'))
    #svmClf = pkl.load(open('svm_classifier.p','rb'))
    return rfClf#, svmClf

def test_classifiers(rfClf,svmClf=None):
    sizes = [int((0.5+0.04*i)*rfClf.size) for i in range(11)]

    rfClf.accur_graphs(sizes,n=2)
   # svmClf.accurGraphs(sizes)

    rf_fprs,rf_tprs,rf_auc = rfClf.ROC(sizes,n=2)
    #svm_fprs,svm_tprs,svm_auc = svmClf.ROC(sizes)

    rfClf.pre_recall(sizes,n=2)

    plt.figure(1)
    plt.subplot(211)
    plt.ylim(0,1)
    plt.xlim(0,1)
    plt.ylabel("True Positive rate")
    plt.xlabel("False Positive Rate")
    plt.plot([0,1],[0,1],'r-')
    rf_roc, = plt.plot(rf_fprs,rf_tprs,'b-',label = "RF ROC, auc = "+str(rf_auc))
    #svm_roc, = plt.plot(svm_fprs,svm_tprs,'y-',label = "SVM ROc, auc = "+str(svm_auc))
    plt.legend(handles = [rf_roc])
    plt.legend(bbox_to_anchor=(0, -0.95), loc=2, borderaxespad=0.)
    plt.savefig("ROC curves.png")
    plt.show()

