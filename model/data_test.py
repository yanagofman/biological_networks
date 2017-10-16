import csv
import pickle as pkl

from create_model import *
from sklearn.ensemble import RandomForestRegressor

def build_classifier(traindata_filename):
    data = list()
    lbls = np.array([])

    with open(traindata_filename, newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        first_line = False
        j = 0
        for row in reader:
            if not first_line:
                first_line = True
                continue
            lbls = np.append(lbls, float(row[2]))
            vec = np.array(row[3:])
            vec[vec == ''] = '0'
            data.append(vec.astype(int))

    print("---finished reading data file---")
    data = np.array(data)
    tresh1 = np.average(lbls) + np.sqrt(np.var(lbls))
    tresh2 = np.average(lbls) - np.sqrt(np.var(lbls))
    data1 = data[lbls > tresh1]
    data2 = data[lbls <= tresh2]
    lbls1 = lbls[lbls > tresh1]
    lbls2 = lbls[lbls <= tresh2]
    lbls = np.append(lbls1, lbls2)
    data = np.append(data1, data2, axis=0)

    lbls[lbls > tresh1] = 1
    lbls[lbls <= tresh2] = 0

    print("---finished building data set---")

    rfClf = RF(data, lbls, int(0.9 * len(data)))
    print("---finished building classifiers---")
    pkl.dump(rfClf, open('random_forest_classifier.p', 'wb'))
    print("---done---")

    return rfClf  # , svmClf



def build_classifier_for_celline(traindata_filename,skip = False):
    data = list()
    lbls = np.array([])

    with open(traindata_filename, newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        first_line = False
        j = 0
        for row in reader:
            if not first_line:
                first_line = True
                continue
            j+=1
            if skip and j == 2:
                continue
            lbls = np.append(lbls, float(row[1]))
            vec = np.array(row[2:])
            vec[vec == ''] = '0'
            data.append(vec.astype(int))

    print("---finished reading data file---")
    data = np.array(data)
    tresh1 = np.average(lbls) + np.sqrt(np.var(lbls))
    tresh2 = np.average(lbls) - np.sqrt(np.var(lbls))
    data1 = data[lbls > tresh1]
    data2 = data[lbls <= tresh2]
    lbls1 = lbls[lbls > tresh1]
    lbls2 = lbls[lbls <= tresh2]
    lbls = np.append(lbls1, lbls2)
    data = np.append(data1, data2, axis=0)

    lbls[lbls > tresh1] = 1
    lbls[lbls <= tresh2] = 0

    print("---finished building data set---")

    rfClf = RF(data, lbls, int(0.9 * len(data)))
    print("---finished building classifiers---")
    pkl.dump(rfClf, open('random_forest_classifier.p', 'wb'))
    print("---done---")

    return rfClf 

def load_data1():
    data_dict = dict()
    lbls_dict = dict()

    with open(traindata_filename, newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        first_line = False
        j = 0
        for row in reader:
            if not (first_line):
                first_line = True
                continue
            if row[0] in data_dict:
                lbls_dict[row[0]] = np.append(lbls_dict[row[0]], float(row[2]))
                vec = np.array(row[3:])
                vec[vec == ''] = '0'
                data_dict[row[0]].append(vec.astype(int))
            else:
                lbls_dict[row[0]] = np.array(float(row[2]))
                vec = np.array(row[3:])
                vec[vec == ''] = '0'
                data_dict[row[0]] = [vec.astype(int)]

    print("---finished reading data file---")
    data = None
    lbls = np.array([])
    for cell in data_dict:
        print('---', cell, '---')
        print(len(data_dict[cell]))
        tresh1 = np.average(lbls_dict[cell]) + np.sqrt(np.var(lbls_dict[cell]))
        tresh2 = np.average(lbls_dict[cell]) - np.sqrt(np.var(lbls_dict[cell]))
        data1 = np.array(data_dict[cell])
        lbls1 = lbls_dict[cell][lbls_dict[cell] > tresh1]

        lbls = np.append(lbls, lbls1)
        if data != None:
            data = np.append(data, data1, axis=0)
        else:
            data = data1

    print("---finished building data set---")
    
    return data,lbls

def load_data(traindata_filename,skip = False):
    data_dict = dict()
    lbls_dict = dict()

    with open(traindata_filename, newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        first_line = False
        j = 0
        for row in reader:
            if not (first_line):
                first_line = True
                continue
            if row[0] in data_dict:
                lbls_dict[row[0]] = np.append(lbls_dict[row[0]], float(row[2]))
                vec = np.array(row[3:])
                vec[vec == ''] = '0'
                data_dict[row[0]].append(vec.astype(int))
            else:
                lbls_dict[row[0]] = np.array(float(row[2]))
                vec = np.array(row[3:])
                vec[vec == ''] = '0'
                data_dict[row[0]] = [vec.astype(int)]

    print("---finished reading data file---")
    data = None
    lbls = np.array([])
    for cell in data_dict:
        print('---', cell, '---')
        print(len(data_dict[cell]))
        tresh1 = np.average(lbls_dict[cell]) + np.sqrt(np.var(lbls_dict[cell]))
        tresh2 = np.average(lbls_dict[cell]) - np.sqrt(np.var(lbls_dict[cell]))
        data1 = np.array(data_dict[cell])[lbls_dict[cell] > tresh1]
        data2 = np.array(data_dict[cell])[lbls_dict[cell] <= tresh2]
        lbls1 = lbls_dict[cell][lbls_dict[cell] > tresh1]
        
        lbls2 = lbls_dict[cell][lbls_dict[cell] <= tresh2]
        lbls = np.append(lbls, lbls1)
        lbls = np.append(lbls, lbls2)
        if data != None:
            data = np.append(data, np.append(data1, data2, axis=0), axis=0)
        else:
            data = np.append(data1, data2, axis=0)

    print("---finished building data set---")
    return data,lbls


def build_classifier2(traindata_filename):
    data_dict = dict()
    lbls_dict = dict()

    with open(traindata_filename, newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        first_line = False
        j = 0
        for row in reader:
            if not (first_line):
                first_line = True
                continue
            if row[0] in data_dict:
                lbls_dict[row[0]] = np.append(lbls_dict[row[0]], float(row[2]))
                vec = np.array(row[3:])
                vec[vec == ''] = '0'
                data_dict[row[0]].append(vec.astype(int))
            else:
                lbls_dict[row[0]] = np.array(float(row[2]))
                vec = np.array(row[3:])
                vec[vec == ''] = '0'
                data_dict[row[0]] = [vec.astype(int)]

    print("---finished reading data file---")
    data = None
    lbls = np.array([])
    for cell in data_dict:
        print('---', cell, '---')
        print(len(data_dict[cell]))
        tresh1 = np.average(lbls_dict[cell]) + np.sqrt(np.var(lbls_dict[cell]))
        tresh2 = np.average(lbls_dict[cell]) - np.sqrt(np.var(lbls_dict[cell]))
        data1 = np.array(data_dict[cell])[lbls_dict[cell] > tresh1]
        data2 = np.array(data_dict[cell])[lbls_dict[cell] <= tresh2]
        lbls1 = np.ones(len(data1))
        lbls2 = np.zeros(len(data2))
        lbls = np.append(lbls, lbls1)
        lbls = np.append(lbls, lbls2)
        if data != None:
            data = np.append(data, np.append(data1, data2, axis=0), axis=0)
        else:
            data = np.append(data1, data2, axis=0)

    print("---finished building data set---")

    rfClf = RF(data, lbls, int(0.9 * len(data)))
    print("---finished building classifiers---")
    pkl.dump(rfClf, open('random_forest_classifier.p', 'wb'))
    print("---done---")

    return rfClf  # , svmClf


def build_classifier3(traindata_filename):
    data_dict = dict()
    lbls_dict = dict()

    with open(traindata_filename, newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        first_line = False
        j = 0
        for row in reader:
            if not (first_line):
                first_line = True
                continue
            if row[0] in data_dict:
                lbls_dict[row[0]] = np.append(lbls_dict[row[0]], float(row[2]))
                vec = np.array(row[3:])
                vec[vec == ''] = '0'
                data_dict[row[0]].append(vec.astype(int))
            else:
                lbls_dict[row[0]] = np.array(float(row[2]))
                vec = np.array(row[3:])
                vec[vec == ''] = '0'
                data_dict[row[0]] = [vec.astype(int)]

    print("---finished reading data file---")
    data = None
    lbls = np.array([])
    for cell in data_dict:
        print('---', cell, '---')
        print(len(data_dict[cell]))
        tresh = np.average(lbls_dict[cell])
        lbls_dict[cell][lbls_dict[cell] > tresh] = 1
        lbls_dict[cell][lbls_dict[cell] <= tresh] = 0
        lbls = np.append(lbls, lbls_dict[cell])
        if data != None:
            data = np.append(data, np.array(data_dict[cell]), axis=0)
        else:
            data = np.array(data_dict[cell])

    print("---finished building data set---")

    rfClf = RF(data, lbls, int(0.9 * len(data)))
    print("---finished building classifiers---")
    pkl.dump(rfClf, open('random_forest_classifier.p', 'wb'))
    print("---done---")

    return rfClf  # , svmClf


def build_classifier4(traindata_filename):
    data_dict = dict()
    lbls_dict = dict()

    with open(traindata_filename, newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        first_line = False
        j = 0
        for row in reader:
            if not (first_line):
                first_line = True
                continue
            if row[0] in data_dict:
                lbls_dict[row[0]] = np.append(lbls_dict[row[0]], float(row[2]))
                vec = np.array(row[3:])
                vec[vec == ''] = '0'
                data_dict[row[0]].append(vec.astype(int))
            else:
                lbls_dict[row[0]] = np.array(float(row[2]))
                vec = np.array(row[3:])
                vec[vec == ''] = '0'
                data_dict[row[0]] = [vec.astype(int)]

    print("---finished reading data file---")
    data = None
    lbls = np.array([])
    for cell in data_dict:
        print('---', cell, '---')
        print(len(data_dict[cell]))
        tresh1 = np.average(lbls_dict[cell]) + np.sqrt(np.var(lbls_dict[cell]))
        tresh2 = np.average(lbls_dict[cell]) - np.sqrt(np.var(lbls_dict[cell]))
        data1 = np.array(data_dict[cell])[lbls_dict[cell] > tresh1]
        data2 = np.array(data_dict[cell])[lbls_dict[cell] < tresh2]
        data3 = np.array(data_dict[cell])[
            abs(lbls_dict[cell] - np.average(lbls_dict[cell])) <= np.sqrt(np.var(lbls_dict[cell]))]
        lbls1 = np.ones(len(data1))
        lbls2 = np.ones(len(data2)) * -1
        lbls3 = np.zeros(len(data3))
        lbls = np.append(lbls, lbls1)
        lbls = np.append(lbls, lbls2)
        lbls = np.append(lbls, lbls3)
        if data != None:
            data = np.append(data, np.append(np.append(data1, data2, axis=0), data3, axis=0), axis=0)
        else:
            data = np.append(np.append(data1, data2, axis=0), data3, axis=0)

    print("---finished building data set---")

    rfClf = RF(data, lbls, int(0.9 * len(data)))
    print("---finished building classifiers---")
    pkl.dump(rfClf, open('random_forest_classifier.p', 'wb'))
    print("---done---")

    return rfClf


def load_classifiers():
    rfClf = pkl.load(open('random_forest_classifier.p', 'rb'))
    # svmClf = pkl.load(open('svm_classifier.p','rb'))
    return rfClf  # , svmClf


def test_classifiers(rfClf, svmClf=None):
    sizes = [int((0.5 + 0.1 * i) * rfClf.size) for i in range(5)]

    rfClf.accur_graphs(sizes, n=1)
    # svmClf.accurGraphs(sizes)

    rf_fprs, rf_tprs, rf_auc = rfClf.ROC(sizes, n=1)
    # svm_fprs,svm_tprs,svm_auc = svmClf.ROC(sizes)

    # rfClf.pre_recall(sizes,n=2)

    plt.figure(1)
    plt.subplot(211)
    plt.ylim(0, 1)
    plt.xlim(0, 1)
    plt.ylabel("True Positive rate")
    plt.xlabel("False Positive Rate")
    plt.plot([0, 1], [0, 1], 'r-')
    rf_roc, = plt.plot(rf_fprs, rf_tprs, 'b-', label="RF ROC, auc = " + str(rf_auc))
    # svm_roc, = plt.plot(svm_fprs,svm_tprs,'y-',label = "SVM ROc, auc = "+str(svm_auc))
    plt.legend(handles=[rf_roc])
    plt.legend(bbox_to_anchor=(0, -0.95), loc=2, borderaxespad=0.)
    plt.savefig("ROC curves.png")
    plt.show()


def compare_to_random_roc(file_name, builder = build_classifier2 ):
    original_clf = builder("training_set_files/no_shuffle/"+file_name+".csv")
    sizes = [int((0.5 + 0.1 * i) * original_clf.size) for i in range(5)]
    or_fprs,or_tprs,or_auc = original_clf.ROC(sizes,1)
    rand_roc = dict()
    rocs = []
    for i in range(1,11):
        clf = builder("training_set_files/shuffle_"+str(i)+"/"+file_name+".csv")
        fprs,tprs,t_auc = clf.ROC(sizes,1)
        rocs+=[(fprs,tprs,t_auc)]
        for i in range(len(fprs)):
            if(fprs[i] in rand_roc):
                rand_roc[fprs[i]] += [tprs[i],1]
            else:
                rand_roc[fprs[i]] = np.array([tprs[i],1])
    ran_fprs = sorted(list(rand_roc.keys()))
    ran_tprs = []
    for i in range(len(ran_fprs)):
        ran_tprs += [rand_roc[ran_fprs[i]][0] / rand_roc[ran_fprs[i]][1]]
    ran_auc = auc(ran_fprs,ran_tprs)
    plt.figure(1)
    plt.subplot(211)
    plt.ylim(0, 1)
    plt.xlim(0, 1)
    plt.ylabel("True Positive rate")
    plt.xlabel("False Positive Rate")
    plt.plot([0, 1], [0, 1], 'r-')
    legend = []
    or_roc, = plt.plot(or_fprs, or_tprs, 'b-', label="RF ROC, auc = " + str(or_auc))
    legend+=[or_roc]
    ran_roc, = plt.plot(ran_fprs, ran_tprs, 'y-', label="RF ROC of average of shuffles, auc = " + str(ran_auc))
    legend+=[ran_roc]
    plt.legend(handles=legend)
    plt.legend(bbox_to_anchor=(0, -0.95), loc=2, borderaxespad=0.)
    plt.savefig("graph_results/"+file_name+"/Roc curve.png")
    plt.show()
    for i in range(10):
        plt.subplot(211)
        plt.ylim(0, 1)
        plt.xlim(0, 1)
        plt.ylabel("True Positive rate")
        plt.xlabel("False Positive Rate")
        plt.plot([0, 1], [0, 1], 'r-')
        gr, = plt.plot(rocs[i][0], rocs[i][1], 'b-', label="RF shuffle "+str(i+1)+" ROC, auc = " + str(rocs[i][2]))
        legend =[gr]
        plt.legend(handles=[gr])
        plt.legend(bbox_to_anchor=(0, -0.95), loc=2, borderaxespad=0.)
        plt.savefig("graph_results/"+file_name+"/Roc curve - shuffle "+str(i+1)+".png")
        plt.show()
        
def compare_to_random_reg(file_name, builder = load_data ):
    rf = RandomForestRegressor(n_jobs = -1)
    dt,lb = builder("training_set_files/no_shuffle/"+file_name+".csv")
    sizes = [int((0.7 + 0.15 * j) * len(lb)) for j in range(2)]
    test_acc = []
    train_acc = []
    for size in sizes:
        train_dt = dt[:size]
        train_lb = lb[:size]
        test_dt = dt[size+1:]
        test_lb = lb[size+1:]
        rf.fit(train_dt,train_lb)
        test_acc += [np.sum(abs(rf.predict(test_dt) - test_lb) <= 10**-7)/len(test_dt)]
        train_acc += [np.sum(abs(rf.predict(train_dt) - train_lb) <= 10**-7)/len(train_dt)]
    rand_test_acc = np.zeros(2)
    rand_train_acc = np.zeros(2)
    for i in range(1,11):
        temp_test_acc = []
        temp_train_acc = []
        print("---",i,"---")
        temp_dt, temp_lb = builder("training_set_files/shuffle_"+str(i)+"/"+file_name+".csv")
        
        for size in sizes:
            train_dt = temp_dt[:size]
            train_lb = temp_lb[:size]
            test_dt = temp_dt[size+1:]
            test_lb = temp_lb[size+1:]
            rf.fit(train_dt,train_lb)
            temp_test_acc += [np.sum(abs(rf.predict(test_dt) - test_lb) <= 10**-7)/len(test_dt)]
            temp_train_acc += [np.sum(abs(rf.predict(train_dt) - train_lb) <= 10**-7)/len(train_dt)]
        rand_test_acc += temp_test_acc
        rand_train_acc += temp_train_acc
    rand_test_acc /= 10
    rand_train_acc/=10
    plt.subplot(211)
    #plt.ylim(0, 1)
    #plt.xlim(0, 1)
    plt.ylabel("Accuracy")
    plt.xlabel("Train data size")
    test, = plt.plot(sizes, test_acc, 'g-', label="Rf Regression accuracy of real data - test")
    train, = plt.plot(sizes, train_acc, 'b-', label="Rf Regression accuracy of real data - train")
    rand_test, = plt.plot(sizes, rand_test_acc, 'r-', label="Rf Regression accuracy of average random data - test")
    rand_train, = plt.plot(sizes, rand_train_acc, 'y-', label="Rf Regression accuracy of average random data - train")
    plt.legend(handles=[test,train,rand_test,rand_train])
    plt.legend(bbox_to_anchor=(0, -0.95), loc=2, borderaxespad=0.)
    plt.savefig("graph_results/"+file_name+"/Roc curve - shuffle "+str(i+1)+".png")
    plt.show()
        
    