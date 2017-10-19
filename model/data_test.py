import csv
import pickle as pkl
from model.create_model import *

#loads list of genes from file for alimination
def load_genes(filename):
    data = set()
    data.add("")
    with open(filename, newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        for row in reader:
            data.add(row[0])
    return np.array(list(data))

#builds an instance of RF to test RandomForestClassifier
#for all the data together:
    #it throws away all samples whose scores are within the standard deviation distance from the mean of the scores
    #for the sampels that were left, all those with scores above the mean it defines as 1, and the rest 0


def build_RF(traindata_filename):
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

    rfClf = RF_TESTS(data, lbls)
    print("---finished building classifiers---")
    #pkl.dump(rfClf, open('random_forest_classifier.p', 'wb'))
    print("---done---")

    return rfClf  # , svmClf



def build_RF_for_celline(traindata_filename,skip = True,genes = None):
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
            if genes != None and not(row[1] in genes):
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
    a = list(range(len(lbls)))
    np.random.shuffle(a)
    lbls = lbls[a]
    data = data[a]

    print("---finished building data set---")

    rfClf = RF_TESTS(data, lbls )
    print("---finished building classifiers---")
    #pkl.dump(rfClf, open('random_forest_classifier.p', 'wb'))
    print("---done---")

    return rfClf 

#Loads the data to test for regression
#for each cell line:
    #it throws away all samples whose scores are within the standard deviation distance from the mean of the scores
#Combines the results for all the cellines
#If genes is a list of genes, it loads only the sampels related to those genes

def load_data(traindata_filename,genes = None):
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
            if genes != None and not(row[1] in genes):
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

def load_data_celline(traindata_filename,genes = None,skip = True):
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
             if genes != None and not(row[1] in genes):
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
     
     print("---finished building data set---")
     return data, lbls

#Loads data for classifier
#for each cell line:
    #it throws away all samples whose scores are within the standard deviation distance from the mean of the scores
    #for the sampels that were left, all those with scores above the mean it defines as 1, and the rest 0
#Combines the results for all the cellines
#If genes is a list of genes, it loads only the sampels related to those genes
def load_data_clf(traindata_filename,genes = None):
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
            if genes != None and not(row[1] in genes):
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
    return data,lbls

#Builds RF to test classifier
#for each cell line:
    #it throws away all samples whose scores are within the standard deviation distance from the mean of the scores
    #for the sampels that were left, all those with scores above the mean it defines as 1, and the rest 0
#Combines the results for all the cellines
#If genes is a list of genes, it loads only the sampels related to those genes
def build_RF2(traindata_filename, genes = None):
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
            if genes != None and not(row[1] in genes):
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
    a = list(range(len(lbls)))
    np.random.shuffle(a)
    rfClf = RF_TESTS(data[a], lbls[a])
    print("---finished building classifiers---")
   # pkl.dump(rfClf, open('random_forest_classifier.p', 'wb'))
    print("---done---")

    return rfClf  # , svmClf

#Builds RF to test classifier
#for each cell line:
    #for the sampels, all those with scores above the mean it defines as 1, and the rest 0
#Combines the results for all the cellines
#If genes is a list of genes, it loads only the sampels related to those genes
def build_RF3(traindata_filename,genes = None):
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
            if genes != None and not(row[1] in genes):
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
        tresh = np.average(lbls_dict[cell])
        lbls_dict[cell][lbls_dict[cell] > tresh] = 1
        lbls_dict[cell][lbls_dict[cell] <= tresh] = 0
        lbls = np.append(lbls, lbls_dict[cell])
        if data != None:
            data = np.append(data, np.array(data_dict[cell]), axis=0)
        else:
            data = np.array(data_dict[cell])

    print("---finished building data set---")

    rfClf = RF_TESTS(data, lbls)
    print("---finished building classifiers---")
    #pkl.dump(rfClf, open('random_forest_classifier.p', 'wb'))
    print("---done---")

    return rfClf  # , svmClf





#Tests and compares the RandomForestClassifier's roc for real data, and shuffeled ontotype data
def compare_to_random_roc(file_name, builder = build_RF2, genes_list = None, num_of_shuffles = 10 ):
    original_clf = builder("training_set_files/no_shuffle/"+file_name+".csv",genes = genes_list)
    or_fprs,or_tprs,or_auc = original_clf.clf_roc()
    rand_roc = dict()
    rocs = []
    for i in range(1,num_of_shuffles+1):
        print("---",i,"---")
        clf = builder("training_set_files/shuffle_"+str(i)+"/"+file_name+".csv",genes = genes_list)
        fprs,tprs,t_auc = clf.clf_roc()
        rocs+=[(fprs,tprs,t_auc)]
        for j in range(len(fprs)):
            if(fprs[j] in rand_roc):
                rand_roc[fprs[j]] += [tprs[j],1]
            else:
                rand_roc[fprs[j]] = np.array([tprs[j],1])
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
    plt.savefig("graph_results/ROC -"+file_name+".png")
    plt.show()
   
#Compares the RandomForestRegressor's squared distance of prediction from real labels, for real data, and shuffeled ontotype data
def compare_to_random_reg(file_name, loader = load_data, genes_list = None ,num_of_shuffles = 10):
    dt,lb = loader("training_set_files/no_shuffle/"+file_name+".csv",genes = genes_list)
    rf = RF_TESTS(dt,lb)
    real_dist = rf.reg_av_squard_distance()
    avg_rand_dist = 0
    for i in range(1,num_of_shuffles+1):
        print("---",i,"---")
        dt,lb = loader("training_set_files/shuffle_"+str(i)+"/"+file_name+".csv",genes = genes_list)
        rf = RF_TESTS(dt,lb)
        dist = rf.reg_av_squard_distance()
        print("dist =",dist)
        print("--------")
        avg_rand_dist += dist/num_of_shuffles
    print("real distance = ",real_dist)
    print("avarage diatnace of random = ",avg_rand_dist)
    return real_dist,avg_rand_dist
    
        
    
