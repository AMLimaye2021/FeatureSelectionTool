import sys
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.model_selection import train_test_split
import GEOparse
import pickle


def getValues(data):
    table = data.table
    table_expression = np.array(table)
    table_info = np.array(data.columns)
    table_info = table_info[:, 1]
    gene_table = table_expression[:, 1]
    table_expression = table_expression[:, 2:]
    table_expression = table_expression.astype(np.float_)
    table_expression = table_expression.transpose()
    return table_expression, table_info

def catTables(Etables, Itables):
    a = Etables[0]
    for i in range(1, len(Etables)):
        a = np.concatenate((a, Etables[i]), axis=0)
    c = Itables[0]
    for j in range(1, len(Itables)):
        c = np.concatenate((c, Itables[j]), axis=None)
    return a,c

def unpackValues(data):
    data_gsms = data.gsms
    dataKeys = list(data_gsms.keys())
    geneData = data_gsms[dataKeys[1]].table
    geneData = np.array(geneData)
    geneIndex = np.zeros(shape=(len(data_gsms[dataKeys[0]].table), 1), dtype=str)
    geneIndex = geneData[:, 0]
    unpacked_data = np.zeros(shape=(len(data_gsms), len(data_gsms[dataKeys[0]].table)))
    # unpacked_data = np.zeros(shape=(len(data_gsms), 1))
    data_class = np.array(data.phenotype_data)
    data_class = data_class[:,12]
    # data_class = data_class[:, 10]
    unpacked_class = np.zeros(shape=len(data_gsms), dtype=int)
    # for c in range(len(data.gsms)):
    #     if data_class[c] == "Lung cancer, pre-operation":
    #         unpacked_class[c] = 1
    #     elif data_class[c] == "Lung cancer, post-operation":
    #         unpacked_class[c] = 2
    for c in range(len(data.gsms)):
        if data_class[c] == "noninfectious":
            unpacked_class[c] = 1
        elif data_class[c] == "sepsis":
            unpacked_class[c] = 2

    for i in range(len(data.gsms)):
        data_table = data_gsms[dataKeys[i]].table
        data_table = np.array(data_table)
        unpacked_data[i, :] = data_table[:, 1]

    return unpacked_data, unpacked_class, geneIndex

def crossValidation(dataExpression, dataClassification, ts):
    Cs = np.linspace(0.001, 2, 25)
    # Cs = np.linspace(0.001, 10, 2)
    # Cs = [0.1]
    cmax = [0, 0, []]
    cVals = np.array([0, 0, [], 0, 0])
    for c in Cs:
        model = svm.LinearSVC(penalty='l1', loss='squared_hinge', dual=False, C=c, max_iter=10000, tol=1e-3)
        split = train_test_split(dataExpression, dataClassification, train_size=ts)
        train_set = [split[0], split[2]]
        test_set = [split[1], split[3]]
        model.fit(train_set[0], train_set[1])
        predict = model.predict(test_set[0])
        count = np.sum(predict == test_set[1])
        if (count / len(predict)) > cmax[1]:
            cmax[0] = c
            cmax[1] = count / len(predict)
            cmax[2] = model.coef_
        v = np.sum(np.abs(model.coef_), axis=0)
        v = np.sum(v > 0)
        cVals = np.vstack((cVals, np.array([c, count/len(predict), model.coef_, ts, v])))
        print(c, count / len(predict))
    cVals = cVals[1:, :]
    return cmax, cVals

def plotRegularization(reg, ts):
    colors = ['tab:red', 'tab:blue', 'tab:orange', 'tab:green']

    plot1 = plt.figure(num=1, figsize=(5,5))

    plt.xlabel('Regularization Parameter')
    plt.ylabel('Accuracy')

    for t in range(0, len(ts)):
        mask = reg[reg[:,3]==ts[t], :]
        plt.plot(mask[:,0], mask[:,1], color=colors[t], label=('Train Split: ' + str(ts[t])))
    plt.legend(loc='lower right')

    plot2 = plt.figure(num=2, figsize=(5,5))
    plt.xlabel('N_Features')
    plt.ylabel('Accuracy')
    for t in range(0, len(ts)):
        mask = reg[reg[:,3]==ts[t], :]
        plt.plot(mask[:,4], mask[:,1], color=colors[t], label=('Train Split: ' + str(ts[t])))

    plt.legend(loc='lower right')
    plt.show()

def geneMax(rM, geneIndex):
    v = rM[2]
    vA = np.sum(np.abs(v), axis=0)
    indices = np.argsort(vA)
    print('Absolute Value Feature Weighting')
    for i in range(20):
        # print(str(geneIndex[indices[i]]) + ': ' + str(v[:, indices[i]]))
        print(str(geneIndex[indices[i]]) + ': ' + str(v[:, indices[i]]))


def main():
    fname = "cache.pkl"
    rebuildcache = False


    if rebuildcache:
        data1 = GEOparse.get_GEO(filepath="./GDS2947.soft.gz")  # Adenoma/Healthy Set (Testing?) (True Count)
        data2 = GEOparse.get_GEO(filepath="./GDS4379.soft.gz")  # Adenocarcinoma (Testing?) (True Count) Samples 64
        data4 = GEOparse.get_GEO(filepath="./GDS4393.soft.gz")  # Metastatic/Tumor Set (True Count)
        data5 = GEOparse.get_GEO(filepath="./GDS4513.soft.gz")  # Tumor/Excised Set (Transformed count) Samples 53
        data6 = GEOparse.get_GEO(filepath="./GDS4516.soft.gz")  # Metastatic/Stage 3 Set (Transformed count) Samples 104
        data7 = GEOparse.get_GEO(filepath="./GSE137140_family.soft.gz")
        data8 = GEOparse.get_GEO(filepath="./GSE134347_family.soft.gz")
        tup = (data1, data2, data4, data5, data6, data7, data8)
        with open(fname, 'wb') as f:
            pickle.dump(tup, f)
    else:
        with open(fname, 'rb') as f:
            tup = pickle.load(f)
        (data1, data2, data4, data5, data6, data7, data8) = tup
    gseData, gseClass, geneIndex = unpackValues(data8)

    # table1_expression, table1_info = getValues(data1)
    # table1_expression = np.log2(table1_expression)
    # table6_expression, table6_info = getValues(data6)
    # table2_expression, table2_info = getValues(data2)

    # catDataE, catDataI = catTables([table1_expression, table6_expression, table2_expression], [table1_info, table6_info,table2_info])
    # infolen = catDataI.shape
    # infolen = infolen[0]
    # classification = np.zeros((infolen), dtype=int)
    # for i in range(infolen):
    #     if (catDataI[i] == "adenoma" or catDataI[i] == "Large Intestine, Villous Adenoma"):
    #         classification[i] = 1
    #     elif (catDataI[i] == "normal mucosa"):
    #         classification[i] = 2
    trainSizes = np.linspace(0.2, 0.8, 4)
    regMax = np.array([0, 0, []])
    regularization = np.array([0, 0, [], 0, 0])

    for t in trainSizes:
        cMax, cVals = crossValidation(gseData, gseClass, t)
        regMax =  np.vstack((regMax,cMax))
        regularization = np.vstack((regularization, cVals))
    regMax = regMax[1:, :]
    regularization = regularization[1:, :]
    plotRegularization(regularization, trainSizes)
    geneMax(regMax[3], geneIndex)
    print(1)
    # cross validation
    # take all test set errors and average on the test set
    # gives relationship between c and the error
    # run cross validation on multiple values of C


if __name__ == "__main__":
    main()
