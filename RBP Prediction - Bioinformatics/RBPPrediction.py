
# @Author - Antony

# Python Script to read feature vectors, preprocessed them and construct models for training and validation of test species for RBP prediction.
# Please update the input files path accordingly.
# Scikit learn SVM and RF packages used in this script to build SVM and RF classifier.
# 

import numpy as np
from sklearn import svm
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
import re
from sklearn.cross_validation import KFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier
import time

accuracy_list=[]

def train(data_path_train, feature_num_train, pos_count_train):
    #Training machine learning model
    i=0
    with open(data_path_train) as f:
        data = f.readlines()
        
    #extracting the desired data rows from the txt data file
    data = [x.strip() for x in data]
    s=''.join(data)
    r = re.findall('c\((.+?)\)', s)
    
    X_train=np.zeros([len(r),feature_num_train], dtype=float)
    for i in range(0,len(r)):
        v = re.findall('\d+\.\d+|[0-9]|NA', r[i])
        for j in range(0,feature_num_train):
            if v[j]=="NA":
                v[j]=0
            X_train[i][j]=v[j]
            
    Y_train=np.zeros([len(X_train)], dtype=int)
    for i in range(0,len(Y_train)):
        if i<pos_count_train:
            Y_train[i]=1
        else:
            Y_train[i]=0
            
    #model trained
    t0=time.time()
	# C and gamma parameters obatined from source system (RPBPred, Zhang et al 2016).
    svm_classifier = svm.SVC(kernel='rbf', C = 185363.800047, gamma=0.000690533966002)
    svm_classifier.fit(X_train,Y_train)
    t1=time.time()
    total=t1-t0
    print("SVM training time in seconds", total)
    
    t0=time.time()
    randomforest_classifier = RandomForestClassifier(random_state=0)
    randomforest_classifier.fit(X_train,Y_train)
    t1=time.time()
    total=t1-t0
    print("Random Forest training time in seconds", total)
    
    
    #testing the trained model on the training data
    
    #SVM classifier
    Y_predicted=svm_classifier.predict(X_train)
    cm=confusion_matrix(Y_train, Y_predicted)
    print("Confusion matrix for SVM Training:", cm)
    TP = cm[0][0]
    FP = cm[0][1]
    FN = cm[1][0]
    TN = cm[1][1]
    sensitivity=TP/float(TP+FN)
    specificity=TN/float(TN+FP)
    precision=TP/float(TP+FP)
    accuracy=(TP+TN)/float(TP+FN+TN+FP)
    f_measure=(2*precision*sensitivity)/float(precision+sensitivity)
    mathews_correlation_coeff=(TP*TN-FP*FN)/float((TP+FN)*(TP+FP)*(TN+FP)*(TN+FN))**2
    print("SVM Classifier Results for Training data:")
    print(f'sensitivity {sensitivity} specificity {specificity} precision {precision} accuracy {accuracy} f_measure {f_measure} mathews_correlation_coeff {mathews_correlation_coeff}')
    k_fold = KFold(len(Y_train), n_folds=10, shuffle=True, random_state=0)
    sign='±'
    b = sign.encode('utf-8')
    k=cross_val_score(svm_classifier, X_train, Y_train, cv=k_fold, n_jobs=1)
    accuracy_list.append(np.mean(k))
    print("10-fold average accuracy for SVM:", np.mean(k), b.decode('utf-8'), np.std(k) * 2)
    print("\n")
    
    #Random Forest Classifier 
    Y_predicted=randomforest_classifier.predict(X_train)
    cm=confusion_matrix(Y_train, Y_predicted)
    print("Confusion matrix for RF Training:", cm)
    TP = cm[0][0]
    FP = cm[0][1]
    FN = cm[1][0]
    TN = cm[1][1]
    sensitivity=TP/float(TP+FN)
    specificity=TN/float(TN+FP)
    precision=TP/float(TP+FP)
    accuracy=(TP+TN)/float(TP+FN+TN+FP)
    f_measure=(2*precision*sensitivity)/float(precision+sensitivity)
    mathews_correlation_coeff=(TP*TN-FP*FN)/float((TP+FN)*(TP+FP)*(TN+FP)*(TN+FN))**2
    print("Random Forest Classifier Results for Training data:")
    print(f'sensitivity {sensitivity} specificity {specificity} precision {precision} accuracy {accuracy} f_measure {f_measure} mathews_correlation_coeff {mathews_correlation_coeff}')
    k_fold = KFold(len(Y_train), n_folds=10, shuffle=True, random_state=0)
    sign='±'
    b = sign.encode('utf-8')
    k=cross_val_score(randomforest_classifier, X_train, Y_train, cv=k_fold, n_jobs=1)
    accuracy_list.append(np.mean(k))
    print("10-fold average accuracy for Random Forest:", np.mean(k), b.decode('utf-8'), np.std(k) * 2)
    print("\n\n")
    
    return svm_classifier, randomforest_classifier



#Validating the trained model on the test species
def test(data_path, feature_num, pos_count, svm_model, randomforest_model):
    i=0
    with open(data_path) as f:
        data = f.readlines()
        
    #removing whitespace characters
    data = [x.strip() for x in data]
    s=''.join(data)
    r = re.findall('c\((.+?)\)', s)

    X_test=np.zeros([len(r),feature_num], dtype=float) 
    for i in range(0,len(r)):
        v = re.findall('\d+\.\d+|[0-9]|NA', r[i])
        for j in range(0,feature_num):
            if v[j]=="NA":
                v[j]=0
            X_test[i][j]=v[j]
            
    Y_test=np.zeros([len(X_test)], dtype=int)
    for i in range(0,len(Y_test)):
        if i<pos_count:
            Y_test[i]=1
        else:
            Y_test[i]=0
    
    #SVM classifier
    t0=time.time()
    Y_predicted=svm_model.predict(X_test)
    t1=time.time()
    total=t1-t0
    print("SVM testing time in seconds", total)
    
    cm=confusion_matrix(Y_test, Y_predicted)
    print("Confusion matrix for SVM on Test dataset:", cm)
    TP = cm[0][0]
    FP = cm[0][1]
    FN = cm[1][0]
    TN = cm[1][1]
    sensitivity=TP/float(TP+FN)
    specificity=TN/float(TN+FP)
    precision=TP/float(TP+FP)
    accuracy=(TP+TN)/float(TP+FN+TN+FP)
    accuracy_list.append(accuracy)
    f_measure=(2*precision*sensitivity)/float(precision+sensitivity)
    mathews_correlation_coeff=(TP*TN-FP*FN)/float((TP+FN)*(TP+FP)*(TN+FP)*(TN+FN))**2
    print("SVM Classifier Results for Test dataset:")
    print(f'sensitivity {sensitivity} specificity {specificity} precision {precision} accuracy {accuracy} f_measure {f_measure} mathews_correlation_coeff {mathews_correlation_coeff}')
    print("\n")

    #Random Forest classifier
    t0=time.time()
    Y_predicted=randomforest_model.predict(X_test)
    t1=time.time()
    total=t1-t0
    print("Random Forest testing time in seconds", total)
    
    cm=confusion_matrix(Y_test, Y_predicted)
    print("Confusion matrix for RF on Test dataset:", cm)
    TP = cm[0][0]
    FP = cm[0][1]
    FN = cm[1][0]
    TN = cm[1][1]
    sensitivity=TP/float(TP+FN)
    specificity=TN/float(TN+FP)
    precision=TP/float(TP+FP)
    accuracy=(TP+TN)/float(TP+FN+TN+FP)
    accuracy_list.append(accuracy)
    f_measure=(2*precision*sensitivity)/float(precision+sensitivity)
    mathews_correlation_coeff=(TP*TN-FP*FN)/float((TP+FN)*(TP+FP)*(TN+FP)*(TN+FN))**2
    print("Random Forest Classifier Results for Testing data:")
    print(f'sensitivity {sensitivity} specificity {specificity} precision {precision} accuracy {accuracy} f_measure {f_measure} mathews_correlation_coeff {mathews_correlation_coeff}')
    print("\n\n")

#Bar chart visualization for accuracy score for each property.
def create_plot(property_name, accuracy_list):
    max_val=np.amax(accuracy_list)
    a=(np.array(accuracy_list)).reshape((4, 2)).T
    x=np.arange(4)
    plt.bar(x + 0.00, a[0], color = 'b', width = 0.25, label='SVM')
    plt.bar(x + 0.25, a[1], color = 'r', width = 0.25, label='Random Forest')
    labels = ['Training Data', 'Human Test', 'Cerevisiae Test', 'Thaliana Test']
    plt.xticks(x+0.122, labels, rotation='horizontal')
    plt.legend(loc="upper right")
    plt.title(property_name)
    ax = plt.gca()
    ax.set_ylim([0,max_val+0.2])
    plt.show()
    del accuracy_list[:]

def main():

    #Function calls for training and, apply trained model on test dataset and then create plot alltogether for one property at a time.
    
	#1st Physicochemical Property
    # function call - Hydrophobicity
    clf1, clf2=train("C:/Users/ADMIN/Desktop/CTD Files/CTD_test.txt", 21, 2780)
    test("C:/Users/ADMIN/Desktop/CTD Files/CTD11_human_test.txt", 21, 967, clf1, clf2)
    test("C:/Users/ADMIN/Desktop/CTD Files/CTD12_cerevisiae_test.txt", 21, 354, clf1, clf2)
    test("C:/Users/ADMIN/Desktop/CTD Files/CTD13_thaliana_test.txt", 21, 456, clf1, clf2)
    create_plot('Hydrophobicity', accuracy_list)
    
    #2nd Property
    # function call - Normalized Van der Waals volume
    clf3, clf4=train("C:/Users/ADMIN/Desktop/CTD Files/CTDVander.txt", 21, 2780)
    test("C:/Users/ADMIN/Desktop/CTD Files/CTDVanderHuman.txt", 21, 967, clf3, clf4)
    test("C:/Users/ADMIN/Desktop/CTD Files/CTDVanderCere.txt", 21, 354, clf3, clf4)
    test("C:/Users/ADMIN/Desktop/CTD Files/CTDVanderThaliana.txt", 21, 456, clf3, clf4)
    create_plot('Van der Waals', accuracy_list)
    
    #3rd Property
    # function call - Charge and Polarity of side chains 
    clf5, clf6=train("C:/Users/ADMIN/Desktop/CTD Files/CTDChain.txt", 30, 2780)
    test("C:/Users/ADMIN/Desktop/CTD Files/CTDChainHuman.txt", 30, 967, clf5, clf6)
    test("C:/Users/ADMIN/Desktop/CTD Files/CTDChainCere.txt", 30, 354, clf5, clf6)
    test("C:/Users/ADMIN/Desktop/CTD Files/CTDChainThaliana.txt", 30, 456, clf5, clf6)
    create_plot('Charge and Polarity', accuracy_list)
    
    #4th property
    # function call - Polarizability
    clf7, clf8=train("C:/Users/ADMIN/Desktop/CTD Files/CTDPolar.txt", 21, 2780)
    test("C:/Users/ADMIN/Desktop/CTD Files/CTDPolarHuman.txt", 21, 967, clf7, clf8)
    test("C:/Users/ADMIN/Desktop/CTD Files/CTDPolarCere.txt", 21, 354, clf7, clf8)
    test("C:/Users/ADMIN/Desktop/CTD Files/CTDPolarThaliana.txt", 21, 456, clf7, clf8)
    create_plot('Polarizability', accuracy_list)
    
    #5th property
    # function call - Polarity
    clf9, clf10=train("C:/Users/ADMIN/Desktop/CTD Files/CTDPolarity.txt", 21, 2780)
    test("C:/Users/ADMIN/Desktop/CTD Files/CTDPolarityHuman.txt", 21, 967, clf9, clf10)
    test("C:/Users/ADMIN/Desktop/CTD Files/CTDPolarityCere.txt", 21, 354, clf9, clf10)
    test("C:/Users/ADMIN/Desktop/CTD Files/CTDPolarityThaliana.txt", 21, 456, clf9, clf10)
    create_plot('Polarity', accuracy_list)
    
main()
