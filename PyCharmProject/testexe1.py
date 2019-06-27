from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import cv2
import os
import h5py
import glob
from matplotlib import pyplot
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.externals import joblib
import matplotlib.pyplot as plt

train_path = 'training'
num_trees = 100
bins = 8
test_size = 0.1
seed = 9

def fd_hu_moments(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    feature = cv2.HuMoments(cv2.moments(image)).flatten()
    return feature

def fd_histogram(image, mask=None):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([image], [0,1,2], None, [bins, bins, bins], [0, 256,0,256,0,256])
    cv2.normalize(hist, hist)
    return hist.flatten()

train_labels = os.listdir(train_path)
train_labels.sort()
print(train_labels)
global_features, labels = [], []
i = 0
j=0
for training_name in train_labels:
    dir = os.path.join(train_path, training_name)
    current_label = training_name
    k=1
    for filename in glob.glob("training\\" + current_label + "\*.jpg"):
        image = cv2.imread(filename)
        print(filename)
        image = cv2.resize(image,(100,100))
        hu_moments = fd_hu_moments(image)
        histogram = fd_histogram(image)
        global_feature = np.hstack([histogram, hu_moments])
        labels.append((current_label))
        global_features.append(global_feature)
        i+=1
        k+=1
        print('Processing folder: ', format((current_label)))
        j+=1
print('\nComplex feature extrac...')

print('feature vector size: ', format(np.array(global_features).shape))
print('training labels size: ', format(np.array(labels).shape))
targetNames = np.unique(labels)
le = LabelEncoder()
target = le.fit_transform(labels)


scaler = MinMaxScaler(feature_range=(0, 1))
rescaled_feature = scaler.fit_transform(global_features)
hf5_data = h5py.File('data.h5', 'w')
hf5_data.create_dataset('dataset_1', data=np.array(rescaled_feature))
hf5_labels = h5py.File('label.h5', 'w')
hf5_labels.create_dataset('dataset_1', data=np.array(target))
hf5_labels.close()
hf5_data.close()



models = []
# models.append(('LR', LogisticRegression(random_state=9)))
# models.append(('LDA', LinearDiscriminantAnalysis()))
# models.append(('KNN', KNeighborsClassifier()))
# models.append(('DTC', DecisionTreeClassifier(random_state= 9)))
# models.append(('RF', RandomForestClassifier(n_estimators=num_trees, random_state=9)))
# models.append(('NB', GaussianNB()))
# models.append(('SVC', SVC(random_state=9)))

hf5_data = h5py.File('data.h5', 'r')
hf5_labels = h5py.File('label.h5', 'r')
global_feature_string = hf5_data['dataset_1']
global_label_string = hf5_labels['dataset_1']
global_feature = np.array(global_feature_string)
global_label = np.array(global_label_string)
hf5_labels.close()
hf5_data.close()
print('feature shape; ',format(global_feature))
print('label shape: ', format(global_label))

(trainData, testData, trainLabel, testLabel) = train_test_split(np.array(global_features),
                                                                np.array(global_label),
                                                                test_size= test_size,
                                                                random_state=seed)
print('\nTrain Data: ', format(trainData.shape))
print('Test Data: ', format(testData.shape))
print('Train Label: ', format(trainLabel.shape))
print('Test Label: ', format(testLabel.shape))

fix_size = tuple((100,100))
model = RandomForestClassifier(n_estimators=num_trees, random_state=9)
model.fit(trainData,trainLabel)
test_path = 'test'
for file in glob.glob('test/*.jpg'):
    image = cv2.imread(file)
    image = cv2.resize(image,fix_size )
    hu_moments = fd_hu_moments(image)
    histogram = fd_histogram(image)
    global_feature = np.hstack([histogram, hu_moments])
    prediction = model.predict(global_feature.reshape(1,-1))[0]
    cv2.putText(image, train_labels[prediction], (20,30), cv2.FONT_HERSHEY_SIMPLEX, 1.0 ,(0,255,255),3)
    plt.imshow(cv2.cvtColor(image,cv2.COLOR_BGR2RGB))
    plt.show()

kfold = KFold(n_splits=10, random_state=7)
cv_results = cross_val_score(model, trainData, trainLabel, cv=kfold, scoring= "accuracy")
print("ACCURACY: ", cv_results.mean())