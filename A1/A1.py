# python A1/A1.py
from PIL import Image
import os

import numpy as np
import pandas as pd
from pandas import DataFrame

import joblib

from keras.preprocessing.image import ImageDataGenerator
from keras.applications.inception_v3 import preprocess_input
import imageio
import cv2

# Import sklearn libraries for using a set of models and pre defined functions to prepare, train and test them
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn import linear_model
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV,learning_curve,ShuffleSplit

# Used to visualise and process images for better performance 
import matplotlib.pyplot as plt
from skimage.color import rgb2gray
from skimage.feature import hog
from skimage import exposure
import seaborn as sns
from progressbar import ProgressBar

# Used to standardise the date and undertake PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
 
# Performance metrics used to represent how well the model peroforms
from sklearn.metrics import accuracy_score
from sklearn.metrics import (confusion_matrix,roc_auc_score, precision_recall_curve, auc,
                             roc_curve, recall_score,accuracy_score, classification_report, f1_score,
                             precision_recall_fscore_support, log_loss)

import keras
import tensorflow as tf
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense


def plot_data_sample(df):
    plt.figure(figsize=(10,10))
    for i in range(25):
        plt.subplot(5,5,i+1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(df["img_name"].iloc[i])
        plt.xlabel(df["gender"].iloc[i])
    plt.show()

def plot_hog_image(image):
    image = Image.open("/Users/hzizi/Desktop/CW/dataset_AMLS_20-21/celeba/img/"+image)
    fd, hog_image = hog(image, orientations=9, pixels_per_cell=(8, 8), 
                    cells_per_block=(2, 2), visualize=True, multichannel=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4), sharex=True, sharey=True) 
    ax1.imshow(image) 
    ax1.set_title('Input image') 

    # Rescale histogram for better display 
    hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10)) 
    ax2.imshow(hog_image_rescaled) 
    ax2.set_title('Histogram of Oriented Gradients')
    plt.show()

    
def plot_eigenfaces(pca):
    fig, axes = plt.subplots(2,10,figsize=(15,3),
    subplot_kw={'xticks':[], 'yticks':[]},
    gridspec_kw=dict(hspace=0.01, wspace=0.01))
    for i, ax in enumerate(axes.flat):
#         ax.imshow(pca.components_[i].reshape(654,178),cmap="gray")
    #     ax.imshow(projected[i].reshape(436,178),cmap="binary")
        ax.imshow(pca.components_[i].reshape(218,178),cmap="gray")
    plt.show()


def plot_pca_projections(pca,X_train):
    projected = pca.inverse_transform(X_train)
    fig, axes = plt.subplots(2,10,figsize=(15, 3), subplot_kw={'xticks':[], 'yticks':[]},gridspec_kw=dict(hspace=0.01, wspace=0.01))
    for i, ax in enumerate(axes.flat):
        ax.imshow(projected[i].reshape(218,178),cmap="binary")
    #     ax.imshow(projected[i].reshape(436,178),cmap="binary")
    #     ax.imshow(projected[i].reshape(654,178),cmap="gray")
    plt.show()


def plot_confusion_matrix(y_test,y_pred):
    cm_LR=confusion_matrix(y_test,y_pred)
    df_cm_LR = pd.DataFrame(cm_LR, index = [i for i in "cm"],columns = [i for i in "cm"])
    plt.figure()
    x_axis_labels = ['Actual Female','Actual Male'] # labels for x-axis
    y_axis_labels = ['Predicted Female','Predicted Male'] # labels for y-axis
    sns.heatmap(df_cm_LR, annot=True,xticklabels=x_axis_labels, yticklabels=y_axis_labels)


def plot_ROC(model,auc_roc,X_test,y_test):
    y_pred_proba = model.predict_proba(X_test)[:,1]
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
    fig, ax = plt.subplots(1,1)
    ax.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % auc_roc)
    ax.plot([0, 1], [0, 1], 'k--')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('Receiver operating characteristic example')
    ax.legend(loc="lower right")
    plt.show()


def plot_learning_curve(estimator, title, X, y, axes=None, ylim=None, cv=3,
                        n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5)):
    if axes is None:
        _, axes = plt.subplots(1, 2, figsize=(20, 5))

    axes[0].set_title(title)
    if ylim is not None:
        axes[0].set_ylim(*ylim)
    axes[0].set_xlabel("Training examples")
    axes[0].set_ylabel("Score")

    train_sizes, train_scores, test_scores, fit_times, _ = \
        learning_curve(estimator, X, y, cv=cv, n_jobs=n_jobs,
                       train_sizes=train_sizes,
                       return_times=True)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    fit_times_mean = np.mean(fit_times, axis=1)
    fit_times_std = np.std(fit_times, axis=1)

    # Plot learning curve
    axes[0].grid()
    axes[0].fill_between(train_sizes, train_scores_mean - train_scores_std,
                         train_scores_mean + train_scores_std, alpha=0.1,
                         color="r")
    axes[0].fill_between(train_sizes, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, alpha=0.1,
                         color="g")
    axes[0].plot(train_sizes, train_scores_mean, 'o-', color="r",
                 label="Training score")
    axes[0].plot(train_sizes, test_scores_mean, 'o-', color="g",
                 label="Cross-validation score")
    axes[0].legend(loc="best")

    # Plot fit_time vs score
    axes[1].grid()
    axes[1].plot(fit_times_mean, test_scores_mean, 'o-')
    axes[1].fill_between(fit_times_mean, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, alpha=0.1)
    axes[1].set_xlabel("fit_times")
    axes[1].set_ylabel("Score")
    axes[1].set_title("Performance of the model")
    return plt


def plot_lbp_image(image):
    img = cv2.imread("/Users/hzizi/Desktop/CW/dataset_AMLS_20-21/celeba/img/"+image, 1) 
    img_lbp = create_lbp_features(img)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4), sharex=True, sharey=True) 
    ax1.imshow(img, cmap=plt.cm.gray) 
    ax1.set_title('Input image') 
    ax2.imshow(img_lbp, cmap=plt.cm.gray) 
    plt.show()


def load_A1_data():
    '''
    == Input ==
    gray_image  : color image of shape (height, width)
    
    == Output ==  
    imgLBP : LBP converted image of the same shape as 
    '''
    df= pd.read_csv("/Users/hzizi/Desktop/CW/dataset_AMLS_20-21/celeba/labels.csv")
    rows = []
    columns = []
    for i in [df.iloc[:,0]]:
        elements=(i.str.split())
    for data in elements:
        rows.append(data[1:4])
    for y in [df.columns[0]]:
        columns = (y.split())
#     original_dataset = DataFrame(rows,columns=columns)
    pbar = ProgressBar()
    for i in pbar(rows):
        i[0] = np.asarray(Image.open("/Users/hzizi/Desktop/CW/dataset_AMLS_20-21/celeba/img/"+i[0]))    
#         i[0] = crop_faces(i[0])

    df = DataFrame(rows,columns=columns)
    df["gender"] = pd.to_numeric(df["gender"])
    df["gender"] = df["gender"].replace(-1, 0)
    df = df.drop(df.columns[[2]], axis=1)
    return df 

def data_partition(df, extraction:str = ""):
    X = pd.DataFrame(df["img_name"].values)
    y = pd.Series(df["gender"].values)

    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=.20,random_state=12039393)
    
#     X_train = create_feature_matrix(X_train[0],extraction)
#     X_test = create_feature_matrix(X_test[0],extraction)

    X_train = joblib.load("X_train_combined.pkl")
    X_test = joblib.load("X_test_combined.pkl")

    # look at the distrubution of labels in the train set
#     print(pd.Series(y_train).value_counts())
#     print(pd.Series(y_test).value_counts())
    print("X_train has shape:", X_train.shape)
    print("y_train has shape:", y_train.shape)
    print("X_test has shape:", X_test.shape)
    print("y_test has shape:", y_test.shape)
    return X_train,X_test,y_train,y_test


def create_feature_matrix(label_dataframe,extraction):
    features_list = []
    pbar = ProgressBar()
    for img_id in pbar(label_dataframe):
        # get features for image
        if extraction == "hog":
            image_features = create_hog_features(img_id)
            features_list.append(image_features)
        if extraction == "lbp":
            image_features = create_lbp_features(img_id)
            features_list.append(image_features)
        if extraction == "combined":
            image_features = create_combined_features(img_id)
            features_list.append(image_features)
        if extraction == "flat":
            image_features = rgb2gray(img_id).flatten()
            features_list.append(image_features)
        if extraction == "unchanged":
            image_features = img_id/255
            features_list.append(image_features)
            
    # convert list of arrays into a matrix
    feature_matrix = np.array(features_list)
    return feature_matrix


def create_hog_features(img):
    fd, hog_image = hog(img, orientations=9, pixels_per_cell=(8, 8), 
                    cells_per_block=(2, 2), visualize=True, multichannel=True, block_norm = "L2-Hys")
    hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))
    flat_features = np.hstack(hog_image_rescaled).flatten()
    return flat_features


def get_pixel(img, center, x, y): 
    adjusted_value = 0
    try: 
#       Set to 1 if greated than the central pixel value else set it to 0
        if img[x][y] >= center: 
            adjusted_value = 1
    except: 
#       Used to bypass the event where a pixel value is null i.e. values in boundaries. 
        pass
    return adjusted_value 


def lbp_calculated_pixel(img, x, y): 
    center = img[x][y] 
    val_ar = [] 

    # top_left 
    val_ar.append(get_pixel(img, center, x-1, y-1)) 
    # top 
    val_ar.append(get_pixel(img, center, x-1, y)) 
    # top_right 
    val_ar.append(get_pixel(img, center, x-1, y + 1)) 
    # right 
    val_ar.append(get_pixel(img, center, x, y + 1)) 
    # bottom_right 
    val_ar.append(get_pixel(img, center, x + 1, y + 1)) 
    # bottom 
    val_ar.append(get_pixel(img, center, x + 1, y)) 
    # bottom_left 
    val_ar.append(get_pixel(img, center, x + 1, y-1)) 
    # left 
    val_ar.append(get_pixel(img, center, x, y-1)) 

    # Now, we need to convert binary values to decimal 
    power_val = [1, 2, 4, 8, 16, 32, 64, 128] 
    val = 0
    for i in range(len(val_ar)): 
        val += val_ar[i] * power_val[i] 
    return val 


def create_lbp_features(img):
    height, width, _ = img.shape 
#   Convert to graysclae beacause the has only one channel . 
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
#   Get array of  same height and width as the RGB image 
    img_lbp = np.zeros((height, width), np.uint8) 
    for i in range(0, height): 
        for j in range(0, width): 
            img_lbp[i, j] = lbp_calculated_pixel(img_gray, i, j) 
    return img_lbp


def data_partition_validate(df,extraction,augmentation=False):
    X = pd.DataFrame(df["img_name"].values)
    y = pd.Series(df["gender"].values)
    X_train, X_test, y_train, y_test = train_test_split(X,
                                                        y,
                                                        test_size=.20,
                                                        random_state=1234123)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=1)
    
    if augmentation is True:
        X_train,y_train = image_generator(X_train,y_train)
    
    X_train = create_feature_matrix(X_train[0],extraction)
    X_test = create_feature_matrix(X_test[0],extraction)
    X_val = create_feature_matrix(X_val[0],extraction)

    # look at the distrubution of labels in the train set
    print("X_train has shape:", X_train.shape)
    print("y_train has shape:", y_train.shape)
    print("")
    print("X_test has shape:", X_test.shape)
    print("y_test has shape:", y_test.shape)
    print("")
    print("X_val has shape:", X_val.shape)
    print("y_val has shape:", y_val.shape)
    
    return X_train,X_test,X_val,y_train,y_test,y_val


def train_validate_CNN(summary:bool=False, epoch:int=15):
    
    model = Sequential()
    model.add(Conv2D(16, 3, input_shape=(218,178 ,3)))
    keras.layers.Dropout(0.5),
    model.add(Activation('relu'))
    model.add(MaxPooling2D())

    model.add(Conv2D(32, 3))
    keras.layers.Dropout(0.5),
    model.add(Activation('relu'))
    model.add(MaxPooling2D())

    model.add(Conv2D(64, 3))
    keras.layers.Dropout(0.5),
    model.add(Activation('relu'))
    model.add(MaxPooling2D())

    model.add(Conv2D(128, 3))
    keras.layers.Dropout(0.5),
    model.add(Activation('relu'))
    model.add(MaxPooling2D())

    model.add(Conv2D(256, 3))
    keras.layers.Dropout(0.5),
    model.add(Activation('relu'))
    model.add(MaxPooling2D())

    model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
    model.add(Dense(256))
    keras.layers.Dropout(0.5),

    model.add(Activation('relu'))
    model.add(Dense(2))
    keras.layers.Dropout(0.5),
    model.add(Activation('softmax'))

    model.compile(loss='binary_crossentropy',
                 optimizer='adam',
                 metrics=['accuracy'])

    if summary == True:
        # Take a look at the model summary
        model.summary()

    history = model.fit(X_train, to_categorical(y_train.values.ravel()), epochs=epoch, 
                        validation_data=(X_val, to_categorical(y_val.values.ravel())))

    return history, model,epoch


def CNN_learning_curve(history,epoch):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs_range = range(epoch)

    plt.figure(figsize=(16, 4))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.show()
    
def CNN_predict():
    y_pred = model.predict(X_test)
    class_names=[0, 1]
    score = tf.nn.softmax(y_pred[0])
    temp = []
    for i in y_pred:
        score = tf.nn.softmax(i)
        i = class_names[np.argmax(score)]
        temp.append(i)
    return temp

    
def create_combined_features(img):
    fd, hog_image = hog(img, orientations=9, pixels_per_cell=(8, 8), 
                    cells_per_block=(2, 2), visualize=True, multichannel=True, block_norm = "L2-Hys")
    hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))
    LBP = create_lbp_features(img)/255
    feat = np.hstack([LBP, hog_image_rescaled]).flatten()
    return feat


def apply_pca(X_train,X_test,plot:bool=False):
    ss = StandardScaler()
    ss = ss.fit(X_train)
    X_train = ss.transform(X_train)
    X_test = ss.transform(X_test)
    
    pca = PCA(n_components = 2700)
    pca = pca.fit(X_train)

    print('Initial train matrix shape is: ', X_train.shape)
    print('Initial test shape is: ', X_test.shape)
    X_train = pca.transform(X_train)
    X_test = pca.transform(X_test)
    print('PCA transformed train shape is: ', X_train.shape)
    print('PCA transformed test shape is: ', X_test.shape)
    if plot is True:
        plt.plot(np.cumsum(pca.explained_variance_ratio_))
        plt.show()
    return X_train,X_test,pca


def grid_search_tuning(model,X_train,y_train):
    print("Hyperparameter Tuning using 5-folds validation")
    if model == "SVM":
        grid= {'kernel':('linear', 'sigmoid'), 'C': [0.1, 1, 10, 100], 
                   'gamma': [0.01, 0.001,0.0001,0.00001]}
        tuned_model = GridSearchCV(SVC(), grid, verbose=1)
        tuned_model.fit(X_train, y_train)
    if model == "LR":
        grid ={'C':[0.001,.009,0.01,.09,1],'solver':['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']}
        tuned_model = GridSearchCV(LogisticRegression(max_iter = 1500), grid, verbose=1)
        tuned_model.fit(X_train, y_train)
    if model == "KNN":
        grid = {'n_neighbors':  range(1,40)}
        tuned_model =  GridSearchCV(KNeighborsClassifier(),grid,verbose=1)
        tuned_model.fit(X_train, y_train)

    print("Best Parameters:\n", tuned_model.best_params_)
    print("Best Estimators:\n", tuned_model.best_estimator_)
        

def train_test(model,X_train,y_train,X_test,y_test):
    model = model
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print('Fitting accuracy'+"\n"+ '**************************')
    train_acc = model.score(X_train,y_train)
    print(train_acc)
    print('Prediction accuracy'+"\n"+'**************************')
    test_acc = model.score(X_test,y_test)
    print(test_acc)
    print("")
    print("************************************************************")
    print("                 LR classification report")
    print("************************************************************")
    print(classification_report(y_test, y_pred))
    return y_pred,train_acc,test_acc, model

def image_generator(X_train,y_train):

    augmentation =ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.25,
    height_shift_range=0.25,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
    )
    augmented_arr = []
    augmented_labels = []

    for arr,label in zip(np.array(X_train.iloc[0:4000,:]), np.array(y_train.iloc[0:4000])):
        aug_iter = augmentation.flow(np.expand_dims(arr[0],0))
        aug_images = [next(aug_iter)[0].astype(np.uint8) for i in range(2)]
        for i in aug_images:
            augmented_arr.append([i])
            augmented_labels.append(label)   

    augmented_arr = DataFrame(augmented_arr)
    augmented_labels = DataFrame(augmented_labels)
    X_train = pd.concat([X_train,augmented_arr])
    y_train = pd.concat([y_train,augmented_labels])
    return X_train,y_train


# Load csv, extract coma separated values for rows and columns to create a clean dataframe
df = load_A1_data()
# plot_data_sample(df)
X_train, X_test, y_train, y_test = data_partition(df,"combined")

X_train, X_test,pca = apply_pca(X_train,X_test,plot=False)
# plot_eigenfaces(pca)
# plot_pca_projections(pca,X_train)


# Run this code to return results for Logistic regression
LR =  LogisticRegression(C=0.001, max_iter=1500, solver='saga')
# plot_learning_curve (LR,"Learning curve for LR",X_train,y_train)
# grid_search_tuning("LR",X_train,y_train)
y_pred_LR,train_acc_LR,test_acc_LR, LR = train_test(LR,X_train,y_train,X_test,y_test)
auc_roc_LR= roc_auc_score(y_test,y_pred_LR)
# plot_ROC(LR,auc_roc_LR,X_test,y_test)
# plot_confusion_matrix(y_test,y_pred_LR)


# # Run this code to return results for Support Vector Machines
# SVM =  SVC(C=1, gamma=1e-05, kernel='sigmoid',probability=True)
# # plot_learning_curve (SVM,"Learning curve for SVM",X_train,y_train)
# # grid_search_tuning("SVM",X_train,y_train)
# y_pred_SVM,train_acc_SVM,test_acc_SVM, SVM = train_test(SVM,X_train,y_train,X_test,y_test)
# auc_roc_SVM= roc_auc_score(y_test,y_pred_SVM)
# # plot_ROC(SVM,auc_roc_SVM,X_test,y_test)
# # plot_confusion_matrix(y_test,y_pred_SVM)


# # Run this code to return results for KNN
# KNN = KNeighborsClassifier(n_neighbors = 38)
# plot_learning_curve (KNN,"Learning curve for KNN",X_train,y_train)
# # grid_search_tuning("KNN",X_train,y_train)
# y_pred_KNN,train_acc_KNN,test_acc_KNN, KNN = train_test(KNN,X_train,y_train,X_test,y_test)
# auc_roc_KNN= roc_auc_score(y_test,y_pred_KNN)
# plot_ROC(KNN,auc_roc_KN,X_test,y_test)
# plot_confusion_matrix(y_test,y_pred_KNN)


# # Run this code to return results for CNN
# X_train, X_test,X_val,y_train, y_test, y_val = data_partition_validate(df,"unchanged", augmentation=False)
# history, model,epoch = train_validate_CNN(epoch=6)
# CNN_learning_curve(history,6)
# y_pred_CNN = CNN_predict()
# plot_confusion_matrix(y_test,y_pred_CNN)
# auc_roc_CNN= roc_auc_score(y_test,y_pred_CNN)
# plot_ROC(model,auc_roc_CNN,X_test,y_test)
# print(classification_report(y_test, y_pred_CNN))
