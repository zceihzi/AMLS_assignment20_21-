# python B1/B1.py
from PIL import Image
import os

import numpy as np
import pandas as pd
from pandas import DataFrame
import joblib


# Import sklearn libraries for using a set of models and pre defined functions to prepare, train and test them
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn import linear_model
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV,learning_curve,ShuffleSplit
from sklearn.cluster import KMeans

# Used to visualise and process images for better performance 
import matplotlib.pyplot as plt
from skimage.color import rgb2gray
from skimage.feature import hog
from skimage import exposure
import seaborn as sns
from progressbar import ProgressBar

# Used to standardise the date and undertake PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler,label_binarize
from sklearn.decomposition import PCA
 
# Performance metrics used to represent how well the model peroforms
from sklearn.metrics import accuracy_score
from sklearn.metrics import (confusion_matrix,roc_auc_score, precision_recall_curve, auc,
                             roc_curve, recall_score,accuracy_score, classification_report, f1_score,
                             precision_recall_fscore_support, log_loss)

import tensorflow as tf
import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.inception_v3 import preprocess_input
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense


def plot_mean_image():
# Access all PNG files in directory
    path = "/Users/hzizi/Desktop/CW/dataset_AMLS_20-21/cartoon_set/img/"
    allfiles=os.listdir(path)
    imlist=[filename for filename in allfiles]
    N=len(imlist)

    # Assuming all images are the same size, get dimensions of first image
    w,h=Image.open(path+imlist[0]).size

    # Create a numpy array of floats to store the average (assume RGB images)
    arr=np.zeros((h,w,3),np.float)

    # Build up average pixel intensities, casting each image as an array of floats
    for im in imlist:
        imarr=np.array(Image.open(path+im).convert('RGB'),dtype=np.float)
        arr=arr+imarr/N

    # Round values in array and cast as 8-bit integer
    arr=np.array(np.round(arr),dtype=np.uint8)

    # Generate, save and preview final image
    out=Image.fromarray(arr,mode="RGB")
    plt.imshow(out)
    
    
def plot_eigenfaces(pca):
    fig, axes = plt.subplots(2,10,figsize=(15,3),
    subplot_kw={'xticks':[], 'yticks':[]},
    gridspec_kw=dict(hspace=0.01, wspace=0.01))
    for i, ax in enumerate(axes.flat):
        ax.imshow(pca.components_[i].reshape(290,260),cmap="gray")
    plt.show()


def plot_pca_projections(pca,X_train):
    projected = pca.inverse_transform(X_train)
    fig, axes = plt.subplots(2,10,figsize=(15, 3), subplot_kw={'xticks':[], 'yticks':[]},gridspec_kw=dict(hspace=0.01, wspace=0.01))
    for i, ax in enumerate(axes.flat):
        ax.imshow(projected[i].reshape(290,260),cmap="gray")
    plt.show()
    
    
def plot_confusion_matrix(y_test,y_pred):
    cm=confusion_matrix(y_test,y_pred)

    x_axis_labels = ['Actual 0','Actual 1','Actual 2','Actual 3','Actual 4'] # labels for x-axis
    y_axis_labels = ['Predicted 0','Predicted 1','Predicted 2','Predicted 3','Predicted 4'] # labels for y-axis

    sns.heatmap(cm,annot=True,xticklabels=x_axis_labels, yticklabels=y_axis_labels)
    plt.xlim(0, cm.shape[0])
    plt.ylim(0, cm.shape[1])
    plt.show()


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

def plot_learning_curve(estimator, title, X, y):
    """
    Generates a plot of the training and validation curves during training
    ----------
    Parameters:
    estimator: The model defined to solve the classification problem
    title: A string that represents the overall graph's title
    X: Typically the set of images we want to use to train our model
    y: The label of each image in X
    ----------
    """
    train_sizes=np.linspace(.1, 1.0, 5)
    cv=3
    fig, ax = plt.subplots(1,1)
    ax.set_title(title)
    ax.set_xlabel("Training examples")
    ax.set_ylabel("Score")

    train_sizes, train_scores, test_scores, fit_times, _ = \
        learning_curve(estimator, X, y, cv=cv,
                       train_sizes=train_sizes,
                       return_times=True)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    fit_times_mean = np.mean(fit_times, axis=1)
    fit_times_std = np.std(fit_times, axis=1)

    # Plot learning curve
    ax.grid()
    ax.fill_between(train_sizes, train_scores_mean - train_scores_std,
                         train_scores_mean + train_scores_std, alpha=0.1,
                         color="r")
    ax.fill_between(train_sizes, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, alpha=0.1,
                         color="g")
    ax.plot(train_sizes, train_scores_mean, 'o-', color="r",
                 label="Training score")
    ax.plot(train_sizes, test_scores_mean, 'o-', color="g",
                 label="Cross-validation score")
    ax.legend(loc="best")
    return plt


def plot_data_sample(df):
    plt.figure(figsize=(10,10))
    for i in range(25):
        plt.subplot(5,5,i+1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(df["file_name"].iloc[i])
        plt.xlabel(df["face_shape"].iloc[i])
    plt.show()
    
    
def crop_images(image):
    img = Image.open("/Users/hzizi/Desktop/CW/dataset_AMLS_20-21/cartoon_set/img/"+str(image)).convert('RGB')
    # Setting the points for cropped image 
    left = 120
    top = 100
    right = 380
    bottom = 390
    # Cropped image of above dimension 
    # (It will not change orginal image) 
    cropped = img.crop((left, top, right, bottom))
    return cropped


def load_B1_data(folder):
    """
    This function loads the raw csv provided and extracts the label of interest given what task we are solving.
    It returns a dataframe with a vectorised image of shape (218,178,3) and their corresponding label
    ----------
    Parameters:
    folder: A string that refers to the name of teh folder we want to use. Eg: "celeba"/"celeba_test"/"cartoon_set"/"cartoon_set_test"
    ----------
    """
    df= pd.read_csv("/Users/hzizi/Desktop/CW/dataset_AMLS_20-21/" +folder+ "/labels.csv")
    rows = []
    columns = []
    for i in [df.iloc[:,0]]:
        elements=(i.str.split())
    for data in elements:
        rows.append(data[1:4])
    for y in [df.columns[0]]:
        columns = (y.split())
    pbar = ProgressBar()
    for i in pbar(rows):
        i[2] = np.asarray(crop_images(i[2]))       
    df = DataFrame(rows,columns=columns)

    df["face_shape"] = pd.to_numeric(df["face_shape"])
    df = df.drop(df.columns[[0]], axis=1)
    return df 


def create_hog_features(img):
    fd, hog_image = hog(img, orientations=9, pixels_per_cell=(8, 8), 
                    cells_per_block=(2, 2), visualize=True, multichannel=True, block_norm = "L2-Hys")
    hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))
    flat_features = np.hstack(hog_image_rescaled).flatten()
    return flat_features


def create_feature_matrix(label_dataframe,extraction="flat"):
    features_list = []
    pbar = ProgressBar()
    for img_id in pbar(label_dataframe):
        if extraction == "unchanged":
            image_features = img_id/255
            features_list.append(image_features)
        if  extraction == "hog":
            image_features = create_hog_features(img_id)
            features_list.append(image_features)
        if extraction == "flat":
            image_features = rgb2gray(img_id).flatten()
            features_list.append(image_features)
    # convert list of arrays into a matrix
    feature_matrix = np.array(features_list)
    return feature_matrix


def data_partition(df,extraction:str="flat"):
    X = pd.DataFrame(df["file_name"].values)
    y = pd.Series(df["face_shape"].values)

    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=.20,random_state=12039393)

    X_train = create_feature_matrix(X_train[0],extraction)
    X_test = create_feature_matrix(X_test[0],extraction)

    
    # look at the distrubution of labels in the train set

    print("X_train has shape:", X_train.shape)
    print("y_train has shape:", y_train.shape)
    print("X_test has shape:", X_test.shape)
    print("y_test has shape:", y_test.shape)
    return X_train,X_test,y_train,y_test


def apply_pca(X_train,X_test,plot:bool=False):
    ss = StandardScaler()
    ss = ss.fit(X_train)
    X_train = ss.transform(X_train)
    X_test = ss.transform(X_test)
    
    pca = PCA(n_components = 100)
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


def data_partition_validate(df,extraction):
    X = pd.DataFrame(df["file_name"].values)
    y = pd.Series(df["face_shape"].values)
    X_train, X_test, y_train, y_test = train_test_split(X,
                                                        y,
                                                        test_size=.20,
                                                        random_state=1234123)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=1)
    
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
    model.add(Conv2D(32, 3, input_shape=(290, 260, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D())

    model.add(Conv2D(64, 3))
    model.add(Activation('relu'))
    model.add(MaxPooling2D())

    model.add(Conv2D(128, 3))
    model.add(Activation('relu'))
    model.add(MaxPooling2D())

    model.add(Conv2D(256, 3))
    model.add(Activation('relu'))
    model.add(MaxPooling2D())

    model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
    model.add(Dense(256))

    model.add(Activation('relu'))
    model.add(Dense(5))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy',
                 optimizer='adam',
                 metrics=['accuracy'])

    if summary == True:
        # Take a look at the model summary
        model.summary()

    history = model.fit(X_train, to_categorical(y_train.values.ravel()), epochs=epoch, 
                        validation_data=(X_val, to_categorical(y_val.values.ravel())))

    return history, model,epoch


def CNN_learning_curve(history,epoch=15):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs_range = range(epoch)

    plt.figure(figsize=(16, 8))
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
    class_names=[0, 1, 2, 3, 4]
    score = tf.nn.softmax(y_pred[0])
    temp = []
    for i in y_pred:
        score = tf.nn.softmax(i)
        i = class_names[np.argmax(score)]
        temp.append(i)
    return temp
        
        
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