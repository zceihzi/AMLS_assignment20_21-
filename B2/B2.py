# python B2/B2.py
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

import keras
import tensorflow as tf
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense


def plot_mean_image():
    """
    This function loads all the images in the cartoon_set folder and plots the mean image by aggregating all of them.   
    ----------
    Parameters: No parameters expected as it reads images from a folder 
    ----------
    """
# Access all PNG files in directory
    path = "Dataset/cartoon_set/img/"
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
    plt.show()
    
def plot_eigenfaces(pca):
    """
    Generates a visualisation of a sample of 10 images after applying PCA  
    ----------
    Parameters:
    pca: The PCA model that was trained on the train set and used to transform the dataset
    ----------
    """
    fig, axes = plt.subplots(2,10,figsize=(15,3),
    subplot_kw={'xticks':[], 'yticks':[]},
    gridspec_kw=dict(hspace=0.01, wspace=0.01))
    for i, ax in enumerate(axes.flat):
        ax.imshow(pca.components_[i].reshape(200,270),cmap="gray")
    plt.show()

def plot_pca_projections(pca,X_train):
    """
    Generates a visualisation of a sample of 10 reconstructed images from the components generated from PCA 
    ----------
    Parameters:
    pca: The PCA model that was trained on the train set and used to transform the dataset
    --------
    """
    projected = pca.inverse_transform(X_train)
    fig, axes = plt.subplots(2,10,figsize=(15, 3), subplot_kw={'xticks':[], 'yticks':[]},gridspec_kw=dict(hspace=0.01, wspace=0.01))
    for i, ax in enumerate(axes.flat):
        ax.imshow(projected[i].reshape(200,270),cmap="gray")
    plt.show()
    
def plot_confusion_matrix(y_test,y_pred):
    """
    Generates a confusion matrix to analyse the prediction of a classifier  
    ----------
    Parameters:
    y_test: The labels for the test set created for a given dataset
    y_pred: An array of numbers generated from the classifier's predictions
    ----------
    """
    cm=confusion_matrix(y_test,y_pred)
    x_axis_labels = ['Actual 0','Actual 1','Actual 2','Actual 3','Actual 4'] # labels for x-axis
    y_axis_labels = ['Predicted 0','Predicted 1','Predicted 2','Predicted 3','Predicted 4'] # labels for y-axis
    sns.heatmap(cm,annot=True,xticklabels=x_axis_labels, yticklabels=y_axis_labels)
    plt.xlim(0, cm.shape[0])
    plt.ylim(0, cm.shape[1])
    plt.show()

def plot_learning_curve(estimator, title, X, y):
    """
    Generates a plot of the training and validation curves during learning
    ----------
    Parameters:
    estimator: The model defined to solve the classification problem
    title: A string that represents the overall graph's title
    X: Typically the set of images we want to use to train our model
    y: The label of each image in X
    ----------
    """
    train_sizes= np.linspace(.1, 1.0, 5)
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
    plt.show()

def plot_glass_filter_efficiency(num:int):
    """
    Generates a visualisation of the result of the clustering algorithm
    ----------
    Parameters:
    num: Integer (between 0 and 3) that correspond to one of the clusters to plot
    ----------
    """
    temp =df[df.glasses == num]
    fig, axes = plt.subplots(20,20,figsize=(9,9), subplot_kw={'xticks':[], 'yticks':[]},
                gridspec_kw=dict(hspace=0.01, wspace=0.01))
    for i, ax in enumerate(axes.flat):
        ax.imshow(temp["file_name"].values[i])
    plt.show()

def plot_data_sample(df):
    """
    Generates a visualisation of the dataset by plotting a sample of 10 images with their label. 
    ----------
    Parameters:
    df: The preprocessed dataframe where images were opened and converted into vectors
    ----------
    """ 
    plt.figure(figsize=(7,10))
    for i in range(20):
        plt.subplot(5,5,i+1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(df["file_name"].iloc[i])
        plt.xlabel(df["eye_color"].iloc[i])
    plt.show()


def crop_images(image,folder):
    img = Image.open("Datasets/"+folder+"/img/"+str(image)).convert('RGB')
    # Setting the points for cropped image 
    left = 160
    top = 200
    right = 340
    bottom = 300
    # Cropped image of above dimension 
    # (It will not change orginal image) 
    cropped = img.crop((left, top, right, bottom))
    return cropped

def load_B2_data(folder):
    """
    This function loads the raw csv provided and extracts the label of interest given what task we are solving.
    It returns a dataframe with a set of vectorised images and their corresponding label
    ----------
    Parameters:
    folder: A string that refers to the name of teh folder we want to use. Eg: "celeba"/"celeba_test"/"cartoon_set"/"cartoon_set_test"
    ----------
    """
    df = pd.read_csv("Datasets/" +folder+ "/labels.csv")
    rows = []
    columns = []
    for i in [df.iloc[:,0]]:
        elements=(i.str.split())
    for data in elements:
        rows.append(data[1:4])
    for y in [df.columns[0]]:
        columns = (y.split())
    original_dataset = DataFrame(rows,columns=columns)
    pbar = ProgressBar()
    for i in pbar(rows):
        i[2] = np.asarray(crop_images(i[2],folder))    
        
    df = DataFrame(rows,columns=columns)
    df["eye_color"] = pd.to_numeric(df["eye_color"])
    df = df.drop(df.columns[[1]], axis=1)
    return df, original_dataset

def create_hog_features(img):
    """
    Extracts hog features from an image and returns a processed 1D vector 
    ----------
    Parameters:
    img: A vectorised image of typically 3 dimensions in our case
    ----------
    """
    fd, hog_image = hog(img, orientations=9, pixels_per_cell=(8, 8), 
                    cells_per_block=(2, 2), visualize=True, multichannel=True, block_norm = "L2-Hys")
    hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))
    flat_features = np.hstack(hog_image_rescaled).flatten()
    return flat_features

def create_feature_matrix(label_dataframe,extraction="flat"):
    """
    Extracts features from each images from a dataframe and returns a numpy array of processed images  
    ----------
    Parameters:
    label_dataframe: The preprocessed dataset containing vectorised images
    extraction: Extraction method to be used. Please not that "flat" consists simply of taking the 
                original image and return a 1D vector from it. "unchanged" is used when predicting
                using a NN. This option simply normalises the data as it will be flattened later.
    ----------
    """
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
            image_features = img_id.flatten()
            features_list.append(image_features)
    # convert list of arrays into a matrix
    feature_matrix = np.array(features_list)
    return feature_matrix

def find_eye(image):
    """
    This function uses the crop module from PILLOW to crop picture to the eye area and extract hog features for clustering
    ----------
    Parameters:
    image: String referring to the file name. Eg: "1.png"
    ----------
    """
    img = Image.open("Datasets/cartoon_set/img/"+str(image)).convert('RGB')
    # Setting the points for cropped image 
    left = 170
    top = 230
    right = 240
    bottom = 300
    # Cropped image of above dimension 
    # (It will not change orginal image) 
    cropped = img.crop((left, top, right, bottom))
#     plt.imshow(cropped, cmap=plt.cm.binary)
    cropped = np.array(cropped)
    fd, hog_image = hog(cropped, orientations=9, pixels_per_cell=(8, 8), 
                    cells_per_block=(2, 2), visualize=True, multichannel=True, block_norm = "L2-Hys")
    hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))
# get all non black Pixels
    return hog_image_rescaled

def add_glass_label(df,original_dataset,plot:bool=False):
    """
    This function uses Kmeans clustering to drop images with opaque classes and plots the returned dataframe as well as 
    a visualisation of how the eye is extracted and processed by the algorithm
    ----------
    Parameters:
    df: The obtained dataframe after preprocessing
    plot: Boolean that specifies whether visualisations of the cropping and kmeans shoudl be returned
    ----------
    """
    cropped_images = []
    temp = []

    images = original_dataset["file_name"]
    pbar = ProgressBar()
    for i in pbar(images):
        cropped_images.append(find_eye(i).flatten())
        temp.append(find_eye(i))

    cropped_images = np.array(cropped_images)    
    print(" K_means in progress to identify back glasses ...")
    # create kmeans object
    kmeans = KMeans(n_clusters=3)
    # fit kmeans object to data
    kmeans.fit(cropped_images)
    # print location of clusters learned by kmeans object
    print(kmeans.cluster_centers_)
    # save new clusters for chart
    y_km = kmeans.fit_predict(cropped_images)
    df["glasses"] = y_km

    if plot is True:
        plt.figure(figsize=(10,10))
        for i in range(25):
            plt.subplot(5,5,i+1)
            plt.xticks([])
            plt.yticks([])
            plt.grid(False)
            plt.imshow(temp[i], cmap=plt.cm.binary)

        plt.show()

        plt.figure(figsize=(10,10))
        for i in range(25):
            plt.subplot(5,5,i+1)
            plt.xticks([])
            plt.yticks([])
            plt.grid(False)
            plt.imshow(df["file_name"].values[i], cmap=plt.cm.binary)
            plt.xlabel(y_km[i])
        plt.show()
    return df

def data_partition(df,extraction:str="flat"):
    """
    Splits data into train and test sets using a ratio of 80/20
    ----------
    Parameters:
    df: The preprocessed dataset containing vectorised images
    extraction: A string specifying which feature extraction method must be used by 
                the function create_feature_matrix(). Eg: "hog", "lbp", "combined"
    ----------
    """
    X = pd.DataFrame(df["file_name"].values)
    y = pd.Series(df["eye_color"].values)
   
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=.20,random_state=12039393)

    new_test_data,original_new_test_data = load_B2_data("cartoon_set_test")
    new_test_data = add_glass_label(new_test_data,original_new_test_data,plot=False)
    res = pd.Series(new_test_data["glasses"]).value_counts().to_dict()
    new_test_data = new_test_data[new_test_data.glasses != min(res, key=lambda k: res[k])]
    
    new_X_test = pd.DataFrame(new_test_data["file_name"].values)
    new_y_test = pd.Series(new_test_data["eye_color"].values)
    
    print("Feature extraction for X_train in progress ...")
    X_train = create_feature_matrix(X_train[0],extraction)
    print("Feature extraction for X_test in progress ...")
    X_test = create_feature_matrix(X_test[0],extraction)
    print("Feature extraction for the additional new_X_test in progress ...")
    new_X_test = create_feature_matrix(new_X_test[0],extraction)

    print("Overall class distribution in the training set ")
    print(pd.Series(y_train).value_counts())
    print("Overall class distribution in the test")
    print(pd.Series(y_test).value_counts())
    print("Overall class distribution in the additional test")
    print(pd.Series(new_y_test).value_counts())
    print("")
    print("X_train has shape:", X_train.shape)
    print("y_train has shape:", y_train.shape)
    print("X_test has shape:", X_test.shape)
    print("y_test has shape:", y_test.shape)
    print("new_X_test has shape:", X_test.shape)
    print("new_y_test has shape:", y_test.shape)
    return X_train,X_test,new_X_test,y_train,y_test,new_y_test

def apply_pca(X_train,X_test,new_X_test,plot:bool=False):
    """
    This function Standardises the data and fit/transforms train and test data into sets with lower dimensional vectors
    ----------
    Parameters: 
    X_train: The training set of the given data
    X_test: The test set of the given data
    plot: Boolean that specifies whether to display the cumulative variance curve or not
    ----------
    """
    ss = StandardScaler()
    ss = ss.fit(X_train)
    X_train = ss.transform(X_train)
    X_test = ss.transform(X_test)
    new_X_test = ss.transform(new_X_test)
    
    pca = PCA(n_components = 100)
    pca = pca.fit(X_train)
   
    print("")
    print('Initial train shape is: ', X_train.shape)
    print('Initial test shape is: ', X_test.shape)
    print('Initial additional test shape is: ', new_X_test.shape)

    X_train = pca.transform(X_train)
    X_test = pca.transform(X_test)
    new_X_test = pca.transform(new_X_test)
    print('PCA transformed train shape is: ', X_train.shape)
    print('PCA transformed test shape is: ', X_test.shape)
    print('PCA transformed additional test shape is: ', new_X_test.shape)
    if plot is True:
        plt.plot(np.cumsum(pca.explained_variance_ratio_))
        plt.show()
    return X_train,X_test,new_X_test,pca

def data_partition_validate(df,extraction):
    """
    This function is the same as data_partition() but is used to retreive data from our original df and normalise images 
    so they can be used later by our CNN. The other difference is that it defines an extra validation set to be used by our 
    Deep learning model.
    ----------
    Parameters:
    df: The preprocessed dataset containing vectorised images
    extraction: A string specifying which feature extraction method must be used by the function create_feature_matrix(). 
    Eg: "hog", "lbp", "combined". For this function it must be set to "unchanged"
    ----------
    """
    X = pd.DataFrame(df["file_name"].values)
    y = pd.Series(df["eye_color"].values)
    X_train, X_test, y_train, y_test = train_test_split(X,
                                                        y,
                                                        test_size=.20,
                                                        random_state=1234123)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=1)
    
    new_test_data,original_new_test_data = load_B2_data("cartoon_set_test")
    new_test_data = add_glass_label(new_test_data,original_new_test_data,plot=False)
    res = pd.Series(new_test_data["glasses"]).value_counts().to_dict()
    new_test_data = new_test_data[new_test_data.glasses != min(res, key=lambda k: res[k])]
    
    new_X_test = pd.DataFrame(new_test_data["file_name"].values)
    new_y_test = pd.Series(new_test_data["eye_color"].values)
    
    X_train = create_feature_matrix(X_train[0],extraction)
    X_test = create_feature_matrix(X_test[0],extraction)
    X_val = create_feature_matrix(X_val[0],extraction)
    new_X_test = create_feature_matrix(new_X_test[0],extraction)
    
    print("Overall class distribution in the training set ")
    print(pd.Series(y_train).value_counts())
    print("Overall class distribution in the test")
    print(pd.Series(y_test).value_counts())
    print("Overall class distribution in the additional test")
    print(pd.Series(new_y_test).value_counts())
    print("")
    print("X_train has shape:", X_train.shape)
    print("y_train has shape:", y_train.shape)
    print("X_test has shape:", X_test.shape)
    print("y_test has shape:", y_test.shape)
    print("X_val has shape:", X_val.shape)
    print("y_val has shape:", y_val.shape)
    print("new_X_test has shape:", X_test.shape)
    print("new_y_test has shape:", y_test.shape)
    return X_train,X_test,new_X_test,X_val,y_train,y_test,new_y_test,y_val


def train_validate_CNN(summary:bool=False, epoch:int=15):
    """
    This function defines our CNN as well as trains it using a validation set.
    ----------
    Parameters:
    summary: A boolean that specifies if a model summary needs to be shown
    epoch: The number of epochs to use to run our model
    ----------
    """
    model = Sequential()
    model.add(Conv2D(16, 3, input_shape=(100, 180, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D())

    model.add(Conv2D(32, 3))
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

    #Fully connected 1st layer
    model.add(Flatten())  # this converts our 3D feature maps to 1D feature vector
    model.add(Dense(256))
    model.add(Activation('relu'))
    
    #Fully connected 2nd layer
    model.add(Dense(5))
    model.add(Activation('softmax'))

# Take a look at the model summary

    model.compile(loss='categorical_crossentropy',
                     optimizer='adam',
                     metrics=['accuracy'])

    if summary == True:
        # Take a look at the model summary
        model.summary()

    history = model.fit(X_train, to_categorical(y_train.values.ravel()), epochs=epoch, 
                        validation_data=(X_val, to_categorical(y_val.values.ravel())))

    return history, model

def CNN_learning_curve(history,epoch=15):
    """
    This function plots the learning curve of teh CNN during training.
    ----------
    Parameters:
    history: Historical data collected during training returned by train_validate_CNN() 
    epoch: The number of epochs used to run our model
    ----------
    """
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
    plt.show()
    
def multi_class_auc(model,y_test,X_test,classes):
    temp = []
    for class_lab in classes:
        temp.append(roc_auc_score(np.where(y_test==class_lab, 1,0), 
                      [val[class_lab] for val in model.predict_proba(X_test)]))
    return sum(temp) / len(temp)

def CNN_predict(X_test):
    """
    This function takes the probability of the CNN's decision for each image and returns the label with the highest one.
    ----------
    Parameters: No parameters expected
    ----------
    """
    y_pred = model.predict(X_test)
    class_names=[0, 1, 2, 3, 4]
    score = tf.nn.softmax(y_pred[0])
    temp = []
    for i in y_pred:
        score = tf.nn.softmax(i)
        i = class_names[np.argmax(score)]
        temp.append(i)
    return temp
        
def train_test(model,X_train,y_train,X_test,y_test,new_X_test,new_y_test):
    """
    This function uses a given model to train and return classification decisions
    ----------
    Parameters: 
    model: The model to tune. A different grid is used depending on this variable
    X_train: The train set of the given data
    y_train: The labels corresponding to X_train
    X_test: The test set of the given data
    y_test: The labels corresponding to X_test
    ----------
    """
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
    print("                 Classification report")
    print("************************************************************")
    print(classification_report(y_test, y_pred))
    print("")
    print('Prediction accuracy for the additional test set'+"\n"+'**************************')
    y_pred2 = model.predict(new_X_test)
    test_acc2 = model.score(new_X_test,new_y_test)
    print(test_acc2)
    print("")
    print("************************************************************")
    print("       Classification report for the additional test set")
    print("************************************************************")
    print(classification_report(new_y_test, y_pred2))
    return y_pred,y_pred2,train_acc,test_acc,test_acc2, model


classes = [0,1,2,3,4]
df,original_dataset = load_B2_data("cartoon_set")
df = add_glass_label(df,original_dataset,plot=False)
res = pd.Series(df["glasses"]).value_counts().to_dict()
df = df[df.glasses != min(res, key=lambda k: res[k])]
# plot_mean_image()
# plot_data_sample(df)
# plot_glass_filter_efficiency(min(res, key=lambda k: res[k]))


X_train, X_test, new_X_test, y_train, y_test, new_y_test = data_partition(df,"flat")
X_train, X_test, new_X_test, pca = apply_pca(X_train,X_test,new_X_test,plot=False)
# plot_eigenfaces(pca)
# plot_pca_projections(pca,X_train)


LR =  LogisticRegression(max_iter=10000,multi_class='multinomial')
print("Results for Logistic Regression in task B2 :")
print("")
y_pred_LR,y_pred2_LR,train_acc_LR,test_acc_LR,test_acc2_LR, LR = train_test(LR,X_train,y_train,X_test,y_test,new_X_test,new_y_test)
auc_roc_LR= multi_class_auc(LR,y_test,X_test,classes)
auc_roc2_LR= multi_class_auc(LR,new_y_test,new_X_test,classes)
# plot_learning_curve (LR,"Learning curve for LR",X_train,y_train)
# print("Plots for the original test set")
# plot_confusion_matrix(y_test,y_pred_LR)
# print("Plots for the additional test set")
# plot_confusion_matrix(new_y_test,y_pred2_LR)


# SVM = SVC(probability=True)
# print("Results for Support Vector Machines in task B2 :")
# print("")
# y_pred_SVM,y_pred2_SVM,train_acc_SVM,test_acc_SVM,test_acc2_SVM, SVM = train_test(SVM,X_train,y_train,X_test,y_test,new_X_test,new_y_test)
# auc_roc_SVM= multi_class_auc(SVM,y_test,X_test,classes)
# auc_roc2_SVM= multi_class_auc(SVM,new_y_test,new_X_test,classes)
# # plot_learning_curve (SVM,"Learning curve for SVM",X_train,y_train)
# # print("Plots for the original test set")
# # plot_confusion_matrix(y_test,y_pred_SVM)
# # print("Plots for the additional test set")
# # plot_confusion_matrix(new_y_test,y_pred2_SVM)



# KNN = KNeighborsClassifier(n_neighbors = 32)
# print("Results for KNN in task B1 :")
# print("")
# y_pred_KNN,y_pred2_KNN,train_acc_KNN,test_acc_KNN,test_acc2_KNN, KNN = train_test(KNN,X_train,y_train,X_test,y_test,new_X_test,new_y_test)
# auc_roc_KNN= multi_class_auc(KNN,y_test,X_test,classes)
# auc_roc2_KNN= multi_class_auc(KNN,new_y_test,new_X_test,classes)
# # plot_learning_curve (KNN,"Learning curve for Tuned KNN",X_train,y_train)
# # print("Plots for the original test set")
# # plot_confusion_matrix(y_test,y_pred_KNN)
# # print("Plots for the additional test set")
# # plot_confusion_matrix(new_y_test,y_pred2_KNN)



# X_train, X_test,new_X_test,X_val,y_train, y_test,new_y_test, y_val = data_partition_validate(df,"unchanged")
# print("Results for CNN in task B2 :")
# print("")
# history, model = train_validate_CNN(epoch=3)
# y_pred_CNN = CNN_predict(X_test)
# train_acc_CNN = history.history["accuracy"][-1]
# test_acc_CNN = accuracy_score(y_test, y_pred_CNN)
# auc_roc_CNN= multi_class_auc(model,y_test,X_test,classes)
# print("AUC CNN for initial test set :", auc_roc_CNN)
# print("************************************************************")
# print("                 Classification report")
# print("************************************************************")
# print(classification_report(y_test, y_pred_CNN))
# y_pred2_CNN = CNN_predict(new_X_test)
# test_acc2_CNN = accuracy_score(new_y_test, y_pred2_CNN)
# print("************************************************************")
# print("       Classification report for the additional test set")
# print("************************************************************")
# print(classification_report(new_y_test, y_pred2_CNN))
# auc_roc2_CNN= multi_class_auc(model,new_y_test,new_X_test,classes)
# print("AUC CNN for the additional test set:", auc_roc2_CNN)
# # CNN_learning_curve(history,len(history.history['accuracy']))
# # print("")
# # print("Plots for the test set")
# # plot_confusion_matrix(y_test,y_pred_CNN)
# # print("")
# # print("Plots for the test set")
# # plot_confusion_matrix(new_y_test,y_pred2_CNN)