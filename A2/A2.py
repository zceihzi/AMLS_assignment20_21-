# python A2/A2.py
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

import tensorflow as tf
import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.inception_v3 import preprocess_input
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense


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
        plt.imshow(df["img_name"].iloc[i])
        plt.xlabel(df["smiling"].iloc[i])
    plt.show()

def plot_hog_image(image):
    """
    Generates a visualisation of converting an image using the hog descriptor. 
    ----------
    Parameters:
    image: A string refering to the file name (image) to plot. Eg: "1.jpg"
    ----------
    """
    image = np.array(Image.open("Datasets/celeba/img/"+image))
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
    """
    Generates a visualisation of a sample of 10 images after applying PCA  
    ----------
    Parameters:
    pca: The PCA model that was trained on the train set and used to transform the dataset
    ----------
    """
    fig, axes = plt.subplots(2,5,figsize=(15,4),
    subplot_kw={'xticks':[], 'yticks':[]},
    gridspec_kw=dict(hspace=0.01, wspace=0.01))
    for i, ax in enumerate(axes.flat):
        ax.imshow(pca.components_[i].reshape(218,356),cmap="binary")
    plt.show()

def plot_pca_projections(pca,X_train):
    """
    Generates a visualisation of a sample of 10 reconstructed images from the components generated from PCA 
    ----------
    Parameters:
    pca: The PCA model that was trained on the train set and used to transform the dataset
    ----------
    """
    projected = pca.inverse_transform(X_train)
    fig, axes = plt.subplots(2,5,figsize=(15, 4), subplot_kw={'xticks':[], 'yticks':[]},gridspec_kw=dict(hspace=0.01, wspace=0.05))
    for i, ax in enumerate(axes.flat):
        ax.imshow(projected[i].reshape(218,356),cmap="binary")
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
    df_cm = pd.DataFrame(cm, index = [i for i in "cm"],columns = [i for i in "cm"])
    plt.figure()
    x_axis_labels = ['Actual Female','Actual Male'] # labels for x-axis
    y_axis_labels = ['Predicted Female','Predicted Male'] # labels for y-axis
    sns.heatmap(df_cm, annot=True,xticklabels=x_axis_labels, yticklabels=y_axis_labels)
    plt.show()

def plot_ROC(model,auc_roc,X_test,y_test):
    """
    Generates a plot of the ROC curve to visualise model performance given a calcuted area under the curve 
    ----------
    Parameters:
    model: The model defined to solve the classification problem
    auc_roc: The area under the curve calculated from comparing the classifier's prediction and the actual labels
    X_test: Vectorised images of the holdout set that has never been used 
    y_test: An array containing the actual label of each test image
    ----------
    """
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
    Generates a plot of the training and validation curves during learning
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

def plot_lbp_image(image):
    """
    Converts and generates a plot of the transformation of an image using LBP descriptors
    ----------
    Parameters:
    image: A string refering to the file name (image) to plot. Eg: "1.jpg"    
    ----------
    """
    img = np.array(Image.open("Datasets/celeba/img/"+image))
    img_lbp = create_lbp_features(img)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4), sharex=True, sharey=True) 
    ax1.imshow(img, cmap=plt.cm.gray) 
    ax1.set_title('Input image') 
    ax2.imshow(img_lbp, cmap=plt.cm.gray) 
    plt.show()

def image_generator(X_train,y_train):
    """
    This function uses the training set to generate synthetic images using rotation and other augmentation techniques.
    A new Training set and labels are returned including the new images.
    ----------
    Parameters: 
    X_train: The train set of the given data
    y_train: The labels corresponding to X_train
    ----------
    """
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

def load_A2_data(folder):
    """
    This function loads the raw csv provided and extracts the label of interest given what task we are solving.
    It returns a dataframe with vectorised images of shape (218,178,3) and their corresponding label
    ----------
    Parameters:
    folder: A string that refers to the name of teh folder we want to use. Eg: "celeba"/"celeba_test"/"cartoon_set"/"cartoon_set_test"
    ----------
    """
    df= pd.read_csv("Datasets/" +folder+ "/labels.csv")
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
        i[0] = np.asarray(Image.open("Datasets/" +folder+ "/img/"+i[0]))    

    df = DataFrame(rows,columns=columns)
    df["smiling"] = pd.to_numeric(df["smiling"])
    df["smiling"] = df["smiling"].replace(-1, 0)
    df = df.drop(df.columns[[1]], axis=1)
    return df 

def data_partition(df, extraction:str = ""):
    """
    Splits data into train and test sets using a ratio of 80/20
    ----------
    Parameters:
    df: The preprocessed dataset containing vectorised images
    extraction: A string specifying which feature extraction method must be used by 
                the function create_feature_matrix(). Eg: "hog", "lbp", "combined"
    ----------
    """
    X = pd.DataFrame(df["img_name"].values)
    y = pd.Series(df["smiling"].values)

    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=.20,random_state=12039393)
    
    new_test_data = load_A2_data("celeba_test")
    new_X_test = pd.DataFrame(new_test_data["img_name"].values)
    new_y_test = pd.Series(new_test_data["smiling"].values)
    
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

def create_feature_matrix(label_dataframe,extraction:str="combined"):
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
        # get features for image
        if extraction == "hog":
            image_features = create_hog_features(img_id)
            features_list.append(image_features)
        if extraction == "lbp":
            image_features = create_lbp_features(img_id).flatten()
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

def create_combined_features(img):
    """
    Extracts both HOG and LBP descriptors and creates feature vectors that combine the information of both feature types
    ----------
    Parameters:
    img: A vectorised image of typically 3 dimensions in our case
    ----------
    """
    fd, hog_image = hog(img, orientations=9, pixels_per_cell=(8, 8), 
                    cells_per_block=(2, 2), visualize=True, multichannel=True, block_norm = "L2-Hys")
    hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))
    LBP = create_lbp_features(img)/255
    feat = np.hstack([LBP, hog_image_rescaled]).flatten()
    return feat

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

def get_pixel(img, center, x, y): 
    """
    Consists of the main step of lbp extraction. Compares value of pixel to a central pixel and assigns 0 or 1. 
    ----------
    Parameters: (Please not that this function is used by the create_lbp_features. It does not need to be used separately.
                Arguments will therefore automatically be populated when used inside the other function)
    img: A vectorised image of typically 3 dimensions in our case
    center: the central position of our matrix
    x: horizontal coordinate of the image
    y: vertical coordinates of the image
    ----------
    """
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
    """
    Calculates the LBP value for a given block. (Process described in litterature review)
    ----------
    Parameters: (Please not that this function is used by the create_lbp_features. It does not need to be used separately.
                Arguments will therefore automatically be populated when used inside the other function)
    img: A vectorised image of typically 3 dimensions in our case
    x: horizontal coordinate of the image
    y: vertical coordinates of the image
    ----------
    """
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
    """
    Uses the above function to generate lbp features for a given image.
    ----------
    Parameters:
    img: A vectorised image of typically 3 dimensions in our case
    ----------
    """
    height, width, _ = img.shape 
#   Convert to graysclae beacause the has only one channel . 
    img_gray = rgb2gray(img)
#   Get array of  same height and width as the RGB image 
    img_lbp = np.zeros((height, width), np.uint8) 
    for i in range(0, height): 
        for j in range(0, width): 
            img_lbp[i, j] = lbp_calculated_pixel(img_gray, i, j) 
    return img_lbp

def data_partition_validate(df,extraction,augmentation=False):
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
    X = pd.DataFrame(df["img_name"].values)
    y = pd.Series(df["smiling"].values)
    X_train, X_test, y_train, y_test = train_test_split(X,
                                                        y,
                                                        test_size=.20,
                                                        random_state=1234123)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=1)
    
    new_test_data = load_A2_data("celeba_test")
    new_X_test = pd.DataFrame(new_test_data["img_name"].values)
    new_y_test = pd.Series(new_test_data["smiling"].values)
    
    if augmentation is True:
        X_train,y_train = image_generator(X_train,y_train)
    
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
    model.add(Conv2D(16, 3, input_shape=(218,178 ,3)))
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

    return history, model

def CNN_learning_curve(history,epoch):
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
    
def CNN_predict(X_test):
    """
    This function takes the probability of the CNN's decision for each image and returns the label with the highest one.
    ----------
    Parameters: No parameters expected
    ----------
    """
    y_pred = model.predict(X_test)
    class_names=[0, 1]
    score = tf.nn.softmax(y_pred[0])
    temp = []
    for i in y_pred:
        score = tf.nn.softmax(i)
        i = class_names[np.argmax(score)]
        temp.append(i)
    return temp

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
    
    pca = PCA(n_components = 3200)
    pca = pca.fit(X_train)
   
    print("")
    print('Initial train shape is: ', X_train.shape)
    print('Initial test shape is: ', X_test.shape)
    print('Initial additional test shape is: ', X_test.shape)

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

def grid_search_tuning(model,X_train,y_train):
    """
    This function uses grid search to hyper tune the main parameters of a given model
    ----------
    Parameters: 
    model: The model to tune. A different grid is used depending on this variable
    X_train: The train set of the given data
    y_train: The labels corresponding to X_train
    ----------
    """
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

print("TASK A2 starting:")
df = load_A2_data("celeba")
X_train, X_test, new_X_test, y_train, y_test, new_y_test = data_partition(df,"combined")
X_train, X_test, new_X_test, pca = apply_pca(X_train,X_test,new_X_test,plot=False)
# plot_data_sample(df)
# plot_eigenfaces(pca)
# plot_pca_projections(pca,X_train)

# # grid_search_tuning("LR",X_train,y_train)
LR =  LogisticRegression(C=0.001, max_iter=1500, solver='newton-cg')
print("Results for Logistic Regression in task A2 :")
print("")
y_pred_LR,y_pred2_LR,train_acc_LR,test_acc_LR,test_acc2_LR, LR = train_test(LR,X_train,y_train,X_test,y_test,new_X_test,new_y_test)
auc_roc_LR= roc_auc_score(y_test,y_pred_LR)
auc_roc2_LR= roc_auc_score(new_y_test,y_pred2_LR)
# plot_learning_curve (LR,"Learning curve for LR",X_train,y_train)
# print("Plots for the original test set")
# plot_ROC(LR,auc_roc_LR,X_test,y_test)
# plot_confusion_matrix(y_test,y_pred_LR)
# print("Plots for the additional test set")
# plot_ROC(LR,auc_roc2_LR,new_X_test,new_y_test)
# plot_confusion_matrix(new_y_test,y_pred2_LR)



# # # grid_search_tuning("SVM",X_train,y_train)
# SVM = SVC(C=1, gamma=1e-05, kernel='sigmoid',probability=True)
# print("Results for Support Vector Machines in task A2:")
# print("")
# y_pred_SVM,y_pred2_SVM,train_acc_SVM,test_acc_SVM,test_acc2_SVM, SVM = train_test(SVM,X_train,y_train,X_test,y_test,new_X_test,new_y_test)
# auc_roc_SVM= roc_auc_score(y_test,y_pred_SVM)
# auc_roc2_SVM= roc_auc_score(new_y_test,y_pred2_SVM)
# # plot_learning_curve (SVM,"Learning curve for SVM",X_train,y_train)
# # print("Plots for the original test set")
# # plot_ROC(SVM,auc_roc_SVM,X_test,y_test)
# # plot_confusion_matrix(y_test,y_pred_SVM)
# # print("Plots for the additional test set")
# # plot_ROC(SVM,auc_roc2_SVM,new_X_test,new_y_test)
# # plot_confusion_matrix(new_y_test,y_pred2_SVM)
# # # # grid_search_tuning("KNN",X_train,y_train)



# # grid_search_tuning("KNN",X_train,y_train)
# KNN = KNeighborsClassifier(n_neighbors = 38)
# print("Results for KNN in task A2:")
# print("")
# y_pred_KNN,y_pred2_KNN,train_acc_KNN,test_acc_KNN,test_acc2_KNN, KNN = train_test(KNN,X_train,y_train,X_test,y_test,new_X_test,new_y_test)
# auc_roc_KNN= roc_auc_score(y_test,y_pred_KNN)
# auc_roc2_KNN= roc_auc_score(new_y_test,y_pred2_KNN)
# # plot_learning_curve (KNN,"Learning curve for KNN",X_train,y_train)
# # print("Plots for the original test set")
# # plot_ROC(KNN,auc_roc_KNN,X_test,y_test)
# # plot_confusion_matrix(y_test,y_pred_KNN)
# # print("Plots for the additional test set")
# # plot_ROC(KNN,auc_roc2_KNN,new_X_test,new_y_test)
# # plot_confusion_matrix(new_y_test,y_pred2_KNN)


# X_train, X_test,new_X_test,X_val,y_train, y_test,new_y_test, y_val = data_partition_validate(df,"unchanged", augmentation=False)
# print("Results for CNN in task A2 :")
# print("")
# history, model = train_validate_CNN(epoch=15)
# y_pred_CNN = CNN_predict(X_test)
# train_acc_CNN = history.history["accuracy"][-1]
# test_acc_CNN = accuracy_score(y_test, y_pred_CNN)
# auc_roc_CNN= roc_auc_score(y_test,y_pred_CNN)
# print("AUC CNN:", auc_roc_CNN)
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
# auc_roc2_CNN= roc_auc_score(new_y_test,y_pred2_CNN)
# print("AUC for second test set CNN:", auc_roc2_CNN)
# # CNN_learning_curve(history,len(history.history['accuracy']))
# # print("")
# # print("Plots for the test set")
# # plot_ROC(model,auc_roc_CNN,X_test,y_test)
# # plot_confusion_matrix(y_test,y_pred_CNN)
# # print("")
# # print("Plots for the test set")
# # print("AUC CNN for the additional test set:", auc_roc2_CNN)
# # plot_ROC(model,auc_roc2_CNN,new_X_test,new_y_test)
# # plot_confusion_matrix(new_y_test,y_pred2_CNN)