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


def plot_hog_image(image):
    fd, hog_image = hog(image, orientations=9, pixels_per_cell=(8, 8), 
                    cells_per_block=(2, 2), visualize=True, multichannel=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4), sharex=True, sharey=True) 

    ax1.imshow(image, cmap=plt.cm.gray) 
    ax1.set_title('Input image') 

    # Rescale histogram for better display 
    hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10)) 

    ax2.imshow(hog_image_rescaled, cmap=plt.cm.gray) 
    ax2.set_title('Histogram of Oriented Gradients')

def plotLBP_image(image):
    imgLBP = image
    vecimgLBP = imgLBP.flatten()

    fig = plt.figure(figsize=(20,8))
    ax  = fig.add_subplot(1,3,1)
    ax.imshow(image)
    ax.set_title("gray scale image")
    ax  = fig.add_subplot(1,3,2)
    ax.imshow(imgLBP,cmap="gray")
    ax.set_title("LBP converted image")
    ax  = fig.add_subplot(1,3,3)
    freq,lbp, _ = ax.hist(vecimgLBP,bins=2**8)
    ax.set_ylim(0,40000)
    lbp = lbp[:-1]
    ## print the LBP values when frequencies are high
    largeTF = freq > 5000
    for x, fr in zip(lbp[largeTF],freq[largeTF]):
        ax.text(x,fr, "{:6.0f}".format(x),color="red")
    ax.set_title("LBP histogram")
    plt.show()
    
def plot_eigenfaces(pca):
    fig, axes = plt.subplots(2,10,figsize=(15,3),
    subplot_kw={'xticks':[], 'yticks':[]},
    gridspec_kw=dict(hspace=0.01, wspace=0.01))
    for i, ax in enumerate(axes.flat):
#         ax.imshow(pca.components_[i].reshape(654,178),cmap="gray")
    #     ax.imshow(projected[i].reshape(436,178),cmap="binary")
        ax.imshow(pca.components_[i].reshape(218,178),cmap="gray")
    

def plot_pca_projections(pca,X_train):
    projected = pca.inverse_transform(X_train)
    fig, axes = plt.subplots(2,10,figsize=(15,3), subplot_kw={'xticks':[], 'yticks':[]},
                gridspec_kw=dict(hspace=0.01, wspace=0.01))
    for i, ax in enumerate(axes.flat):
        ax.imshow(projected[i].reshape(218,178),cmap="binary")
    #     ax.imshow(projected[i].reshape(436,178),cmap="binary")
    #     ax.imshow(projected[i].reshape(654,178),cmap="gray")
    


def plot_learning_curve(estimator, title, X, y, axes=None, ylim=None, cv=3,
                        n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5)):
    """
    Generate 3 plots: the test and training learning curve, the training
    samples vs fit times curve, the fit times vs score curve.

    Parameters
    ----------
    estimator : object type that implements the "fit" and "predict" methods
        An object of that type which is cloned for each validation.

    title : string
        Title for the chart.

    X : array-like, shape (n_samples, n_features)
        Training vector, where n_samples is the number of samples and
        n_features is the number of features.

    y : array-like, shape (n_samples) or (n_samples, n_features), optional
        Target relative to X for classification or regression;
        None for unsupervised learning.

    axes : array of 3 axes, optional (default=None)
        Axes to use for plotting the curves.

    ylim : tuple, shape (ymin, ymax), optional
        Defines minimum and maximum yvalues plotted.

    cv : int, cross-validation generator or an iterable, optional
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:

          - None, to use the default 5-fold cross-validation,
          - integer, to specify the number of folds.
          - :term:`CV splitter`,
          - An iterable yielding (train, test) splits as arrays of indices.

        For integer/None inputs, if ``y`` is binary or multiclass,
        :class:`StratifiedKFold` used. If the estimator is not a classifier
        or if ``y`` is neither binary nor multiclass, :class:`KFold` is used.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validators that can be used here.

    n_jobs : int or None, optional (default=None)
        Number of jobs to run in parallel.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

    train_sizes : array-like, shape (n_ticks,), dtype float or int
        Relative or absolute numbers of training examples that will be used to
        generate the learning curve. If the dtype is float, it is regarded as a
        fraction of the maximum size of the training set (that is determined
        by the selected validation method), i.e. it has to be within (0, 1].
        Otherwise it is interpreted as absolute sizes of the training sets.
        Note that for classification the number of samples usually have to
        be big enough to contain at least one sample from each class.
        (default: np.linspace(0.1, 1.0, 5))
    """
    
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
    '''
    == Input ==
    gray_image  : color image of shape (height, width)
    
    == Output ==  
    imgLBP : LBP converted image of the same shape as 
    '''
    X = pd.DataFrame(df["img_name"].values)
    y = pd.Series(df["gender"].values)

    X_train, X_test, y_train, y_test = train_test_split(X,
                                                        y,
                                                        test_size=.20,
                                                        random_state=12039393)

    X_train = create_feature_matrix(X_train[0],extraction)
    X_test = create_feature_matrix(X_test[0],extraction)

    # X_train = joblib.load("X_train_fusion.pkl")
    # X_test = joblib.load("X_test_fusion.pkl")
    
    
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
            image_features = img_id.flatten()
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


def create_lbp_features(img):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    LBP_image = np.zeros_like(gray_image)
    neighboor = 3 
    for ih in range(0,image.shape[0] - neighboor):
        for iw in range(0,image.shape[1] - neighboor):
            ### Step 1: 3 by 3 pixel
            img          = gray_image[ih:ih+neighboor,iw:iw+neighboor]
            center       = img[1,1]
            img01        = (img >= center)*1.0
            img01_vector = img01.T.flatten()
            ### Step 2: **Binary operation**:
            img01_vector = np.delete(img01_vector,4)
            ### Step 3: Decimal: Convert the binary operated values to a digit.
            where_img01_vector = np.where(img01_vector)[0]
            if len(where_img01_vector) >= 1:
                num = np.sum(2**where_img01_vector)
            else:
                num = 0
            LBP_image[ih+1,iw+1] = num
    return(LBP_image)
    

def create_combined_features(img):
    fd, hog_image = hog(img, orientations=9, pixels_per_cell=(8, 8), 
                    cells_per_block=(2, 2), visualize=True, multichannel=True, block_norm = "L2-Hys")
    hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))
    LBP = getLBP_image(img)/255
    feat = np.hstack([LBP, hog_image_rescaled]).flatten()
    return feat

def apply_pca(X_train,X_test):
    ss = StandardScaler()
    ss = ss.fit(X_train)
    X_train = ss.transform(X_train)
    X_test = ss.transform(X_test)
    
    pca = PCA(n_components = 2700)
    pca = pca.fit(X_train)
    plt.plot(np.cumsum(pca.explained_variance_ratio_))

    print('Initial train matrix shape is: ', X_train.shape)
    print('Initial test shape is: ', X_test.shape)
    X_train = pca.transform(X_train)
    X_test = pca.transform(X_test)
    print('PCA transformed train shape is: ', X_train.shape)
    print('PCA transformed test shape is: ', X_test.shape)
    plt.show()
    return X_train,X_test,pca


def train_test_LR():
    LR = LogisticRegression(C=0.01, max_iter=1500, solver='liblinear')
    LR.fit(X_train, y_train)
    y_pred_LR = LR.predict(X_test)
    print('LR Fitting accuracy'+"\n"+ '**************************')
    train_acc = LR.score(X_train,y_train)
    print(train_acc)
    print('LR Prediction accuracy'+"\n"+'**************************')
    test_acc = LR.score(X_test,y_test)
    print(test_acc)
    print("")
    print("************************************************************")
    print("                 LR classification report")
    print("************************************************************")
    print(classification_report(y_test, y_pred_LR))
    return y_test, y_pred_LR,train_acc,test_acc, LR


def plot_confusion_matrix(y_test,y_pred_LR):
    cm_LR=confusion_matrix(y_test,y_pred_LR)
    df_cm_LR = pd.DataFrame(cm_LR, index = [i for i in "cm"],columns = [i for i in "cm"])
    plt.figure()
    x_axis_labels = ['Actual Female','Actual Male'] # labels for x-axis
    y_axis_labels = ['Predicted Female','Predicted Male'] # labels for y-axis
    sns.heatmap(df_cm_LR, annot=True,xticklabels=x_axis_labels, yticklabels=y_axis_labels)

def plot_ROC():
    y_pred_proba_LR = LR.predict_proba(X_test)[:,1]
    fpr_LR, tpr_LR, thresholds_LR = roc_curve(y_test, y_pred_proba_LR)
    plt.plot([0,1],[0,1],'k--')
    plt.plot(fpr_LR,tpr_LR)
    plt.plot(fpr_LR,tpr_LR, color='blue',label = 'AUC = %0.2f' % auc_roc_LR)
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title("LR ROC curve")
    plt.legend(loc="lower right")
    plt.show()


df = load_A1_data()
X_train, X_test, y_train, y_test = data_partition(df,"hog")

# to save it
# X_train_file="X_train_hog.pkl"  
# joblib.dump(X_train, X_train_file)
# X_test_file="X_test_hog.pkl"  
# joblib.dump(X_test, X_test_file)
 
X_train, X_test,pca = apply_pca(X_train,X_test)

# plot_eigenfaces(pca,218,178)

# plot_pca_projections(pca,X_train)

plot_learning_curve (LogisticRegression(C=0.01, max_iter=1500, solver='liblinear'),
                                        "Learning curve for LR",
                                        X_train,y_train)

y_test, y_pred_LR,train_acc,test_acc, LR = train_test_LR()

plot_confusion_matrix(y_test,y_pred_LR)

auc_roc_LR= roc_auc_score(y_test,y_pred_LR)





