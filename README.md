# ELEC0134-Assignment

This folder project consists of four main folders. For each task, 4 implementations were tested and described in the report. 

For each task a common structure can be observed. 

1- Data is loaded, partitionned and processed using PCA. Please note the the functions : plot_data_sample() - plot_eigenfaces() - plot_pca_projections(pca,X_train) were commented out as they slowed down the process of returned final results. In addition. depending on the examinator environment, the matplotlib backend might lead the terminal to crash. If you have decided to uncomment any function that involves plotting, please make sure you close the plot window to allow the code to continue.  

2- The four blocks of code that are visible correspond to one model each. Logistic regression was left uncommented by default as it was the fastest and most performing model. Please note again that all plot functions were commented out for convenience and avoid terminal to freze due to different setups. (This problem is actually an open issue on github that has not been fixed yet)

3- The uncommented block consists of declaring the model to be used and obtain all the metrics to display after fitting and running the model on the train and test data. 


NOTE : 

- Please note that the task that consumes most power and time if feature extraction which takes an average of 22 minutes to terminate. Each of the two subtasks for problem A take approximately 40 minutes to complete. The other two subtasks for task B take approximately 20 minutes to end each. 

- In order to obtain results for all tasks, please run the main.py file. This file will automatically execute the other files and return a table with the results of logistic regression for each task both for the initial test set and for the new one provided one week before the deadline. 

- Alternatively you can run each file separatle for example : python A1/A1.py. This will execute the file individually. The results and observations are printed for all files in case you opt for this alternative

- The functions were also commented for a better understanding 


