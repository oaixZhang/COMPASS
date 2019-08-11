# iCOMPASS
A composite model for the diagnoses of AD. 
It's an implementation of paper, iCOMPASS: A computational model to predict the development of Alzheimer’s disease spectrum with clinical, genetic and image features.

We uesd ADNI data in this project.
dataprocess.py is for data preprocessing.
svr.py is for the regression task, predicting MMSE changes.
svm.py is the classification part, identifying those with significant declines in MMSE score.
