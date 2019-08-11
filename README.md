# iCOMPASS
A composite model for the diagnoses of AD. 
It's an implementation of paper, iCOMPASS: A computational model to predict the development of Alzheimer’s disease spectrum with clinical, genetic and image features.

We uesd data downloaded from Alzheimer's Disease Neuroimaging Initiative: ADNI.

Pandas, sklearn, matplotlib are needed in this project.

dataprocess.py is for data preprocessing. We used samples whose clinical and MRI data were both availiable in ADNI-1, GO and 2 dataset. 

svr.py is for the regression task, predicting MMSE changes. Both Pearson and Spearman correlation coefficients are calculated. Parameters were determined through gridsearch.

svm.py is the classification part, identifying those with significant declines in MMSE score. Area under ROC curve and precision-recall curve were adopted to measure the performance.
