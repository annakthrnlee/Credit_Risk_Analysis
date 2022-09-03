# Credit_Risk_Analysis

## Overview: 
Credit risk is an inherently unbalanced classification problem, as good loans easily outnumber risky loans. Using my skills in data preparation, statistical reasoning, and machine learning I employed different techniques to train and evaluate models with unbalanced classes. 

- Using the credit card credit dataset from LendingClub (can be found under my resources folder), a peer-to-peer lending services company, I oversampled the data using the RandomOverSampler function and SMOTE algorithms. Then, I undersampled the data using the ClusterCentroids algorithm. 
- Next, I used a combinatorial approach of over- and undersampling using the SMOTEENN algorithm. 
- Finally, I compared two new machine learning models that reduce bias, using both BalancedRandomForestClassifier and EasyEnsembleClassifier functions, to predict credit risk.

## Resources:
- Data (CSV) source: LoanStats_2019Q1.csv.zip
- Software: Python 3.9.7 and Jupyter Notebook (double chcek versions)

## Results: balanced accuracy scores and the precision and recall scores of all six machine learning models. 
First, let's review what exactly we're looking at in each model.
- Balanced accuracy score: Is a machine learning error metric for binary and multi-class classification models.
- Precision: Also known as positive predictive value (PPV), is a measure of how reliable a positive classification. Precision is obtained by dividing the number of true positives (TP) by the number of all positives (i.e., the sum of true positives and false positives, or TP + FP). Precision = TP/(TP + FP).
- Recall scores: Also known as sensitivity, Is a measure of how well a machine learning model can detect positive instances. Sensitivity = TP/(TP + FN)
- F1 Score: Also called the harmonic mean, can be characterized as a single summary statistic of precision and sensitivity. F1 score = 2(Precision * Sensitivity)/(Precision + Sensitivity).

### Balance accuracy scores: 
First, let's review the scores of the oversampling (naive random oversampling) machine learning model. 

<img width="546" alt="Screen Shot 2022-09-02 at 11 02 48 AM" src="https://user-images.githubusercontent.com/104043438/188203263-77922411-f0a4-4348-b6c6-b9854d8a2470.png">
<img width="498" alt="Screen Shot 2022-09-02 at 11 03 18 AM" src="https://user-images.githubusercontent.com/104043438/188203324-d49f5fa4-70be-46cc-a178-acb3d39144e7.png">
<img width="808" alt="Screen Shot 2022-09-02 at 11 03 30 AM" src="https://user-images.githubusercontent.com/104043438/188203357-75d55a39-3c7b-4307-b8bb-a04e9e1596d2.png">

For this model: 
- Balanced accuracy score was = 0.6314677834584286 or 0.63 
- Precision: After reviewing the confusion matrix (which is the table of true positives, false positives, true negatives, and false negatives), let's break it down: I created this chart for simplicity reasons.
- The high_risk precision score is 0.01 (1%) with a 0.58 (58%) sensitivity score. On the contrary, low_risk precision has a score of 1.00 (100%) with a sensitivity score of 0.81 (81%). 
<img width="694" alt="Screen Shot 2022-09-02 at 11 20 55 AM" src="https://user-images.githubusercontent.com/104043438/188205892-75e36e5e-069f-483a-a328-7120eccc6bd9.png">

