# Credit_Risk_Analysis

## Overview: 
Credit risk is an inherently unbalanced classification problem, as good loans easily outnumber risky loans. Using my skills in data preparation, statistical reasoning, and machine learning I employed different techniques to train and evaluate models with unbalanced classes. 

- Using the credit card credit dataset from LendingClub (can be found under my resources folder), a peer-to-peer lending services company, I oversampled the data using the RandomOverSampler function and SMOTE algorithms. Then, I undersampled the data using the ClusterCentroids algorithm. 
- Next, I used a combinatorial approach of over- and undersampling using the SMOTEENN algorithm. 
- Finally, I compared two new machine learning models that reduce bias, using both BalancedRandomForestClassifier and EasyEnsembleClassifier functions, to predict credit risk.

## Resources:
- Data (CSV) source: LoanStats_2019Q1.csv.zip
- Software: Python 3.9.7 and Jupyter Notebook (double chcek versions)

## Results: 
First, let's review what exactly we're looking at in each model.
- Balanced accuracy score: Is a machine learning error metric for binary and multi-class classification models.
- Precision: Also known as positive predictive value (PPV), is a measure of how reliable a positive classification. Precision is obtained by dividing the number of true positives (TP) by the number of all positives (i.e., the sum of true positives and false positives, or TP + FP). Precision = TP/(TP + FP).
- Recall scores: Also known as sensitivity, Is a measure of how well a machine learning model can detect positive instances. Sensitivity = TP/(TP + FN)
- F1 Score: Also called the harmonic mean, can be characterized as a single summary statistic of precision and sensitivity. F1 score = 2(Precision * Sensitivity)/(Precision + Sensitivity).

### Oversampling (naive random oversampling) machine learning model
<img width="546" alt="Screen Shot 2022-09-02 at 11 02 48 AM" src="https://user-images.githubusercontent.com/104043438/188203263-77922411-f0a4-4348-b6c6-b9854d8a2470.png">
<img width="498" alt="Screen Shot 2022-09-02 at 11 03 18 AM" src="https://user-images.githubusercontent.com/104043438/188203324-d49f5fa4-70be-46cc-a178-acb3d39144e7.png">
<img width="808" alt="Screen Shot 2022-09-02 at 11 03 30 AM" src="https://user-images.githubusercontent.com/104043438/188203357-75d55a39-3c7b-4307-b8bb-a04e9e1596d2.png">

- Balanced accuracy score was = 0.6314677834584286 or 0.63 
- Precision matrix:
  - True positives: 50
  - False positives: 5337
  - Sensitivity/recall of this model: 50 / (50 + 37) = 0.58
  - Precision of this model: 50 / (50 + 5337) = 0.0093

This chart represents the models precision matrix, I wrote it out in a simpler format for my readers to understand what I'm referring to moving forward.

<img width="694" alt="Screen Shot 2022-09-02 at 11 20 55 AM" src="https://user-images.githubusercontent.com/104043438/188205892-75e36e5e-069f-483a-a328-7120eccc6bd9.png">

### SMOTE Oversampling
<img width="461" alt="Screen Shot 2022-09-02 at 11 38 17 PM" src="https://user-images.githubusercontent.com/104043438/188257471-fe86dc25-d0ea-43db-b079-680369a59b6b.png">
<img width="365" alt="Screen Shot 2022-09-02 at 11 38 29 PM" src="https://user-images.githubusercontent.com/104043438/188257475-23cee35b-6344-437e-9049-095af7ed45c7.png">
<img width="805" alt="Screen Shot 2022-09-02 at 11 38 41 PM" src="https://user-images.githubusercontent.com/104043438/188257481-eb3283c8-e213-4125-b51b-262ab0a74d4c.png">

- Balanced accuracy score was = .6268316069795457 or 0.63
- Precision matrix:
  - True positives: 53
  - False positives: 6086
  - Sensitivity/recall of this model: 53 / (53 + 34) = 0.61
  - Precision of this model: 53 / (53 + 6086) = 0.0086
  
### Undersampling
<img width="819" alt="Screen Shot 2022-09-03 at 3 06 31 PM" src="https://user-images.githubusercontent.com/104043438/188287672-ac17ad7f-b2ef-47ab-bc81-823e4d4b1feb.png">

- Balanced accuracy score was = 0.5126747001543042 or 0.51
- Precision matrix: 
  - True positives: 50
  - False positives: 9404
  - Sensitivity/recall of this model: 50 / (50 + 37) = 0.58
  - Precision of this model: 50 / (50 + 9404) = 0.0053

### Combination (Over & Under) Sampling
<img width="445" alt="Screen Shot 2022-09-03 at 12 02 19 AM" src="https://user-images.githubusercontent.com/104043438/188258102-671df80b-84c8-4125-8042-05892476b499.png">
<img width="398" alt="Screen Shot 2022-09-03 at 12 02 30 AM" src="https://user-images.githubusercontent.com/104043438/188258105-39c35491-f381-4c53-b379-065eb9641fdc.png">
<img width="801" alt="Screen Shot 2022-09-03 at 12 02 39 AM" src="https://user-images.githubusercontent.com/104043438/188258110-7ff36aab-9e41-4a2b-a45e-2a682a87da62.png">
- Balanced accuracy score was = 0.6413505042081133 or 0.64
- Precision matrix: 
  - True positives: 61
  - False positives: 7163
  - Sensitivity/recall of this model: 61 / (61 + 26) = 0.7
  - Precision of this model: 61 / (61 + 7163) = 0.0084
  
#### Ensemble Learners: Balanced Random Forest Classifier
<img width="596" alt="Screen Shot 2022-09-03 at 3 10 46 PM" src="https://user-images.githubusercontent.com/104043438/188287750-8069d42b-d38d-47aa-a3ad-dbe70e55980f.png">
<img width="851" alt="Screen Shot 2022-09-03 at 3 10 58 PM" src="https://user-images.githubusercontent.com/104043438/188287755-9f1d1808-695a-476f-8275-6131043a70b7.png">
- Balanced accuracy score was = 0.795829959187949 or 0.80
- Precision matrix: 
  - True positives: 62
  - False positives: 2071
  - Sensitivity/recall of this model: 62 / (62 + 25) = 0.71
  - Precision of this model: 61 / (61 + 2071) = 0.029

#### Easy Ensemble AdaBoost Classifier 
<img width="525" alt="Screen Shot 2022-09-03 at 3 12 43 PM" src="https://user-images.githubusercontent.com/104043438/188287801-60f85ed7-7f0c-4de7-abd8-8b94da26c834.png">
<img width="808" alt="Screen Shot 2022-09-03 at 3 12 52 PM" src="https://user-images.githubusercontent.com/104043438/188287806-7ebc34af-66b5-4bde-9202-5f4d5d9923c5.png">
- Balanced accuracy score was = 0.9263912558266958 or 0.93
- Precision matrix: 
  - True positives: 79
  - False positives: 946
  - Sensitivity/recall of this model: 79 / (79 + 8) = 0.91
  - Precision of this model: 79 / (79 + 946) = 0.077
abac
## Results: 
