

# Breast-Cancer-Classification-using-SVM


I have to classify the breast cancer data using Support vector machine .

The breast dataset download from sklearn in build dataset.

Here we are find benign and malignant cancer.(Two class classification)
Our dataset  divide into two part of data .That is data_fetaures and labels. The label data is represent 1 and 0
# Model Process:
     1.load dataset.
     2.pre processing 
     3.scalling the data
     4.build the SVM Model and Train the data.
     5.Validation process.

# First I try do basic SVM Model and get low accuracy values.

Accuracy = 0.631578947368421

confusion_mat =  [[  0  63]
                 [  0 108]]
# confusion_mat =               precision    recall  f1-score   support

                             0       0.00      0.00      0.00        63
                             1       0.63      1.00      0.77       108

                      accuracy                           0.63       171
                     macro avg       0.32      0.50      0.39       171
                  weighted avg       0.40      0.63      0.49       171


# Then Use Grid Search CV Algorithm and got a high accuracy values.
 this support vector classifier (SVC) is using the radial basis function kernel (RBF).
 
  Accuracy = 0.9649122807017544
  
  confusion_metric = [[ 58   5]
                     [  1 107]]
                  
# confusion_metric =               precision    recall  f1-score   support

                             0       0.98      0.92      0.95        63
                             1       0.96      0.99      0.97       108

                        accuracy                           0.96       171
                       macro avg       0.97      0.96      0.96       171
                    weighted avg       0.97      0.96      0.96       171


Finally got the good accuracy values.
Error function Graph
Thanks for Reading 



