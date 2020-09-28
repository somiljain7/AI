## All Metrics in Classification

**Classification Problem**
Default threshold in `0.5` 
- Class Labels 
- Probabilities (AUC, ROC, PR Curve)

**Dataset (based on labels)**
- Nearly equal number of records for both labels => `Accuracy`
- Unequal records for labels => Recall, Precision, Fß (F1 Score) and `not Accuracy`

1. Confusion Matrix 
2x2 matrix 
- top (Actual Values)
- left Predicted Values

![Confusion Matrix](https://miro.medium.com/max/712/1*Z54JgbS4DUwWSknhDCvNTQ.png)

- False +ve => Type 1 Error
- False -ve => Type 2 Error

Aim - 
- Reducing Type1 and Type2 Error
- Accurate values T+ve and T-ve

**Parameters**
- Balanced Dataset

Accuracy = (TP + TN) / (TP + FP + FN + TN)

- Imbalanced Dataset

Recall(True +ve Rate or Sensitivity) = Out of all +ve values , how many +ve correctly predicted

Precision(+ve Pred value) = Out of all actual predicted result, how many actual +ve 

![Recall Precision](https://miro.medium.com/max/888/1*7J08ekAwupLBegeUI8muHA.png)


2. Fß Score - To reduce type 1 and type 2 errors

![Fß Score](https://miro.medium.com/max/1092/1*2YIU9iAzaE_g91vv2XU7Ew.png)

when ß = 1 => Harmonic mean
- When F+ve and F-ve is important then ß = 1
- Type 1 error ß > 1
- Type 2 error ß = 0.5 to 1 mostly 0.5
