training.ipynb:
This is the file that produces all diagrams, training data and results in the report. 
It implements 0R and tuning processes of MNB and MLP. All details are output in the notebook.

pred.py:
This is a .py file that output the results of kaggle competition by applying MLPClassifier(activation='logistic',max_iter=2,learning_rate='adaptive').
The hyperparameters are found after training.ipynb and use the optimal setttings.
The result file: pred_MLP.csv is output by this python script.
