import glob  
import numpy as np

from utils.ploting import report, plot_precision_recall_curve, plot_roc_curve

PREDICT = None
TARGET = None 

for outs in glob.glob("saved/Exp3/*/outs*"):
     fold = outs.split("/")[-2].split("fold")[-1]
     if int(fold) not in range(0,20):
          print(fold)
          continue
     predict = np.load(outs).astype(int)
     target = np.load(outs.replace('outs','trgs')).astype(int)
     if PREDICT is None:
          PREDICT = predict
          TARGET = target
     else:
          PREDICT = np.concatenate((PREDICT, predict), axis=0)
          TARGET = np.concatenate((TARGET, target), axis=0)
          
print("Total Sample: ", len(TARGET))
          
report(TARGET, PREDICT, [0,1,2,3,4], "saved/confusion_matrix.png")