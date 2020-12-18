import pandas as pd
from A1 import A1 
from A2 import A2
from B1 import B1
from B2 import B2


print("FINAL RESULTS GIVEN ALL TASKS:")
print("")
columns = ["Train Accuracy", "Test Accuracy","New Test Accuracy", "AUC", "AUC2"]
data = {'Task A1':[A1.train_acc_LR, A1.test_acc_LR, A1.test_acc2_LR, A1.auc_roc_LR, A1.auc_roc2_LR], 
        'Task A2':[A2.train_acc_LR, A2.test_acc_LR, A2.test_acc2_LR, A2.auc_roc_LR, A2.auc_roc2_LR],
        "Task B1":[B1.train_acc_LR, B1.test_acc_LR, B1.test_acc2_LR, B1.auc_roc_LR, B1.auc_roc2_LR],
        "Task B2":[B2.train_acc_LR, B2.test_acc_LR, B2.test_acc2_LR, B2.auc_roc_LR, B2.auc_roc2_LR]
        }

final_results = pd.DataFrame.from_dict(data, orient='index',columns=columns).T
print(final_results)


