##################################################################################################
#### Anomaly detection in a smart building setup                                              ####
##################################################################################################

###### Data preprocessing #######
from pandas import read_csv
import numpy as np

#load CSV
building_data = read_csv('smartbuilding_anom.csv')
df = building_data

principal_mapping = {'User1':1,'User2':2,'User3':3,'System':4}
df['principal'] = df['principal'].map(principal_mapping)

device_mapping = {'Phone':1, 'Smart Camera':2,'Climate Sensmitter':3,'Smart Lighting':4}
df['device'] = df['device'].map(device_mapping)

activity_mapping = {'User Feedback':1,'Building Automation':2} 
df['activity'] = df['activity'].map(activity_mapping)

message_type_mapping = {'Command':1,'Data':2} 
df['message'] = df['message'].map(message_type_mapping)

attribute_type_mapping = {'State':1,'Count':2,'Direction':3,'Presence':4,'Temperature':5,'Brightness':6} 
df['attribute'] = df['attribute'].map(attribute_type_mapping)

criteria = [df['value']=='left', df['value']=='right', df['value']=='in', df['value']=='out', df['value']=='silent', df['value']=='convo/meeting', df['value']=='gathering']
values = [1, 2, 3, 4, 5, 6, 7]
df['value'] = np.select(criteria, values, df['value'])

df

anomaly_set = df[(df['attribute'] == 1)]
df = anomaly_set
display(df)

###### Training and metrics #######

# Import the classifiers
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, roc_curve, roc_auc_score, precision_recall_curve, confusion_matrix, accuracy_score 
from sklearn.model_selection import train_test_split
import seaborn as sns

confusion = True
verbose = True
return_metrics = False

def calc_f1(p_and_r):
    p, r = p_and_r 
    return (2*p*r)/(p+r)   

# Instantiate the classfiers and make a list
classifiers = [LogisticRegression(random_state=1234), 
               GaussianNB(), 
               KNeighborsClassifier(), 
               DecisionTreeClassifier(random_state=1234),
               RandomForestClassifier(random_state=1234)]

# Define a result table as a DataFrame
result_table = pd.DataFrame(columns=['classifiers','fpr','tpr','auc','acc','f1'])

# Split the data in training and testing subsets
x = df[['principal', 'device', 'activity', 'message', 'attribute', 'value', 'time_hdn']].values
y = df.values[:,7] # anomaly 
y=y.astype('float') 

X_train,X_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=0)

# Train the models and record the results
for cls in classifiers:
    model = cls.fit(X_train, y_train)
    yproba = model.predict_proba(X_test)[::,1]
    fpr, tpr, _ = roc_curve(y_test, yproba)

    # Calculate precision-recall curve 
    precision, recall, threshold = precision_recall_curve(y_test, yproba)  
    # Find the threshold value that gives the best F1 Score 
    best_f1_index =np.argmax([calc_f1(p_r) for p_r in zip(precision, recall)])  
    best_threshold, best_precision, best_recall = threshold[best_f1_index], precision[best_f1_index], recall[best_f1_index]
    # Calulcate predictions based on the threshold value
    y_test_pred = np.where(yproba > best_threshold, 1, 0) 
    
    # Calculate all metrics
    f1 = f1_score(y_test, y_test_pred, pos_label = 1, average = 'binary')  
    auc = roc_auc_score(y_test, yproba)
    acc = accuracy_score(y_test, y_test_pred)

    result_table = result_table.append({'classifiers':cls.__class__.__name__,
                                        'fpr':fpr, 
                                        'tpr':tpr, 
                                        'auc':auc,
                                        'acc':acc,
                                        'f1':f1}, ignore_index=True)

    if confusion:
        # Calculate and display the confusion matrix
        cm = confusion_matrix(y_test, y_test_pred)

        plt.title('Confusion Matrix')
        sns.set(font_scale=1.0) #for label size
        sns.heatmap(cm, annot = True, fmt = 'd', xticklabels = ['No Anomaly', 'Anomaly'], yticklabels = ['No Anomaly', 'Anomaly'], annot_kws={"size": 14}, cmap = 'Blues')# font size

        plt.xlabel('Truth')
        plt.ylabel('Prediction')

        #
        plt.savefig(cls.__class__.__name__+"_confusion_matrix" +".png")
        
        plt.show()
        
    if verbose:
        print('F1: {:.3f} | Pr: {:.3f} | Re: {:.3f} | AUC: {:.3f} | Accuracy: {:.3f} \n'.format(f1, best_precision, best_recall, auc, acc))
    
    if return_metrics:
        print(np.array([f1, best_precision, best_recall, auc, acc]))
    

# Set name of the classifiers as index labels
result_table.set_index('classifiers', inplace=True)

display(result_table) 

fig = plt.figure(figsize=(8,6))

for i in result_table.index:
    plt.plot(result_table.loc[i]['fpr'], 
             result_table.loc[i]['tpr'], 
             label="{}, AUC={:.3f}".format(i, result_table.loc[i]['auc']))
    
plt.plot([0,1], [0,1], color='orange', linestyle='--')

plt.xticks(np.arange(0.0, 1.1, step=0.1))
plt.xlabel("Flase Positive Rate", fontsize=15)

plt.yticks(np.arange(0.0, 1.1, step=0.1))
plt.ylabel("True Positive Rate", fontsize=15)

plt.title('ROC Curve Analysis', fontweight='bold', fontsize=15)
plt.legend(prop={'size':13}, loc='lower right')

plt.savefig("roc_curve.png")
plt.show()

