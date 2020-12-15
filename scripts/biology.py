import numpy as np
from sklearn.decomposition import TruncatedSVD
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from utils import load_data, load_test_sequences, print_summary, df_to_array, add_false_class_2, kmeans_xufive
from SenSR import train_nn, train_fair_nn

import warnings
warnings.filterwarnings('ignore')

# Load data
data_int, embeddings, X_train, X_test, y_train, y_test, middle, train_vocab, test_vocab = load_data('X_test_HCV_ternary_0.3_adjusted', 'y_test_HCV_ternary_0.3_adjusted')

# Count classes in original data
count = data_int.loc[data_int['result'] != 'MIDDLE']
count = count.loc[:,'result'].value_counts()
count = count.to_frame()
count = count.reset_index()
count.columns = ['CLASS', 'COUNT']
count

# Plot intinal count
plt.rcParams['figure.figsize'] = (10.0, 8.0)

ax = sns.barplot(x="CLASS", y="COUNT", data=count, palette="pastel")

plt.xticks(fontsize=16)
plt.yticks(fontsize=16)

plt.yticks(np.arange(0, 1700, 300))

plt.xlabel("CLASS", fontsize=16)
plt.ylabel("COUNT", fontsize=16)

x = np.arange(len(count["CLASS"]))
y = np.array(list(count["COUNT"]))
for a,b in zip(x,y):
    plt.text(a, b+0.05, '%.0f' % b, ha='center', va= 'bottom',fontsize=16)

plt.show()

# Load test sequences and their embeddings
test_df, test_sequences_embed = load_test_sequences(embeddings)

# Convert Dataframe into Numpy array
test_sequences_embed, embeddings, X_train, X_test, y_train, y_test, middle = df_to_array(test_sequences_embed, embeddings, X_train, X_test, y_train, y_test, middle)

# KMeans
result, cores = kmeans_xufive(test_sequences_embed, 2)
result = pd.DataFrame(result)
result[0] = result[0].apply(lambda x : 'Class 2' if x == 1 else 'Class 1')
result.columns = ['result']
test_df = test_df.drop(columns = ['result'])
test_df = pd.concat([test_df, result], axis = 1)

# Add false class 2
test_df = add_false_class_2(test_df, number = 100)

# Count classes after adding false
count = test_df.loc[:,'result'].value_counts()
count = count.to_frame()
count = count.reset_index()
count.columns = ['CLASS', 'COUNT']
count

# Plot count
plt.rcParams['figure.figsize'] = (10.0, 8.0)

ax = sns.barplot(x="CLASS", y="COUNT", data=count, palette="pastel")

plt.xticks(fontsize=16)
plt.yticks(fontsize=16)

plt.yticks(np.arange(0, 350, 100))

plt.xlabel("CLASS", fontsize=16)
plt.ylabel("COUNT", fontsize=16)

x = np.arange(len(count["CLASS"]))
y = np.array(list(count["COUNT"]))
for a,b in zip(x,y):
    plt.text(a, b+0.05, '%.0f' % b, ha='center', va= 'bottom',fontsize=16)

plt.show()

# Baseline
baseline_weights, _, baseline_test_logits = train_nn(X_train, y_train, X_test=X_test, y_test=y_test, epoch=2000, batch_size=1000)
baseline_accuracy = (baseline_test_logits.argmax(axis=1)==y_test.argmax(axis=1)).mean()
_, baseline_names_logits, _ = train_nn(test_sequences_embed, y_train=None, weights=baseline_weights, epoch=0)
test_df['baseline_logits'] = baseline_names_logits[:,1] - baseline_names_logits[:,0]
print_summary(test_df, 'baseline', baseline_accuracy)

# SenSR_0 expert
expert_sens_directions = np.copy(test_sequences_embed)
sensr0_expert_weights, _, sensr0_expert_test_logits = train_fair_nn(X_train, y_train, expert_sens_directions, X_test = X_test, y_test=y_test)
sensr0_expert_accuracy = (sensr0_expert_test_logits.argmax(axis=1)==y_test.argmax(axis=1)).mean()
_, sensr0_expert_names_logits, _ = train_nn(test_sequences_embed, y_train=None, weights=sensr0_expert_weights, epoch=0)
test_df['sensr0_expert_logits'] = sensr0_expert_names_logits[:,1] - sensr0_expert_names_logits[:,0]
print_summary(test_df, 'sensr0_expert', sensr0_expert_accuracy)

# Calculate quantile
class_1 = test_df.loc[test_df['result'] == 'Class 1']
class_2 = test_df.loc[test_df['result'] == 'Class 2']
Q1 = class_1['baseline_logits'].quantile(0.25)
Q2 = class_1['baseline_logits'].quantile(0.5)
Q3 = class_1['baseline_logits'].quantile(0.75)
IQR = Q3 - Q1
print('Baseline Class 1 quantile: \n''Q1 =' + str(Q1) + '\n' + 'Q2 = ' + str(Q2) + '\n' + 'Q3 = ' + str(Q3) + '\n' + 'IQR = ' + str(IQR))

Q1 = class_2['baseline_logits'].quantile(0.25)
Q2 = class_2['baseline_logits'].quantile(0.5)
Q3 = class_2['baseline_logits'].quantile(0.75)
IQR = Q3 - Q1
print('Baseline Class 2 quantile: \n''Q1 =' + str(Q1) + '\n' + 'Q2 = ' + str(Q2) + '\n' + 'Q3 = ' + str(Q3) + '\n' + 'IQR = ' + str(IQR))

class_1 = test_df.loc[test_df['result'] == 'Class 1']
class_2 = test_df.loc[test_df['result'] == 'Class 2']
Q1 = class_1['sensr0_expert_logits'].quantile(0.25)
Q2 = class_1['sensr0_expert_logits'].quantile(0.5)
Q3 = class_1['sensr0_expert_logits'].quantile(0.75)
IQR = Q3 - Q1
print('SenSR_0 expert Class 1 quantile: \n''Q1 =' + str(Q1) + '\n' + 'Q2 = ' + str(Q2) + '\n' + 'Q3 = ' + str(Q3) + '\n' + 'IQR = ' + str(IQR))

Q1 = class_2['sensr0_expert_logits'].quantile(0.25)
Q2 = class_2['sensr0_expert_logits'].quantile(0.5)
Q3 = class_2['sensr0_expert_logits'].quantile(0.75)
IQR = Q3 - Q1
print('SenSR_0 expert Class 2 quantile: \n''Q1 =' + str(Q1) + '\n' + 'Q2 = ' + str(Q2) + '\n' + 'Q3 = ' + str(Q3) + '\n' + 'IQR = ' + str(IQR))