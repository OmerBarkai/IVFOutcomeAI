""""
IVF Pregnancy Prediction with Microbiome and Inflammatory Analysis

This script analyzes microbiome and cytokine data collected from individuals during
3 timepoints of IVF treatment to predict pregnancy outcomes using machine learning.

"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import LeaveOneOut
from imblearn.over_sampling import SMOTE
from AniML_utils_Publishing import *
import seaborn as sns
import shap
import matplotlib
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
matplotlib.use('Qt5Agg')
plt.ion()

# Define your time points and feature sets
time_points = ['1A', '2A', '3A']
feature_sets = {'Cytokines': slice(1, 21),  # Assuming Cytokines are in columns 1 to 20
                'Bacteria': slice(25, 275),  # Assuming Bacteria are in columns from 25 onward
                'Cytokines and Bacteria': None
                }  # Cytokines and Bacteria Bacteria and Cytokines will be concatenated

# Initialize a dictionary to store the feature importances for each combination
importance_map = {}
shap_map = {}
f1_map = {}

# Load the CSV file into a DataFrame
data_path=''
Microbiome = pd.read_csv(r'IVFOutcomeAI_data/miio_all.csv')
ShannIdx = pd.read_csv(r'IVFOutcomeAI_data/MIIO_shannonIndex_vaginalMicrobime.csv', sep='\s+', usecols=['Shannon_index', 'sample'])
IVFcyType = pd.read_csv(r'IVFOutcomeAI_data/MIIO_IVFcycleType.csv')

df=pd.merge(Microbiome,ShannIdx, on='sample')
df_IVFcyType=pd.merge(Microbiome,IVFcyType, on='sample')
class_thresh = 0.5
y_is = 'YesPreg'

# Loop over each feature set
for set_name, set_slice in feature_sets.items():
    # Loop over each time point
    for time_point in time_points:
        print(f'Time point {time_point} Features: {set_name} ')
        print(len)
        # Filter rows where 'sample' contains the current time point
        df_time = df[df['sample'].str.contains(time_point)]
        Shan = df[df['sample'].str.contains(time_point)]['Shannon_index']
        df_IVFcyType_time= df_IVFcyType[df_IVFcyType['sample'].str.contains(time_point)][['sample','Cycle.IVF.Cryothaw']]

        # Select the appropriate features
        if set_slice is not None:
            X = pd.concat([df_time.iloc[:, set_slice],
                           # Shan, #To add Shannon Index
                           ], axis=1)
        else:  # Concatenate Cytokines and Bacteria Bacteria and Cytokines
            X = pd.concat([df_time.iloc[:, feature_sets['Cytokines']],
                           df_time.iloc[:, feature_sets['Bacteria']],
                           # Shan, #To add Shannon Index
                           ], axis=1)

        # Drop columns not needed (columns_out)
        columns_out = 'rodentium|NA|unassigned unassigned'
        X = X.filter(regex=f'^(?!.*({columns_out})).*$')


        # Drop columns that don't have at least 50% of their values > 0
        min_sample_thresh = 0.5
        X = X.loc[:, (X > 0).sum() >= int(len(X) * min_sample_thresh)]

        # Drop the columns where none of the values are greater than or equal to 0.01
        if set_name != 'Cytokines' and not X[feature_sets['Bacteria']].empty:
            X = X.drop(columns=[col for col in X if (X[col] < 0.01).all()])

        y = df['outcome'] == y_is
        y = df_time['outcome'] == y_is
        y = y.reset_index(drop=True)
        original_n=len(y)
        # print('Shuffling per-SMOTE')
        # from sklearn.utils import shuffle
        # y = shuffle(y, random_state=4)


        # Upsample the minority class using SMOTE
        smote = SMOTE(random_state=42)
        X_smote, y_smote = smote.fit_resample(X, y)
        print(f'Before: {len(y)}, Smote: {len(y_smote)}')

        # Initialize Leave-One-Out Cross-Validation
        loo = LeaveOneOut()
        all_importances = pd.DataFrame()

        # Initialize variables to keep track of predictions and actual values
        y_true = []
        y_pred = []
        # Perform LOO-CV
        for train_index, test_index in loo.split(X_smote):
            X_train, X_test = X_smote.iloc[train_index], X_smote.iloc[test_index]
            y_train, y_test = y_smote.iloc[train_index], y_smote.iloc[test_index]

            # # learning curve
            # n_subj=27
            # X_train, X_test, y_train, y_test = X_train[0:n_subj], X_test[0:n_subj], y_train[0:n_subj], y_test[0:n_subj]

            # Scale the features
            scaler = StandardScaler()
            X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
            X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)

            # Train the model using SVC
            model = SVC(kernel='linear', random_state=42)
            model.fit(X_train_scaled, y_train)
            all_importances = pd.concat([all_importances, pd.Series(model.coef_[0], index=X_train.columns)], axis=1)


            y_test_pred = model.predict(X_test_scaled)
            y_true.append(y_test.values[0])
            y_pred.append(y_test_pred[0])

        # Calculate mean feature importance
        feature_importances = all_importances.mean(axis=1)
        # Save the feature importances for this time point and feature set
        importance_map[(time_point, set_name)] = feature_importances


        n_smotes = len(X_smote) - len(X)
        y_true_original = y_true[0:-n_smotes]
        y_pred_original = y_pred[0:-n_smotes]
        accuracy = accuracy_score(y_true_original, y_pred_original)
        f1 = f1_score(y_true_original, y_pred_original)
        print(f'Original LOO-CV | Accuracy: {accuracy:.2f} | F1: {f1:.2f}')

        # Save the F1 score for this time point and feature set
        f1_map[(time_point, set_name)] = f1

        plt.figure(figsize=(3, 3))
        colors = df_IVFcyType_time['Cycle.IVF.Cryothaw'].map({'IVF': 'blue', 'Cryothaw': 'red'})
        plt.scatter(y_true_original + np.random.uniform(-0.07, 0.07, size=np.size(y_true_original)),
                 y_pred_original + np.random.uniform(-0.1, 0.1, size=np.size(y_pred_original)),
                color = colors, # Color by IVF status
                alpha = 0.5, s=6)  # Make the points slightly transparent
        plt.xlabel(f'Ground truth ({y.name})');
        plt.xticks([0, 1], ['False', 'True'])
        plt.ylabel('Prediction');
        plt.yticks([0, 1], ['False', 'True'])
        plt.title(f'Originial Subjects LOO-CV \n {set_name}_{time_point}  F1: {f1:.2f} | Accuracy: {accuracy:.2f}')
        # Add a legend to indicate color mapping
        handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=6, label='IVF'),
                   plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=6, label='Non-IVF')]
        plt.legend(handles=handles, title='Cycle IVF')
        plt.tight_layout()
        #
        ConfusionMatrixDisplay.from_predictions(y_true_original, y_pred_original,
                                                display_labels=model.classes_.astype(int), normalize='true',
                                                cmap='Purples')
        plt.title(f'Originial Subjects LOO-CV \n {set_name}_{time_point}  F1: {f1:.2f} | Acc: {accuracy:.2f}', fontsize=10)
        for im in plt.gca().get_images():  # set clim manually within the image
            im.set_clim(vmin=0, vmax=1)
            im.figure.set_size_inches(3, 3)
        font = {'family': 'Arial', 'weight': 'normal', 'size': 8}
        plt.rc('font', **font)
        plt.tight_layout()

        # Calculate SHAP values
        explainer = shap.LinearExplainer(model, X_train_scaled[0:original_n])
        shap_values = explainer.shap_values(X_train_scaled[0:original_n])
        plt.figure()
        plt.title(f'{set_name}_{time_point}  F1: {f1:.2f} | Accuracy: {accuracy:.2f}')
        shap.summary_plot(shap_values, X_train_scaled[0:original_n])

        Explainer = shap.Explainer(model, X_train_scaled[0:original_n])
        shap_values = Explainer(X_train_scaled[0:original_n])

        shap_map[(time_point, set_name)] = pd.Series(np.mean(np.abs(shap_values.values),axis=0), index=X_train_scaled.columns)

        N_display = 10
        plt.figure()
        plt.subplot(1, 2, 1)
        plt.title(f'{set_name}_{time_point}  F1: {f1:.2f} | Accuracy: {accuracy:.2f}')
        shap.summary_plot(shap_values, X_train_scaled[0:original_n], max_display=N_display)
        plt.gca().tick_params(labelsize=10)
        plt.tight_layout()
        plt.subplot(1, 2, 2)
        # shap.plots.bar(shap_values, max_display=10)
        # plt.tight_layout()
        sorted_idx = np.argsort(np.abs(shap_values.values).mean(0))[::-1]
        top_N_idx = sorted_idx[:N_display]  # Select indices of the top 10 features

        # Filter SHAP values for only the top 10 features
        shap_values_topN = shap.Explanation(
            values=shap_values.values[:, top_N_idx],
            base_values=shap_values.base_values,
            data=shap_values.data[:, top_N_idx],
            feature_names=[shap_values.feature_names[i] for i in top_N_idx]
        )
        # plt.figure(figsize=(2.0, 1.5)) #(w,h)
        plt.xticks(rotation = 45, ha='right')
        colors = plt.cm.Blues(np.linspace(0.4, 1, N_display))  # Adjust the range for desired shade
        for i, (category, value) in enumerate(zip(shap_values_topN.feature_names[::-1], (np.mean(np.abs(shap_values_topN.values), axis=0)/np.max(np.mean(np.abs(shap_values_topN.values), axis=0)))[::-1])): # Plot horizontal lines with varying shades of blue
            if category in df_time.iloc[:, feature_sets['Bacteria']].columns:
                plt.hlines(category, xmin=0, xmax=value, color=plt.cm.Purples(np.linspace(0.2, 0.8, N_display))[i], linewidth=10)
            if category in df_time.iloc[:, feature_sets['Cytokines']].columns:
                plt.hlines(category, xmin=0, xmax=value, color=plt.cm.Oranges(np.linspace(0.2, 0.8, N_display))[i], linewidth=10)
        publish_figure()
        plt.gca().tick_params(labelsize=10)
        plt.xlabel('Feature Importance')
        publish_figure()
        plt.tight_layout()


#%% Plot Importance map by Coeff_[0]
# Create a DataFrame with all the importance values for heatmap
all_features = pd.concat([importance_map[key] for key in importance_map], axis=1).fillna(np.nan)
# Create multi-index columns (time point and feature set)
all_features.columns = pd.MultiIndex.from_tuples(importance_map.keys(), names=["Time Point", "Feature Set"])

normalized_features = np.abs(all_features)/np.abs(all_features).max(axis=0)

#Cannot cluster with NaNs, replace with zero
normalized_features[np.isnan(normalized_features)]=0
annotation_matrix = np.where(normalized_features != 0, np.round(normalized_features,2), "") # To not annotate 0's in the heatmap, replace with ""

plt.figure(figsize=(12, 8))
sns.heatmap(normalized_features, cmap='Blues', annot=annotation_matrix, fmt='')
plt.title('Feature Importance Heatmap')
plt.tight_layout()

#Cluster the heatmap
# Create a clustermap to order features by similarity
sns.clustermap(normalized_features, cmap='Blues', annot=annotation_matrix, fmt='', figsize=(12, 8), yticklabels=True)
plt.title('Feature Importance Clustered')
plt.tight_layout()
#%% Plot Importance map by SHAP
# Create a DataFrame with all the importance values for heatmap
shap_features = pd.concat([shap_map[key] for key in shap_map], axis=1).fillna(np.nan)
# Create multi-index columns (time point and feature set)
shap_features.columns = pd.MultiIndex.from_tuples(shap_map.keys(), names=["Time Point", "Feature Set"])

normalized_shap_features = np.abs(shap_features)/np.abs(shap_features).max(axis=0)

#Cannot cluster with NaNs, replace with zero
normalized_shap_features[np.isnan(normalized_shap_features)]=0
shap_annotation_matrix = np.where(normalized_shap_features != 0, np.round(normalized_shap_features,2), "") # To not annotate 0's in the heatmap, replace with ""
plt.figure(figsize=(7, 8))
sns.heatmap(normalized_shap_features, cmap='Blues', annot=shap_annotation_matrix, fmt='')
plt.title('SHAP Importance Heatmap')
plt.tight_layout()

#Cluster the heatmap
# Create a clustermap to order features by similarity
sns.clustermap(normalized_shap_features, cmap='Blues', annot=shap_annotation_matrix, fmt='', figsize=(8, 8), yticklabels=True)
plt.title('SHAP Importance Clustered')
plt.tight_layout()


#%% Reshape the f1_map into a 3x3 DataFrame
plt.close("all")
f1_df = pd.DataFrame(f1_map, index=['F1 Score']).T.unstack().reset_index()
f1_df.set_index('index', inplace=True)
# Plot the heatmap
plt.figure(figsize=(4, 3))
sns.heatmap(f1_df['F1 Score'].T, annot=True, cmap='Purples', cbar=True, linecolor='w', linewidth=1)
plt.title('F1 Scores')
plt.tight_layout()




# Create a subplot grid with multiple rows and one column (vertical layout)
n_rows = len(importance_map)
fig, axes = plt.subplots(1,n_rows,figsize=(16, 2 * n_rows), sharex=False)

# Loop over the time point/feature set combinations and plot each one
for i, ((time_point, feature_set), importances) in enumerate(importance_map.items()):
    # Sort the feature importances in descending order
    sorted_importances = np.abs(importances).sort_values(ascending=False)

    # Convert the feature names to a 2D array to match the shape of the data
    feature_names = np.array(sorted_importances.index).reshape(-1, 1)

    # Create a heatmap with feature names as annotation on each subplot
    sns.heatmap(sorted_importances.values[:, np.newaxis], ax=axes[i], cmap='coolwarm',
                annot=feature_names, fmt='', cbar=False,
                yticklabels=False, xticklabels=False, annot_kws={"size": 6},
                linewidth=1)

    axes[i].set_title(f'{time_point.split("A")[0]} - {feature_set}', pad=10) # Set the title for each subplot
    axes[i].set_xticks([]) # Remove x-ticks

# Add a smaller colorbar to the right side of the last heatmap
cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])  # Adjusted size for a smaller colorbar
plt.colorbar(axes[-1].collections[0], cax=cbar_ax)

# Adjust the layout to fit everything properly
plt.tight_layout(rect=[0, 0, 0.9, 1])  # leave space for the colorbar on the right
plt.show()


if 0:
    #%%
    def figures_to_files():
        save_to_folder=r'J:\My Drive\MIIO_AI\noShannon/'
        save_figs('svg',save_to_folder, transparent=True)
        save_figs('png', save_to_folder)


    figures_to_files()
