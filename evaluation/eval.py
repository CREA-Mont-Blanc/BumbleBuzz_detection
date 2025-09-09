import pandas as pd
import os
from sklearn.metrics import f1_score, recall_score, precision_score, roc_auc_score
import numpy as np
import argparse


COLUMNS_VARS = "Group_buzz"

parser = argparse.ArgumentParser(
    description="Evaluate detection results (output from process.py in eavt)"
)
parser.add_argument(
    "--dir",
    default="example/metadata",
    type=str,
    help="Path to bumblebee dataset folder",
)
parser.add_argument(
    "--pred", default="detection", type=str, help="Path to predictions folder"
)
parser.add_argument("--name", default="", type=str, help="name of site")
args = parser.parse_args()

# Load the predictions (output from process.py)
predictions = os.path.join(args.dir, args.pred, f"indices_{args.name}.csv")

Df = pd.read_csv(predictions)

# Load all ground truth csv files
gt_files = []
for root, dirs, files in os.walk(os.path.join(args.dir, args.name)):
    for file in files:
        if os.path.splitext(file)[1] == ".txt":
            gt_files.append(os.path.join(root, file))

# Check if any ground truth files were found
if not gt_files:
    print("No ground truth files found.")
    exit()

# Load all ground truth dataframes
dfs = []
df_preds_gt = []
for gt_file in gt_files:
    df = pd.read_csv(gt_file, sep="\t", header=None)
    # add a column with the name of the file
    df["file"] = os.path.basename(gt_file)

    ## find the predictions corresponding to the annotations
    ## filter the predictions Dataframe to only include the ones that are in the ground truth file
    gtbasename = os.path.basename(gt_file)
    gtbasename = os.path.splitext(gtbasename)[0]
    ## remove '_labels.txt' and replace with '.wav'
    gtbasename = gtbasename.replace("_labels", "")
    gtbasename = gtbasename + ".wav"
    # get the predictions that correspond to the gtbasename
    # filter predictions for this wav and work on a copy to avoid SettingWithCopyWarning
    df_pred = Df[Df.loc[:, "name"] == gtbasename].copy()
    if df_pred.empty:
        # no predictions for this wav, skip and continue
        continue
    df_pred.loc[:, "buzzlabel"] = 0

    # for each row in the gt file, check if the start and end times are in the predictions
    df["pred"] = 0.0
    df["flacfile"] = "null"
    for index, row in df.iterrows():
        start = row[0]
        end = row[1]
        pred = df_pred[df_pred.loc[:, "start"] >= start]
        pred2 = df_pred[
            (df_pred.loc[:, "start"] >= start) & (df_pred.loc[:, "start"] + 5 <= end)
        ]
        df_pred.loc[
            (df_pred.loc[:, "start"] >= start) & (df_pred.loc[:, "start"] + 5 <= end),
            "buzzlabel",
        ] = 1
        if pred.empty:
            # no matching prediction chunk for this GT row -> leave pred as 0 and flacfile as 'null'
            df.at[index, "pred"] = 0.0
            df.at[index, "flacfile"] = "null"
            continue

        # we just take the first flacfile that matches the start time
        df.at[index, "flacfile"] = pred.iloc[0]["flacfile"]
        if not pred2.empty:
            # if there are multiple predictions (event longer than 5 seconds), take the mean
            buzz = np.mean(pred2.loc[:, COLUMNS_VARS])
            df.at[index, "pred"] = buzz
        else:
            # if the ground truth event is shorter than 5 seconds, take the first chunk of 5 seconds
            buzz = pred.iloc[0][COLUMNS_VARS]  # new name of buzz
            df.at[index, "pred"] = buzz

    df_preds_gt.append(df_pred)
    dfs.append(df)
# Concatenate all dataframes into a single dataframe
df_gt = pd.concat(dfs, ignore_index=True)
df_final_preds_gt = pd.concat(df_preds_gt, ignore_index=True)

## rename the columns
df_gt.columns = ["start", "end", "event", "file", "pred", "flacfile"]


# Calculate the AUC
aucscore = roc_auc_score(
    df_final_preds_gt["buzzlabel"], df_final_preds_gt[COLUMNS_VARS]
)
print(f"AUC: {aucscore:.4f} for site {args.name}")

# compute the roc curve and find the threshold that gives the best f1 score
from sklearn.metrics import roc_curve

fpr, tpr, thresholds = roc_curve(
    df_final_preds_gt["buzzlabel"], df_final_preds_gt[COLUMNS_VARS]
)
## plot the curve
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="whitegrid")
plt.figure(figsize=(10, 6))
plt.plot(fpr, tpr, color="blue", label="ROC curve (area = %0.2f)" % aucscore)
plt.plot([0, 1], [0, 1], color="red", linestyle="--")
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Receiver Operating Characteristic")
plt.legend(loc="lower right")
plt.savefig(os.path.join(args.dir, args.pred, f"{args.name}_roc_curve.png"))
plt.close()

f1_scores = []
for threshold in thresholds:
    y_pred = (df_final_preds_gt[COLUMNS_VARS] >= threshold).astype(int)
    f1 = f1_score(df_final_preds_gt["buzzlabel"], y_pred)
    f1_scores.append(f1)
best_threshold = thresholds[np.argmax(f1_scores)]
print(f"Best threshold: {best_threshold:.4f} for site {args.name}")
# Calculate the precision and recall
precision = precision_score(
    df_final_preds_gt["buzzlabel"],
    (df_final_preds_gt[COLUMNS_VARS] >= best_threshold).astype(int),
)
recall = recall_score(
    df_final_preds_gt["buzzlabel"],
    (df_final_preds_gt[COLUMNS_VARS] >= best_threshold).astype(int),
)
print(f"Precision: {precision:.4f} for site {args.name}")
print(f"Recall: {recall:.4f} for site {args.name}")
# Calculate the f1 score
f1 = f1_score(
    df_final_preds_gt["buzzlabel"],
    (df_final_preds_gt[COLUMNS_VARS] >= best_threshold).astype(int),
)
print(f"F1 score: {f1:.4f} for site {args.name}")

## Calculate the scores for TPR of 0.9 and 0.95
tpr_values = [0.9, 0.95]
results = []

for tpr_target in tpr_values:
    fpr_target = fpr[np.argmax(tpr >= tpr_target)]
    print(f"FPR at TPR {tpr_target}: {fpr_target:.4f} for site {args.name}")
    # Corresponding threshold
    threshold_target = thresholds[np.argmax(tpr >= tpr_target)]
    print(f"Threshold at TPR {tpr_target}: {threshold_target:.4f} for site {args.name}")
    # Calculate precision and recall at TPR target
    precision_target = precision_score(
        df_final_preds_gt["buzzlabel"],
        (df_final_preds_gt[COLUMNS_VARS] >= threshold_target).astype(int),
    )
    recall_target = recall_score(
        df_final_preds_gt["buzzlabel"],
        (df_final_preds_gt[COLUMNS_VARS] >= threshold_target).astype(int),
    )
    print(f"Precision at TPR {tpr_target}: {precision_target:.4f} for site {args.name}")
    print(f"Recall at TPR {tpr_target}: {recall_target:.4f} for site {args.name}")
    # Calculate F1 score at TPR target
    f1_target = f1_score(
        df_final_preds_gt["buzzlabel"],
        (df_final_preds_gt[COLUMNS_VARS] >= threshold_target).astype(int),
    )
    print(f"F1 score at TPR {tpr_target}: {f1_target:.4f} for site {args.name}")

    # create a new column in the predictions dataframe by thresholding the Group_buzz column accordingly, name this column pred_{thr}
    df_final_preds_gt[f"pred_{tpr_target}"] = (
        df_final_preds_gt[COLUMNS_VARS] >= threshold_target
    ).astype(int)

    results.append(
        {
            "TPR": tpr_target,
            "FPR": fpr_target,
            "Threshold": threshold_target,
            "Precision": precision_target,
            "Recall": recall_target,
            "F1 score": f1_target,
        }
    )

# Add the results for TPR 0.9 and 0.95 to the dataframe
df_results = pd.DataFrame(
    {
        "AUC": [aucscore],
        "Best threshold": [best_threshold],
        "Precision": [precision],
        "Recall": [recall],
        "F1 score": [f1],
        "FPR at TPR 0.9": [results[0]["FPR"]],
        "Threshold at TPR 0.9": [results[0]["Threshold"]],
        "Precision at TPR 0.9": [results[0]["Precision"]],
        "Recall at TPR 0.9": [results[0]["Recall"]],
        "F1 score at TPR 0.9": [results[0]["F1 score"]],
        "FPR at TPR 0.95": [results[1]["FPR"]],
        "Threshold at TPR 0.95": [results[1]["Threshold"]],
        "Precision at TPR 0.95": [results[1]["Precision"]],
        "Recall at TPR 0.95": [results[1]["Recall"]],
        "F1 score at TPR 0.95": [results[1]["F1 score"]],
    }
)
df_results.to_csv(
    os.path.join(args.dir, args.pred, f"{args.name}_metrics.csv"), index=False
)
print(df_results)

df_gt.to_csv(os.path.join(args.dir, args.pred, f"{args.name}_gtonly.csv"), index=False)

## change column order to have buzz	domesticanimals	dB	buzzlabel	pred_0.9	pred_0.95
cols = df_final_preds_gt.columns.tolist()
cols = cols[-3:] + cols[:-4]
df_final_preds_gt = df_final_preds_gt[cols]


df_final_preds_gt.to_csv(
    os.path.join(args.dir, args.pred, f"{args.name}_gt_preds.csv"), index=False
)
