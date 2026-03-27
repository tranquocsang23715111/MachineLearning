# Binary classification training script with detailed comments for students.
#
# This script:
# 1) Loads the binary dataset (healthy vs diseased) from CSV
# 2) Splits data into train/validation/test sets
# 3) Trains several models and selects the best via validation accuracy
# 4) Re-trains the best model on train+val and evaluates on test
# 5) Trains an SGD logistic classifier over epochs to plot train/val log loss
# 6) Saves confusion matrix plots and CSVs

import os
import csv
from typing import Tuple, List

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
	accuracy_score,
	precision_score,
	recall_score,
	f1_score,
	confusion_matrix,
	roc_auc_score,
	classification_report,
	log_loss,
)
import matplotlib.pyplot as plt
import seaborn as sns


# Input CSV and output plots directory (relative to this file)
DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "binary_classification.csv")
PLOTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "plots")


def load_binary_dataset(path: str) -> Tuple[np.ndarray, np.ndarray]:
	"""Load binary dataset and map string labels to integers.

	Returns:
		X: features as float array of shape (n_samples, n_features)
		y: int labels (0 for healthy, 1 for diseased)
	"""
	features: List[List[float]] = []
	labels: List[str] = []
	# Open CSV and read rows
	with open(path, newline="") as f:
		reader = csv.reader(f)
		headers = next(reader)  # skip header row
		for row in reader:
			features.append([float(x) for x in row[:-1]])  # all columns except last are features
			labels.append(row[-1])  # last column is label
	# Convert to numpy arrays
	X = np.array(features, dtype=float)
	y = np.array(labels)
	# Map text labels to integers for modeling/metrics
	label_to_int = {"healthy": 0, "diseased": 1}
	if not set(label_to_int.keys()).issuperset(set(np.unique(y))):
		raise ValueError("Unexpected labels in binary dataset. Expected 'healthy' and 'diseased'.")
	y_int = np.array([label_to_int[val] for val in y], dtype=int)
	return X, y_int


def split_data(X: np.ndarray, y: np.ndarray, seed: int = 7):
	"""Split into train, validation, test.

	We first split out test (20%), then split the remaining into train (60%) and val (20%).
	"""
	# Split out test set (20% of total)
	X_train_val, X_test, y_train_val, y_test = train_test_split(
		X, y, test_size=0.2, random_state=seed, stratify=y
	)
	# From the remaining 80%, take 25% as val => 0.8 * 0.25 = 20% of total
	X_train, X_val, y_train, y_val = train_test_split(
		X_train_val, y_train_val, test_size=0.25, random_state=seed, stratify=y_train_val
	)
	return X_train, X_val, X_test, y_train, y_val, y_test


def evaluate_on_test(name: str, model, X_test, y_test) -> None:
	"""Compute predictions and print/save common binary classification metrics and confusion matrix."""
	# Compute class probabilities if available (some models support predict_proba)
	y_prob = None
	if hasattr(model, "predict_proba"):
		y_prob = model.predict_proba(X_test)[:, 1]
	elif hasattr(model, "decision_function"):
		# Some models output decision scores; convert to [0,1] range for ROC AUC
		scores = model.decision_function(X_test)
		s_min, s_max = scores.min(), scores.max()
		y_prob = (scores - s_min) / (s_max - s_min + 1e-12)
	# Predicted classes
	y_pred = model.predict(X_test)

	# Scalar metrics
	acc = accuracy_score(y_test, y_pred)
	prec = precision_score(y_test, y_pred, zero_division=0)
	rec = recall_score(y_test, y_pred, zero_division=0)
	f1 = f1_score(y_test, y_pred, zero_division=0)
	# Confusion matrix with explicit label order: row=true, col=pred
	label_order = [0, 1]
	cm = confusion_matrix(y_test, y_pred, labels=label_order)
	# ROC AUC requires probability-like scores
	auc = roc_auc_score(y_test, y_prob) if y_prob is not None else float("nan")

	# Print results to console
	print(f"\n=== {name} Test Metrics ===")
	print(f"Accuracy: {acc:.4f}")
	print(f"Precision: {prec:.4f}")
	print(f"Recall: {rec:.4f}")
	print(f"F1-score: {f1:.4f}")
	print(f"ROC-AUC: {auc:.4f}")
	print("Confusion Matrix (array):")
	print(cm)
	print("Confusion Matrix (list):")
	print(cm.tolist())
	print("Classification Report:")
	print(classification_report(y_test, y_pred, target_names=["healthy", "diseased"]))

	# Save confusion matrix heatmap
	os.makedirs(PLOTS_DIR, exist_ok=True)
	plt.figure(figsize=(4, 4))
	sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
		xticklabels=["healthy", "diseased"], yticklabels=["healthy", "diseased"])
	plt.xlabel("Predicted")
	plt.ylabel("True")
	plt.title("Confusion Matrix - Binary")
	plt.tight_layout()
	plt.savefig(os.path.join(PLOTS_DIR, f"binary_confusion_matrix.png"))
	plt.close()

	# Also save confusion matrix as CSV with labeled rows/cols
	csv_path = os.path.join(PLOTS_DIR, "binary_confusion_matrix.csv")
	with open(csv_path, mode="w", newline="") as f:
		writer = csv.writer(f)
		writer.writerow(["", "pred_healthy", "pred_diseased"])
		writer.writerow(["true_healthy", cm[0, 0], cm[0, 1]])
		writer.writerow(["true_diseased", cm[1, 0], cm[1, 1]])


def sigmoid(x: np.ndarray) -> np.ndarray:
	"""Sigmoid function used to convert decision scores to probabilities."""
	return 1.0 / (1.0 + np.exp(-x))

def select_best_model_via_val(X_train, y_train, X_val, y_val):
	"""Train several models and return the one with highest validation accuracy (name, params)."""
	candidates = []

	# Logistic Regression (with standardization inside a pipeline)
	for C in [0.1, 1.0, 10.0]:
		for penalty in ["l2"]:
			pipe = Pipeline([
				("scaler", StandardScaler()),
				("clf", LogisticRegression(C=C, penalty=penalty, solver="lbfgs", max_iter=1000)),
			])
			pipe.fit(X_train, y_train)
			val_acc = pipe.score(X_val, y_val)
			candidates.append(("LogisticRegression", {"C": C, "penalty": penalty}, pipe, val_acc))

	# k-NN (with standardization)
	for k in [3, 5, 11]:
		pipe = Pipeline([
			("scaler", StandardScaler()),
			("clf", KNeighborsClassifier(n_neighbors=k)),
		])
		pipe.fit(X_train, y_train)
		val_acc = pipe.score(X_val, y_val)
		candidates.append(("KNeighborsClassifier", {"n_neighbors": k}, pipe, val_acc))

	# Decision Tree (no scaling needed)
	for depth in [None, 5, 10, 20]:
		model = DecisionTreeClassifier(max_depth=depth, random_state=0)
		model.fit(X_train, y_train)
		val_acc = model.score(X_val, y_val)
		candidates.append(("DecisionTreeClassifier", {"max_depth": depth}, model, val_acc))

	# Random Forest (no scaling needed)
	for n_estimators in [100, 300]:
		for max_depth in [None, 10, 20]:
			model = RandomForestClassifier(
				n_estimators=n_estimators,
				random_state=0,
				max_depth=max_depth,
				n_jobs=-1,
			)
			model.fit(X_train, y_train)
			val_acc = model.score(X_val, y_val)
			candidates.append((
				"RandomForestClassifier",
				{"n_estimators": n_estimators, "max_depth": max_depth},
				model,
				val_acc,
			))

	# Choose the highest validation accuracy
	best = max(candidates, key=lambda t: t[3])
	best_name, best_params, best_model, best_val = best
	print(f"Best validation model: {best_name} with params {best_params}, val_acc={best_val:.4f}")
	return best_name, best_params


def refit_best_and_evaluate(best_name: str, best_params: dict, X_train, y_train, X_val, y_val, X_test, y_test):
	"""Refit the selected model on train+val and evaluate it on the test set."""
	# Combine train and val to train the final model
	X_tr = np.vstack([X_train, X_val])
	y_tr = np.concatenate([y_train, y_val])

	# Recreate the best model with the chosen hyperparameters, train, then evaluate
	if best_name == "LogisticRegression":
		pipe = Pipeline([
			("scaler", StandardScaler()),
			("clf", LogisticRegression(C=best_params["C"], penalty=best_params["penalty"], solver="lbfgs", max_iter=1000)),
		])
		pipe.fit(X_tr, y_tr)
		evaluate_on_test("LogisticRegression", pipe, X_test, y_test)
	elif best_name == "KNeighborsClassifier":
		pipe = Pipeline([
			("scaler", StandardScaler()),
			("clf", KNeighborsClassifier(n_neighbors=best_params["n_neighbors"])),
		])
		pipe.fit(X_tr, y_tr)
		evaluate_on_test("KNeighborsClassifier", pipe, X_test, y_test)
	elif best_name == "DecisionTreeClassifier":
		model = DecisionTreeClassifier(max_depth=best_params["max_depth"], random_state=0)
		model.fit(X_tr, y_tr)
		evaluate_on_test("DecisionTreeClassifier", model, X_test, y_test)
	elif best_name == "RandomForestClassifier":
		model = RandomForestClassifier(
			n_estimators=best_params["n_estimators"],
			random_state=0,
			max_depth=best_params["max_depth"],
			n_jobs=-1,
		)
		model.fit(X_tr, y_tr)
		evaluate_on_test("RandomForestClassifier", model, X_test, y_test)
	else:
		raise ValueError("Unknown model name")


def main():
	"""Entry point that ties together loading data, training, selection, and evaluation."""
	print(f"Loading binary dataset from: {DATA_PATH}")
	X, y = load_binary_dataset(DATA_PATH)
	X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y, seed=7)
	# Select best classical model by validation accuracy and evaluate on test
	best_name, best_params = select_best_model_via_val(X_train, y_train, X_val, y_val)
	refit_best_and_evaluate(best_name, best_params, X_train, y_train, X_val, y_val, X_test, y_test)


if __name__ == "__main__":
	main()
