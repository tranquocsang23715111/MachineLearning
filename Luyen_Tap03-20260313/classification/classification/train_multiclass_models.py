# Multiclass classification training script with detailed comments for students.
#
# This script:
# 1) Loads the multiclass dataset (beginner/intermediate/advanced) from CSV
# 2) Splits data into train/validation/test sets
# 3) Trains several models and selects the best via validation accuracy
# 4) Re-trains the best model on train+val and evaluates on test
# 5) Trains an SGD multinomial logistic classifier over epochs to plot train/val log loss
# 6) Saves confusion matrix plots and CSVs

import os
import csv
from typing import Tuple, List

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import (
	accuracy_score,
	precision_recall_fscore_support,
	confusion_matrix,
	classification_report,
	log_loss,
)
import matplotlib.pyplot as plt
import seaborn as sns


# Input CSV and output plots directory (relative to this file)
DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "multiclass_classification.csv")
PLOTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "plots")


def load_multiclass_dataset(path: str) -> Tuple[np.ndarray, np.ndarray, List[str]]:
	"""Load multiclass dataset and map string labels to integers.

	Returns:
		X: features as float array (n_samples, n_features)
		y: int labels (0..K-1)
		target_names: list of original class names in the order used for y
	"""
	features: List[List[float]] = []
	labels: List[str] = []
	# Open CSV and read rows
	with open(path, newline="") as f:
		reader = csv.reader(f)
		headers = next(reader)  # skip header row
		for row in reader:
			features.append([float(x) for x in row[:-1]])  # features
			labels.append(row[-1])  # label
	# Convert features to numpy
	X = np.array(features, dtype=float)
	# Build mapping from label name to integer id in sorted order to be consistent
	unique = sorted(list(set(labels)))
	label_to_int = {label: idx for idx, label in enumerate(unique)}
	y = np.array([label_to_int[val] for val in labels], dtype=int)
	return X, y, unique


def split_data(X: np.ndarray, y: np.ndarray, seed: int = 7):
	"""Split into train, validation, test (60/20/20)."""
	# Split out test set (20%)
	X_train_val, X_test, y_train_val, y_test = train_test_split(
		X, y, test_size=0.2, random_state=seed, stratify=y
	)
	# From the remaining 80%, take 25% as validation (=> 20% of total)
	X_train, X_val, y_train, y_val = train_test_split(
		X_train_val, y_train_val, test_size=0.25, random_state=seed, stratify=y_train_val
	)
	return X_train, X_val, X_test, y_train, y_val, y_test


def evaluate_on_test(name: str, model, X_test, y_test, target_names: List[str]) -> None:
	"""Compute predictions, print metrics, and save confusion matrix plot/CSV."""
	# Predict class ids for test samples
	y_pred = model.predict(X_test)
	# Overall accuracy
	acc = accuracy_score(y_test, y_pred)
	# Per-class precision/recall/F1 and support counts
	prec, rec, f1, support = precision_recall_fscore_support(y_test, y_pred, average=None, zero_division=0)
	# Confusion matrix with explicit label order [0..K-1]
	label_order = list(range(len(target_names)))
	cm = confusion_matrix(y_test, y_pred, labels=label_order)

	# Print scalar and per-class metrics
	print(f"\n=== {name} Test Metrics ===")
	print(f"Accuracy: {acc:.4f}")
	print("Per-class precision/recall/F1:")
	for i, t in enumerate(target_names):
		print(f"  {t:>12} -> P={prec[i]:.4f}, R={rec[i]:.4f}, F1={f1[i]:.4f}, N={support[i]}")
	print("Confusion Matrix (array):")
	print(cm)
	print("Confusion Matrix (list):")
	print(cm.tolist())
	print("Classification Report:")
	print(classification_report(y_test, y_pred, target_names=target_names, zero_division=0))

	# Save confusion matrix heatmap with matching label order on axes
	os.makedirs(PLOTS_DIR, exist_ok=True)
	plt.figure(figsize=(5, 4))
	sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
		xticklabels=target_names, yticklabels=target_names)
	plt.xlabel("Predicted")
	plt.ylabel("True")
	plt.title("Confusion Matrix - Multiclass")
	plt.tight_layout()
	plt.savefig(os.path.join(PLOTS_DIR, f"multiclass_confusion_matrix.png"))
	plt.close()

	# Also save confusion matrix as CSV with labeled rows/cols
	csv_path = os.path.join(PLOTS_DIR, "multiclass_confusion_matrix.csv")
	with open(csv_path, mode="w", newline="") as f:
		writer = csv.writer(f)
		writer.writerow(["true/pred", *[f"pred_{t}" for t in target_names]])
		for i, t in enumerate(target_names):
			writer.writerow([f"true_{t}", *[cm[i, j] for j in range(len(target_names))]])

def select_best_model_via_val(X_train, y_train, X_val, y_val):
	"""Train several models and return the one with highest validation accuracy (name, params)."""
	candidates = []

	# k-NN (with standardization)
	for k in [3, 5, 11]:
		pipe = Pipeline([
			("scaler", StandardScaler()),
			("clf", KNeighborsClassifier(n_neighbors=k)),
		])
		pipe.fit(X_train, y_train)
		val_acc = pipe.score(X_val, y_val)
		candidates.append(("KNeighborsClassifier", {"n_neighbors": k}, pipe, val_acc))

	# Decision Tree
	for depth in [None, 5, 10, 20]:
		model = DecisionTreeClassifier(max_depth=depth, random_state=0)
		model.fit(X_train, y_train)
		val_acc = model.score(X_val, y_val)
		candidates.append(("DecisionTreeClassifier", {"max_depth": depth}, model, val_acc))

	# Random Forest
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

	# Choose the model with the highest validation accuracy
	best = max(candidates, key=lambda t: t[3])
	best_name, best_params, best_model, best_val = best
	print(f"Best validation model: {best_name} with params {best_params}, val_acc={best_val:.4f}")
	return best_name, best_params


def refit_best_and_evaluate(best_name: str, best_params: dict, X_train, y_train, X_val, y_val, X_test, y_test, target_names: List[str]):
	"""Refit the selected model on train+val and evaluate it on the test set."""
	# Merge train and val for final training
	X_tr = np.vstack([X_train, X_val])
	y_tr = np.concatenate([y_train, y_val])

	# Recreate the best model with tuned hyperparameters, train, then evaluate
	if best_name == "KNeighborsClassifier":
		pipe = Pipeline([
			("scaler", StandardScaler()),
			("clf", KNeighborsClassifier(n_neighbors=best_params["n_neighbors"])),
		])
		pipe.fit(X_tr, y_tr)
		evaluate_on_test("KNeighborsClassifier", pipe, X_test, y_test, target_names)
	elif best_name == "DecisionTreeClassifier":
		model = DecisionTreeClassifier(max_depth=best_params["max_depth"], random_state=0)
		model.fit(X_tr, y_tr)
		evaluate_on_test("DecisionTreeClassifier", model, X_test, y_test, target_names)
	elif best_name == "RandomForestClassifier":
		model = RandomForestClassifier(
			n_estimators=best_params["n_estimators"],
			random_state=0,
			max_depth=best_params["max_depth"],
			n_jobs=-1,
		)
		model.fit(X_tr, y_tr)
		evaluate_on_test("RandomForestClassifier", model, X_test, y_test, target_names)
	else:
		raise ValueError("Unknown model name")


def main():
	"""Entry point to load data, train models, select best, and evaluate."""
	print(f"Loading multiclass dataset from: {DATA_PATH}")
	X, y, target_names = load_multiclass_dataset(DATA_PATH)
	X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y, seed=7)
	# Select best classical model by validation accuracy and evaluate on test
	best_name, best_params = select_best_model_via_val(X_train, y_train, X_val, y_val)
	refit_best_and_evaluate(best_name, best_params, X_train, y_train, X_val, y_val, X_test, y_test, target_names)


if __name__ == "__main__":
	main()
