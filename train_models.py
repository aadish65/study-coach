import os
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.sparse import load_npz
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score, 
    classification_report, 
    confusion_matrix
)

def main():
    print("Starting Day 3: Machine Learning Model Training...")
    
    # Define file paths
    processed_dir = os.path.join("dataset", "processed")
    models_dir = "models"
    
    # Ensure models directory exists
    os.makedirs(models_dir, exist_ok=True)
    
    # 1. Load Data
    try:
        print("Loading preprocessed TF-IDF feature matrices and labels...")
        X_train = load_npz(os.path.join(processed_dir, "X_train.npz"))
        X_valid = load_npz(os.path.join(processed_dir, "X_valid.npz"))
        X_test = load_npz(os.path.join(processed_dir, "X_test.npz"))
        
        # Load labels assuming the target is the first column
        y_train = pd.read_csv(os.path.join(processed_dir, "y_train.csv")).iloc[:, 0]
        y_valid = pd.read_csv(os.path.join(processed_dir, "y_valid.csv")).iloc[:, 0]
        y_test = pd.read_csv(os.path.join(processed_dir, "y_test.csv")).iloc[:, 0]
        
        print(f"Data loaded successfully!")
        print(f"Training set: {X_train.shape[0]} samples")
        print(f"Validation set: {X_valid.shape[0]} samples")
        print(f"Test set: {X_test.shape[0]} samples")
        
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    # 2. Initialize Models
    models = {
        "Logistic Regression": LogisticRegression(
            class_weight="balanced", 
            max_iter=2000, 
            random_state=42
        ),
        "Multinomial Naive Bayes": MultinomialNB(),
        "LinearSVC": LinearSVC(
            class_weight="balanced", 
            max_iter=2000, 
            random_state=42,
            dual="auto" # Recommended for newer scikit-learn versions
        )
    }

    # 3. Train and Evaluate Models on Validation Data
    print("\nTraining models and evaluating on validation data...")
    results = []
    trained_models = {}

    for name, model in models.items():
        print(f"Training {name}...")
        try:
            # Train the model (No dense array conversion needed)
            model.fit(X_train, y_train)
            trained_models[name] = model
            
            # Predict on validation data
            y_val_pred = model.predict(X_valid)
            
            # Calculate metrics (Macro-average for 6 classes)
            acc = accuracy_score(y_valid, y_val_pred)
            prec = precision_score(y_valid, y_val_pred, average="macro", zero_division=0)
            rec = recall_score(y_valid, y_val_pred, average="macro", zero_division=0)
            f1 = f1_score(y_valid, y_val_pred, average="macro", zero_division=0)
            
            results.append({
                "Model": name,
                "Accuracy": acc,
                "Precision (Macro)": prec,
                "Recall (Macro)": rec,
                "F1-Score (Macro)": f1
            })
            
        except Exception as e:
            print(f"Error training {name}: {e}")

    # 4. Compare Models
    results_df = pd.DataFrame(results)
    print("\nModel Comparison (Validation Data):")
    print(results_df.to_string(index=False))
    
    # Save comparison to CSV
    comp_csv_path = os.path.join(models_dir, "model_comparison.csv")
    results_df.to_csv(comp_csv_path, index=False)
    print(f"Saved model comparison metrics to {comp_csv_path}")

    # 5. Generate Comparison Bar Chart
    plt.figure(figsize=(10, 6))
    results_melted = results_df.melt(id_vars="Model", var_name="Metric", value_name="Score")
    sns.barplot(x="Model", y="Score", hue="Metric", data=results_melted, palette="viridis")
    plt.title("Model Comparison on Validation Data")
    plt.ylim(0, 1.0)
    plt.ylabel("Score")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    comp_img_path = os.path.join(models_dir, "model_comparison.png")
    plt.savefig(comp_img_path)
    plt.close()
    print(f"Saved model comparison chart to {comp_img_path}")

    # 6. Select the Best Model
    best_model_name = results_df.loc[results_df["F1-Score (Macro)"].idxmax()]["Model"]
    best_model = trained_models[best_model_name]
    print(f"\nBest Model selected based on Validation Macro F1-Score: {best_model_name}")

    # 7. Evaluate Best Model on Test Data
    print("\nEvaluating Best Model on unseen Test Data...")
    try:
        y_test_pred = best_model.predict(X_test)
        
        print("\n--- Classification Report (Test Data) ---")
        print(classification_report(y_test, y_test_pred, zero_division=0))
        
        # 8. Generate and Save Confusion Matrix
        cm = confusion_matrix(y_test, y_test_pred, labels=best_model.classes_)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", 
                    xticklabels=best_model.classes_, 
                    yticklabels=best_model.classes_)
        plt.title(f"Confusion Matrix - {best_model_name} (Test Data)")
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        plt.tight_layout()
        cm_img_path = os.path.join(models_dir, "confusion_matrix.png")
        plt.savefig(cm_img_path)
        plt.close()
        print(f"Saved confusion matrix to {cm_img_path}")
        
    except Exception as e:
        print(f"Error during final evaluation: {e}")

    # 9. Save the Best Model
    try:
        best_model_path = os.path.join(models_dir, "best_model.pkl")
        joblib.dump(best_model, best_model_path)
        print(f"\nSuccessfully saved the best model to {best_model_path}")
    except Exception as e:
        print(f"Error saving the best model: {e}")

    print("Day 3 Training Completed.")

if __name__ == "__main__":
    main()