import os
import sys
import joblib
import pandas as pd
import scipy.sparse
from sklearn.feature_extraction.text import TfidfVectorizer

def preprocess_and_vectorize():
    # Define file paths based on the project structure
    cleaned_dir = os.path.join('dataset', 'cleaned')
    processed_dir = os.path.join('dataset', 'processed')
    models_dir = 'models'

    # Ensure output directories exist
    os.makedirs(processed_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)

    file_paths = {
        'train': os.path.join(cleaned_dir, 'train.csv'),
        'valid': os.path.join(cleaned_dir, 'valid.csv'),
        'test': os.path.join(cleaned_dir, 'test.csv')
    }

    try:
        print("Loading cleaned datasets...")
        # Load datasets
        df_train = pd.read_csv(file_paths['train'])
        df_valid = pd.read_csv(file_paths['valid'])
        df_test = pd.read_csv(file_paths['test'])

        # Verify required columns exist
        for name, df in zip(['train', 'valid', 'test'], [df_train, df_valid, df_test]):
            if not {'statement', 'label'}.issubset(df.columns):
                raise ValueError(f"Missing 'statement' or 'label' column in {name}.csv")

        print("Handling missing text data...")
        # Handle missing text safely by replacing NaN with empty strings
        df_train['statement'] = df_train['statement'].fillna('')
        df_valid['statement'] = df_valid['statement'].fillna('')
        df_test['statement'] = df_test['statement'].fillna('')

        # Extract input features (X) and target labels (y)
        X_train, y_train = df_train['statement'], df_train['label']
        X_valid, y_valid = df_valid['statement'], df_valid['label']
        X_test, y_test = df_test['statement'], df_test['label']

        print("Initializing and fitting TF-IDF Vectorizer...")
        # Initialize TF-IDF Vectorizer with required parameters
        vectorizer = TfidfVectorizer(
            max_features=30000,
            stop_words='english',
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.95
        )

        # Fit on training data ONLY to prevent data leakage, then transform
        X_train_tfidf = vectorizer.fit_transform(X_train)
        
        # Transform validation and testing data using the fitted vectorizer
        X_valid_tfidf = vectorizer.transform(X_valid)
        X_test_tfidf = vectorizer.transform(X_test)

        print("Saving processed features and labels...")
        # Save sparse feature matrices to .npz format
        scipy.sparse.save_npz(os.path.join(processed_dir, 'X_train.npz'), X_train_tfidf)
        scipy.sparse.save_npz(os.path.join(processed_dir, 'X_valid.npz'), X_valid_tfidf)
        scipy.sparse.save_npz(os.path.join(processed_dir, 'X_test.npz'), X_test_tfidf)

        # Save labels to CSV
        y_train.to_csv(os.path.join(processed_dir, 'y_train.csv'), index=False, header=['label'])
        y_valid.to_csv(os.path.join(processed_dir, 'y_valid.csv'), index=False, header=['label'])
        y_test.to_csv(os.path.join(processed_dir, 'y_test.csv'), index=False, header=['label'])

        print("Saving TF-IDF vectorizer model...")
        # Save the vectorizer for future inference
        vectorizer_path = os.path.join(models_dir, 'tfidf_vectorizer.pkl')
        joblib.dump(vectorizer, vectorizer_path)

        # Display success metrics
        print("\n--- Process Completed Successfully ---")
        print(f"Vocabulary Size (Features): {len(vectorizer.vocabulary_)}")
        print(f"Training Data Shape:   {X_train_tfidf.shape}")
        print(f"Validation Data Shape: {X_valid_tfidf.shape}")
        print(f"Testing Data Shape:    {X_test_tfidf.shape}")
        print(f"\nFiles saved in '{processed_dir}/' and '{models_dir}/'.")

    except FileNotFoundError as e:
        print(f"\nError: Could not find the dataset file. {e}", file=sys.stderr)
        print("Ensure you are running the script from the 'zero-rag-project' root folder.", file=sys.stderr)
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}", file=sys.stderr)

if __name__ == "__main__":
    preprocess_and_vectorize()