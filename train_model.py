# train_model.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from xgboost import XGBClassifier
from joblib import dump
from feature_extraction import FeatureExtraction
from scipy import sparse
from tqdm import tqdm
import json
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report

def load_data(benign_path='benign_urls.csv', malicious_path='malicious_urls.csv'):
    b_chunks = pd.read_csv(benign_path, chunksize=100000)
    m_chunks = pd.read_csv(malicious_path, chunksize=100000)

    b_list, m_list = [], []
    for chunk in b_chunks:
        if 'url' not in chunk.columns:
            chunk = chunk.rename(columns={chunk.columns[0]: 'url'})
        chunk = chunk[['url']].dropna().drop_duplicates()
        chunk['label'] = 0
        b_list.append(chunk)

    for chunk in m_chunks:
        if 'url' not in chunk.columns:
            chunk = chunk.rename(columns={chunk.columns[0]: 'url'})
        chunk = chunk[['url']].dropna().drop_duplicates()
        chunk['label'] = 1
        m_list.append(chunk)

    df = pd.concat(b_list + m_list, ignore_index=True)
    df = df.drop_duplicates(subset=['url']).reset_index(drop=True)
    return df

def extract_engineered_features(urls):
    feats = []
    for u in tqdm(urls, desc="Extracting features"):
        fe = FeatureExtraction(u)
        feats.append(fe.to_dict())
    df = pd.DataFrame(feats).fillna(0)
    return df

def main():
    print("Loading data...")
    df = load_data()
    print(f"Total URLs loaded: {len(df)}")
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    X_text = df['url'].astype(str).values
    y = df['label'].values

    print("Extracting engineered features...")
    X_eng = extract_engineered_features(X_text)
    print("Engineered features shape:", X_eng.shape)

    # Save engineered feature names
    X_eng.columns.to_series().to_json('engineered_feature_names.json')

    print("Fitting TF-IDF...")
    tfidf = TfidfVectorizer(analyzer='char_wb', ngram_range=(3,6), max_features=20000)
    X_tfidf = tfidf.fit_transform(X_text)
    print("TF-IDF shape:", X_tfidf.shape)

    print("Scaling engineered features...")
    scaler = StandardScaler()
    X_eng_scaled = scaler.fit_transform(X_eng)

    print("Combining features...")
    X_combined = sparse.hstack([X_tfidf, sparse.csr_matrix(X_eng_scaled)], format='csr')

    print("Train/test split...")
    X_train, X_test, y_train, y_test = train_test_split(
        X_combined, y, test_size=0.2, stratify=y, random_state=42
    )

    print("Training XGBoost...")
    model = XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        tree_method='hist',
        eval_metric='logloss',
        n_jobs=-1,
        random_state=42,
        use_label_encoder=False
    )
    model.fit(X_train, y_train)

    print("Evaluating...")
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:,1]
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("ROC AUC:", roc_auc_score(y_test, y_proba))
    print(classification_report(y_test, y_pred))

    print("Saving artifacts...")
    dump(model, 'phishing_model.joblib')
    dump(tfidf, 'tfidf_vectorizer.joblib')
    dump(scaler, 'scaler.joblib')
    print("Saved: phishing_model.joblib, tfidf_vectorizer.joblib, scaler.joblib")
    print("Done.")

if __name__ == '__main__':
    main()
