import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns

# Load and prepare data
def load_data():
    df = pd.read_csv('numeric_data.csv')
    
    # Encode labels if not already encoded
    if 'label' not in df.columns:
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        df['label'] = le.fit_transform(df['slice_type'])
    
    X = df.drop(['slice_type', 'label'], axis=1, errors='ignore')
    y = df['label']
    
    return X, y

# Train and evaluate models
def evaluate_models(X, y):
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Scale features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # Initialize models
    models = {
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'SVM': SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42),
        'XGBoost': xgb.XGBClassifier(n_estimators=100, random_state=42)
    }
    
    # Train and evaluate each model
    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)
        
        results[name] = {
            'accuracy': acc,
            'report': report
        }
        
        print(f"\n{name} Results:")
        print(f"Accuracy: {acc:.4f}")
        print(classification_report(y_test, y_pred))
    
    return results

# Feature importance analysis
def analyze_features(X, y):
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    
    # Get feature importances
    importances = model.feature_importances_
    features = X.columns
    indices = np.argsort(importances)[::-1]
    
    # Plot feature importance
    plt.figure(figsize=(10, 6))
    plt.title("Feature Importances")
    plt.bar(range(X.shape[1]), importances[indices], align='center')
    plt.xticks(range(X.shape[1]), [features[i] for i in indices], rotation=90)
    plt.tight_layout()
    plt.savefig('feature_importance.png')
    plt.close()
    
    return importances

# Correlation analysis
def plot_correlation_matrix(df):
    plt.figure(figsize=(12, 10))
    corr = df.corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm', 
                cbar=True, square=True)
    plt.title('Feature Correlation Matrix')
    plt.tight_layout()
    plt.savefig('correlation_matrix.png')
    plt.close()

def main():
    # Load data
    X, y = load_data()
    
    # Convert to DataFrame for visualization
    df = X.copy()
    df['label'] = y
    
    # Perform analyses
    results = evaluate_models(X, y)
    importances = analyze_features(X, y)
    plot_correlation_matrix(df)
    
    # Save results
    import json
    with open('traditional_ml_results.json', 'w') as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    main()