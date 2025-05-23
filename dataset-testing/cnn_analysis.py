import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder

# Configuration
PATTERN_TYPES = ['guided', 'perlin', 'wallpaper', 'fractal']
IMG_SIZE = 16
BATCH_SIZE = 32
EPOCHS = 30

# Load data and labels
def load_data(pattern_type='guided'):
    # Load pattern data
    patterns = np.load(f'{pattern_type}_patterns.npy')
    
    # Load metadata
    df = pd.read_csv('numeric_data.csv')
    
    # Encode labels if not already present
    if 'label' not in df.columns:
        le = LabelEncoder()
        df['label'] = le.fit_transform(df['slice_type'])
    
    # Ensure shapes match
    assert len(patterns) == len(df), "Patterns and metadata length mismatch"
    
    return patterns, df['label'].values

# Build CNN model
def build_model(input_shape, num_classes):
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=input_shape),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        
        Conv2D(64, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        
        Conv2D(128, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        
        Flatten(),
        Dense(256, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    
    model.compile(optimizer=Adam(learning_rate=0.001),
                 loss='sparse_categorical_crossentropy',
                 metrics=['accuracy'])
    
    return model

# Train and evaluate model
def train_and_evaluate(X, y, pattern_name):
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Build model
    model = build_model((IMG_SIZE, IMG_SIZE, 3), len(np.unique(y)))
    
    # Train model
    history = model.fit(
        X_train, y_train,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        validation_data=(X_test, y_test),
        verbose=1
    )
    
    # Evaluate
    y_pred = np.argmax(model.predict(X_test), axis=1)
    report = classification_report(y_test, y_pred, output_dict=True)
    
    # Plot training history
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title(f'{pattern_name} - Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title(f'{pattern_name} - Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(f'{pattern_name}_training_history.png')
    plt.close()
    
    return model, report, history

# Compare pattern types
def compare_patterns():
    results = {}
    
    for pattern in PATTERN_TYPES:
        print(f"\n=== Processing {pattern} patterns ===")
        X, y = load_data(pattern)
        model, report, history = train_and_evaluate(X, y, pattern)
        results[pattern] = {
            'report': report,
            'val_accuracy': max(history.history['val_accuracy'])
        }
    
    # Plot comparison
    plt.figure(figsize=(10, 6))
    sns.barplot(x=list(results.keys()), 
                y=[results[p]['val_accuracy'] for p in results])
    plt.title('Validation Accuracy by Pattern Type')
    plt.ylabel('Accuracy')
    plt.ylim(0, 1)
    plt.savefig('pattern_comparison.png')
    plt.close()
    
    return results

def main():
    # Compare all pattern types
    results = compare_patterns()
    
    # Save results
    import json
    with open('cnn_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # Print summary
    print("\n=== Final Results ===")
    for pattern, data in results.items():
        print(f"\n{pattern.upper()} Patterns:")
        print(f"Best Validation Accuracy: {data['val_accuracy']:.4f}")
        print("Classification Report:")
        print(classification_report(None, None, output_dict=data['report']))

if __name__ == "__main__":
    # Set random seeds for reproducibility
    np.random.seed(42)
    tf.random.set_seed(42)
    
    main()