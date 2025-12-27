#!/usr/bin/env python3
"""
CLI for Breast Cancer Classification
Predict if breast cancer is Benign or Malignant based on cell measurements
"""
import argparse
import sys
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import joblib
import os


def load_or_train_model(data_path='data.csv', model_path='model.pkl', scaler_path='scaler.pkl'):
    """Load existing model and scaler or train new ones"""
    if os.path.exists(model_path) and os.path.exists(scaler_path):
        print(f"Loading existing model from {model_path}...")
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        return model, scaler
    
    print("Training new model...")
    try:
        # Load data
        data = pd.read_csv(data_path)
        
        # Drop unnecessary columns
        drop_var = ['Unnamed: 32', 'id']
        data.drop(drop_var, axis=1, inplace=True, errors='ignore')
        
        # Separate features and labels
        X = data.drop('diagnosis', axis=1)
        y = data['diagnosis']
        
        # Create and fit scaler
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)
        
        # Train model
        model = LogisticRegression(max_iter=10000, random_state=42)
        model.fit(X_train, y_train)
        
        # Save model and scaler
        joblib.dump(model, model_path)
        joblib.dump(scaler, scaler_path)
        print(f"Model trained and saved to {model_path}")
        print(f"Scaler saved to {scaler_path}")
        print(f"Model accuracy: {model.score(X_test, y_test):.2%}")
        
        return model, scaler
    except Exception as e:
        print(f"Error training model: {e}")
        sys.exit(1)


def predict_diagnosis(model, scaler, features):
    """Predict diagnosis from features"""
    try:
        # Scale features using the fitted scaler (suppress feature names warning)
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            features_scaled = scaler.transform(features.reshape(1, -1))
        prediction = model.predict(features_scaled)[0]
        probability = model.predict_proba(features_scaled)[0]
        
        return prediction, probability
    except Exception as e:
        print(f"Error making prediction: {e}")
        sys.exit(1)


def get_feature_names():
    """Return list of feature names in order"""
    return [
        'radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean',
        'smoothness_mean', 'compactness_mean', 'concavity_mean',
        'concave points_mean', 'symmetry_mean', 'fractal_dimension_mean',
        'radius_se', 'texture_se', 'perimeter_se', 'area_se',
        'smoothness_se', 'compactness_se', 'concavity_se',
        'concave points_se', 'symmetry_se', 'fractal_dimension_se',
        'radius_worst', 'texture_worst', 'perimeter_worst', 'area_worst',
        'smoothness_worst', 'compactness_worst', 'concavity_worst',
        'concave points_worst', 'symmetry_worst', 'fractal_dimension_worst'
    ]


def interactive_mode(model, scaler):
    """Interactive mode for predictions with simplified input"""
    print("\n=== Breast Cancer Diagnosis Predictor ===")
    print("Note: This model requires 30 features. For demo purposes, we'll use simplified input.")
    print("In a real scenario, these values would come from medical tests.\n")
    
    print("Please provide the following measurements (or press Enter to use defaults):")
    print("Typical ranges: radius (10-20), texture (10-30), perimeter (50-150)")
    print("               area (200-1500), smoothness (0.05-0.15)\n")
    
    try:
        # Get simplified input from user
        radius = input("Mean Radius (default: 14.0): ").strip()
        radius = float(radius) if radius else 14.0
        
        texture = input("Mean Texture (default: 19.0): ").strip()
        texture = float(texture) if texture else 19.0
        
        perimeter = input("Mean Perimeter (default: 92.0): ").strip()
        perimeter = float(perimeter) if perimeter else 92.0
        
        area = input("Mean Area (default: 655.0): ").strip()
        area = float(area) if area else 655.0
        
        smoothness = input("Mean Smoothness (default: 0.096): ").strip()
        smoothness = float(smoothness) if smoothness else 0.096
        
        # Generate remaining features based on typical patterns
        # This is a simplified approach for demo purposes
        features = np.array([
            radius, texture, perimeter, area, smoothness,
            0.10, 0.08, 0.05, 0.16, 0.06,  # mean features
            radius*0.03, texture*0.04, perimeter*0.3, area*2, smoothness*0.006,
            0.002, 0.003, 0.002, 0.01, 0.001,  # se features
            radius*1.1, texture*1.1, perimeter*1.2, area*1.3, smoothness*1.15,
            0.13, 0.11, 0.08, 0.21, 0.08  # worst features
        ])
        
        diagnosis, probability = predict_diagnosis(model, scaler, features)
        
        print(f"\n{'='*50}")
        print(f"Predicted Diagnosis: {diagnosis}")
        if diagnosis == 'M':
            print(f"  → Malignant (Cancerous)")
            print(f"  Confidence: {probability[1]:.1%}")
        else:
            print(f"  → Benign (Non-cancerous)")
            print(f"  Confidence: {probability[0]:.1%}")
        print(f"{'='*50}\n")
        
        print("⚠️  Note: This is for educational purposes only.")
        print("    Always consult healthcare professionals for actual diagnosis.")
        
    except ValueError:
        print("Error: Please enter valid numeric values")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n\nExiting...")
        sys.exit(0)


def main():
    parser = argparse.ArgumentParser(
        description='Predict breast cancer diagnosis (Benign/Malignant) from cell measurements',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Interactive mode (simplified input)
  python cli.py
  
  # Show model information
  python cli.py --info

Note: This tool requires 30 features for accurate prediction.
      Interactive mode uses simplified input for demonstration.
        """
    )
    
    parser.add_argument('--data', default='data.csv', help='Path to data CSV file')
    parser.add_argument('--model', default='model.pkl', help='Path to model file')
    parser.add_argument('--scaler', default='scaler.pkl', help='Path to scaler file')
    parser.add_argument('--info', action='store_true', help='Show model information')
    parser.add_argument('--train', action='store_true', help='Force retrain the model')
    
    args = parser.parse_args()
    
    # Force retrain if requested
    if args.train:
        if os.path.exists(args.model):
            os.remove(args.model)
        if os.path.exists(args.scaler):
            os.remove(args.scaler)
        print("Existing model and scaler removed. Retraining...")
    
    # Load or train model and scaler
    model, scaler = load_or_train_model(args.data, args.model, args.scaler)
    
    if args.info:
        print("\n=== Model Information ===")
        print(f"Model Type: Logistic Regression")
        print(f"Number of Features: 30")
        print(f"Classes: B (Benign), M (Malignant)")
        print(f"\nFeature Requirements:")
        for i, feature in enumerate(get_feature_names(), 1):
            print(f"  {i:2d}. {feature}")
        return
    
    # Interactive mode
    interactive_mode(model, scaler)


if __name__ == '__main__':
    main()
