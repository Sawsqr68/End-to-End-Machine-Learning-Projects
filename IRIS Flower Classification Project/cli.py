#!/usr/bin/env python3
"""
CLI for IRIS Flower Classification
Predict iris flower species based on measurements
"""
import argparse
import numpy as np
import sys
try:
    from sklearn.externals import joblib
except ImportError:
    import joblib


def load_model(model_path='./data/model.sav'):
    """Load the trained model"""
    try:
        model = joblib.load(model_path)
        return model
    except FileNotFoundError:
        print(f"Error: Model file not found at {model_path}")
        sys.exit(1)
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)


def predict_species(model, sepal_length, sepal_width, petal_length, petal_width):
    """Predict iris species from measurements"""
    try:
        import pandas as pd
        # Use DataFrame with feature names to avoid warning
        features = pd.DataFrame([[sepal_length, sepal_width, petal_length, petal_width]], 
                               columns=['sepal_length', 'sepal_width', 'petal_length', 'petal_width'])
        prediction = model.predict(features)[0]
        return prediction
    except Exception as e:
        print(f"Error making prediction: {e}")
        sys.exit(1)


def interactive_mode(model):
    """Interactive mode for predictions"""
    print("\n=== IRIS Flower Species Predictor ===")
    print("Enter flower measurements to predict the species\n")
    
    try:
        sepal_length = float(input("Sepal Length (cm): "))
        sepal_width = float(input("Sepal Width (cm): "))
        petal_length = float(input("Petal Length (cm): "))
        petal_width = float(input("Petal Width (cm): "))
        
        species = predict_species(model, sepal_length, sepal_width, petal_length, petal_width)
        print(f"\nPredicted Species: {species}")
        
    except ValueError:
        print("Error: Please enter valid numeric values")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n\nExiting...")
        sys.exit(0)


def main():
    parser = argparse.ArgumentParser(
        description='Predict IRIS flower species from measurements',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Interactive mode
  python cli.py
  
  # Predict with arguments
  python cli.py --sepal-length 5.1 --sepal-width 3.5 --petal-length 1.4 --petal-width 0.2
  
  # Short form
  python cli.py -sl 5.1 -sw 3.5 -pl 1.4 -pw 0.2
        """
    )
    
    parser.add_argument('-sl', '--sepal-length', type=float, help='Sepal length in cm')
    parser.add_argument('-sw', '--sepal-width', type=float, help='Sepal width in cm')
    parser.add_argument('-pl', '--petal-length', type=float, help='Petal length in cm')
    parser.add_argument('-pw', '--petal-width', type=float, help='Petal width in cm')
    parser.add_argument('-m', '--model', default='./data/model.sav', help='Path to model file')
    
    args = parser.parse_args()
    
    # Load model
    model = load_model(args.model)
    
    # Check if all measurements are provided
    if all([args.sepal_length, args.sepal_width, args.petal_length, args.petal_width]):
        # Argument mode
        species = predict_species(model, args.sepal_length, args.sepal_width, 
                                args.petal_length, args.petal_width)
        print(f"Predicted Species: {species}")
    elif any([args.sepal_length, args.sepal_width, args.petal_length, args.petal_width]):
        # Partial arguments provided
        print("Error: Please provide all four measurements or none for interactive mode")
        parser.print_help()
        sys.exit(1)
    else:
        # Interactive mode
        interactive_mode(model)


if __name__ == '__main__':
    main()
