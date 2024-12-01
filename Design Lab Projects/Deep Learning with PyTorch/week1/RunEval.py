import argparse
import sys

def load_and_evaluate(model_path, model_type):
    """
    Loads the specified model and runs the evaluation.
    
    Parameters:
    - model_path: str, the path to the saved model file.
    - model_type: str, the type of the model ('DNN', 'CNN', or 'LSTM').
    """
    if model_type == 'DNN':
        from DNN_eval import evaluate_model
    elif model_type == 'CNN':
        from CNN_eval import evaluate_model
    elif model_type == 'LSTM':
        from LSTM_eval import evaluate_model
    else:
        print(f"Error: Unsupported model type '{model_type}'.")
        sys.exit(1)

    # Run the evaluation and print the results
    evaluation_metrics = evaluate_model(model_path)
    print("Evaluation Metrics:", evaluation_metrics)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a Deep Learning model.")
    parser.add_argument("model_path", type=str, help="Path to the saved model file.")
    parser.add_argument("model_type", type=str, choices=['DNN', 'CNN', 'LSTM'], help="Type of the DL model ('DNN', 'CNN', 'LSTM').")
    
    args = parser.parse_args()
    
    # Load and evaluate the specified model
    load_and_evaluate(args.model_path, args.model_type)
