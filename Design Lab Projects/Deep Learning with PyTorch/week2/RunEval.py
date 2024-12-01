import sys
import subprocess

def run_evaluation(model_type, checkpoint_path, test_dataset_path):
    """
    Runs the appropriate evaluation script based on the model type.
    Only AutoModel.py accepts command-line arguments for checkpoint_path and test_dataset_path.

    Parameters:
    - model_type: A string indicating the model type ('DNN', 'CNN', or 'AutoModel').
    - checkpoint_path: The path to the model's checkpoint (used only for AutoModel).
    - test_dataset_path: The path to the test dataset (used only for AutoModel).
    """
    if model_type == 'DNN':
        script_name = 'DNN_classification.py'
        command = ['python3', script_name]
    elif model_type == 'CNN':
        script_name = 'CNN_classification.py'
        command = ['python3', script_name]
    elif model_type == 'AutoModel':
        script_name = 'AutoModel.py'
        command = ['python3', script_name, checkpoint_path, test_dataset_path]
    else:
        raise ValueError("Unsupported model type. Choose from 'DNN', 'CNN', or 'AutoModel'.")

    # Run the command
    subprocess.run(command, check=True)

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python3 RunEval.py <model_type> <checkpoint_path> <test_dataset_path>")
        sys.exit(1)
    
    model_type = sys.argv[1]
    checkpoint_path = sys.argv[2]
    test_dataset_path = sys.argv[3]

    run_evaluation(model_type, checkpoint_path, test_dataset_path)
