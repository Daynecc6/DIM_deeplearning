#!/usr/bin/env python3
import subprocess
import sys
import os

def main():
    """
    Driver script to reproduce the main evaluation results by calling evaluate_models.py
    with a predefined set of important run IDs.
    """
    run_ids_to_evaluate = ["1", "11", "323", "114", "119"]
    
    print("Starting evaluation of main model variants...")
    print(f"This will run: python evaluate_models.py --runs {' '.join(run_ids_to_evaluate)}")
    print("-" * 70)

    try:
        # Get the directory of the current script
        script_dir = os.path.dirname(os.path.abspath(__file__))
        # Construct the absolute path to evaluate_models.py
        evaluate_script_path = os.path.join(script_dir, "evaluate_models.py")

        print(f"DEBUG: sys.executable is: {sys.executable}")
        print(f"DEBUG: current working directory is: {os.getcwd()}")
        print(f"DEBUG: path to evaluate_models.py is: {evaluate_script_path}")
        
        # Construct the command using the absolute path to the script
        command = [sys.executable, evaluate_script_path, "--runs"] + run_ids_to_evaluate
        
        # Run the command
        # We capture and print output in real-time.
        # Set PYTHONIOENCODING=utf-8 for the subprocess to ensure it uses UTF-8 for its stdio.
        sub_env = {**os.environ, 'PYTHONIOENCODING': 'utf-8'}
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, universal_newlines=True, encoding='utf-8', errors='replace', env=sub_env)
        
        if process.stdout:
            for line in process.stdout:
                print(line, end='') # Print each line as it comes
        
        process.wait() # Wait for the subprocess to complete

        print("-" * 70)
        if process.returncode == 0:
            print("Evaluation completed successfully.")
            print("Please find the generated plots (e.g., accuracy_comparison.png, etc.) in the project directory.")
        else:
            print(f"Evaluation script failed with exit code {process.returncode}.")
            
    except FileNotFoundError:
        print("Error: evaluate_models.py not found. Make sure it's in the same directory as this script.")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()