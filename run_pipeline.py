"""
Master Script to Run Complete Pipeline
This script runs all steps in sequence: EDA -> Preprocessing -> Model Building
"""

import os
import sys
import subprocess
import time
from datetime import datetime

def print_header(text):
    """Print formatted header"""
    print("\n" + "="*80)
    print(f"  {text}")
    print("="*80 + "\n")

def run_script(script_name, description):
    """Run a Python script and handle errors"""
    print_header(f"STEP: {description}")
    print(f"Running: {script_name}")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    start_time = time.time()
    
    try:
        result = subprocess.run(
            [sys.executable, script_name],
            check=True,
            capture_output=True,
            text=True
        )
        
        # Print output
        if result.stdout:
            print(result.stdout)
        
        elapsed_time = time.time() - start_time
        print(f"\n✅ SUCCESS - Completed in {elapsed_time:.2f} seconds")
        return True
        
    except subprocess.CalledProcessError as e:
        elapsed_time = time.time() - start_time
        print(f"\n❌ ERROR - Failed after {elapsed_time:.2f} seconds")
        print(f"\nError output:\n{e.stderr}")
        return False

def main():
    """Main function to run the complete pipeline"""
    
    print_header("ALZHEIMER'S STAGE PREDICTION - COMPLETE PIPELINE")
    print("This script will run the complete analysis pipeline:")
    print("  1. Exploratory Data Analysis (EDA)")
    print("  2. Data Preprocessing & Feature Extraction")
    print("  3. Model Building & Training")
    print("\nThis may take 30-60 minutes depending on your system.")
    print("\nPress Ctrl+C to cancel at any time.")
    
    # Confirm
    try:
        input("\nPress Enter to continue or Ctrl+C to cancel...")
    except KeyboardInterrupt:
        print("\n\nCancelled by user.")
        return
    
    # Change to script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    
    # Pipeline steps
    steps = [
        ("1_eda_analysis.py", "Exploratory Data Analysis"),
        ("2_data_preprocessing.py", "Data Preprocessing & Feature Extraction"),
        ("3_model_building.py", "Model Building & Training")
    ]
    
    # Track overall progress
    overall_start = time.time()
    completed_steps = 0
    
    # Run each step
    for script_name, description in steps:
        success = run_script(script_name, description)
        
        if success:
            completed_steps += 1
        else:
            print_header("PIPELINE FAILED")
            print(f"Failed at step: {description}")
            print(f"Completed {completed_steps}/{len(steps)} steps")
            print("\nPlease check the error messages above and fix any issues.")
            return
    
    # Pipeline completed successfully
    overall_time = time.time() - overall_start
    
    print_header("PIPELINE COMPLETED SUCCESSFULLY! 🎉")
    print(f"Total time: {overall_time:.2f} seconds ({overall_time/60:.2f} minutes)")
    print(f"Completed steps: {completed_steps}/{len(steps)}")
    print("\n" + "="*80)
    print("NEXT STEPS:")
    print("="*80)
    print("\n1. Review the generated plots:")
    print("   - EDA plots: csvcaueeg/eda_plots/")
    print("   - Model plots: csvcaueeg/model_plots/")
    print("\n2. Check model performance:")
    print("   - Open: csvcaueeg/model_metadata.json")
    print("\n3. Make predictions:")
    print("   - Command line: python 4_prediction_pipeline.py <edf_file> --age <age>")
    print("   - Web app: streamlit run app.py")
    print("\n4. Launch the web application:")
    print("   streamlit run app.py")
    print("\n" + "="*80)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n❌ Pipeline interrupted by user.")
    except Exception as e:
        print(f"\n\n❌ Unexpected error: {str(e)}")
        import traceback
        traceback.print_exc()
