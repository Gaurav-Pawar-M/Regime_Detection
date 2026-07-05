import json
import os
import subprocess
import sys
import tempfile

def extract_and_run_notebook(notebook_path):
    print(f"\n{'='*50}\nRunning {notebook_path}\n{'='*50}")
    
    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)
        
    code_cells = []
    for cell in nb['cells']:
        if cell['cell_type'] == 'code':
            source = ''.join(cell['source'])
            # Don't run display or % magic commands that break outside jupyter
            source = source.replace("display(", "print(")
            # Comment out %matplotlib inline
            source = "\n".join([line if not line.strip().startswith("%") else f"#{line}" for line in source.split("\n")])
            code_cells.append(source)
            
    full_code = "\n\n".join(code_cells)
    
    # We must run it from the root directory because all notebooks assume paths like '../data/'
    # Wait, the notebooks are IN the notebooks folder. So if we run them from inside notebooks/
    # it works. Let's create the temp file inside notebooks/ to maintain relative paths.
    
    temp_fd, temp_path = tempfile.mkstemp(suffix='.py', dir='../notebooks')
    try:
        with os.fdopen(temp_fd, 'w', encoding='utf-8') as f:
            f.write("import matplotlib\nmatplotlib.use('Agg')\nimport matplotlib.pyplot as plt\n\n")
            f.write(full_code)
            
        result = subprocess.run([sys.executable, os.path.basename(temp_path)], cwd='../notebooks', check=True)
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

def run_script(script_path):
    print(f"\n{'='*50}\nRunning {script_path}\n{'='*50}")
    # run from scripts directory
    subprocess.run([sys.executable, os.path.basename(script_path)], cwd='../scripts', check=True)

if __name__ == '__main__':
    # Execute Pipeline Stage 1
    try:
        extract_and_run_notebook('../notebooks/01_data_acquisition.ipynb')
        run_script('../scripts/01b_fetch_extra.py')
        extract_and_run_notebook('../notebooks/02_baseline_hmm.ipynb')
        extract_and_run_notebook('../notebooks/03_event_calendar.ipynb')
        extract_and_run_notebook('../notebooks/06_per_stock_fitting.ipynb')
        
        print("\n\n>>> STAGE 1 COMPLETE! <<<")
        print("Please manually verify the disagreements in data/disagreement_table.parquet.")
        
    except Exception as e:
        print(f"\nPipeline failed: {e}")
        sys.exit(1)
