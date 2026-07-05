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
            source = source.replace("display(", "print(")
            source = "\n".join([line if not line.strip().startswith("%") else f"#{line}" for line in source.split("\n")])
            code_cells.append(source)
            
    full_code = "\n\n".join(code_cells)
    
    temp_fd, temp_path = tempfile.mkstemp(suffix='.py', dir='../notebooks')
    try:
        with os.fdopen(temp_fd, 'w', encoding='utf-8') as f:
            f.write("import matplotlib\nmatplotlib.use('Agg')\nimport matplotlib.pyplot as plt\n\n")
            f.write(full_code)
            
        result = subprocess.run([sys.executable, os.path.basename(temp_path)], cwd='../notebooks', check=True)
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

if __name__ == '__main__':
    try:
        extract_and_run_notebook('../notebooks/03_event_calendar.ipynb')
        extract_and_run_notebook('../notebooks/06_per_stock_fitting.ipynb')
        
        print("\n\n>>> STAGE 4 COMPLETE! <<<")
        print("Disagreements have been generated based on REAL corporate events.")
        
    except Exception as e:
        print(f"\nPipeline failed: {e}")
        sys.exit(1)
