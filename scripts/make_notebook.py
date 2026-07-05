import json
import re

with open('11_shap_feature_attribution.py', 'r', encoding='utf-8') as f:
    code = f.read()

# Split by the header comments
parts = re.split(r'(# ──.*?──+)\n', code)

cells = []
# First part might be imports
if parts[0].strip():
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + '\n' for line in parts[0].strip().split('\n')]
    })

for i in range(1, len(parts), 2):
    header = parts[i]
    content = parts[i+1].strip()
    
    # We can put the header as a markdown cell or just keep it in the code
    # The user provided it as python comments, so let's keep it as code cells
    source = header + '\n' + content
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + '\n' for line in source.split('\n')]
    })

notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 4
}

with open('../notebooks/11_shap_feature_attribution.ipynb', 'w', encoding='utf-8') as f:
    json.dump(notebook, f, indent=1)
print("Successfully generated notebook!")
