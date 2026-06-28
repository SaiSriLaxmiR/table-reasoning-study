#!/bin/bash
# run_all.sh — Run the complete Table Reasoning Study pipeline
# Run from inside the table-reasoning-study/ folder

set -e

echo "=============================================="
echo "TABLE REASONING STUDY — Full Pipeline"
echo "=============================================="

# Create folders
mkdir -p data results

# Step 1: Collect tables from Wikipedia
echo ""
echo "Step 1/4: Collecting Wikipedia infoboxes..."
python collect_tables.py

# Step 2: Generate questions per table
echo ""
echo "Step 2/4: Generating questions..."
python generate_questions.py

# Step 3: Generate perturbations
echo ""
echo "Step 3/4: Generating perturbations..."
python perturb_tables.py

# Step 4: Evaluate models
echo ""
echo "Step 4/4: Evaluating models..."
python evaluate_models.py

# Step 5: Analyse results
echo ""
echo "Step 5/5: Analysing results..."
python analyse_results.py

echo ""
echo "=============================================="
echo "Done. Results in results/findings.csv"
echo "=============================================="
