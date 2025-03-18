import pandas as pd
import re
import sys
import os

def extract_param_count(model_name):
    """Extract parameter count from model name string."""
    # Look for patterns like "0.5B", "7B", "32B", "1.5B", etc.
    match = re.search(r'(\d+(?:\.\d+)?)[bB]', model_name)
    if match:
        return float(match.group(1))
    else:
        # For models that don't follow the pattern
        print(f"Warning: Couldn't extract parameter count from {model_name}")
        return None

def add_param_count_column(input_csv, output_csv=None):
    """Add parameter count column to CSV file."""
    if output_csv is None:
        # Create output filename based on input filename
        base, ext = os.path.splitext(input_csv)
        output_csv = f"{base}_with_params{ext}"
    
    # Read CSV (Auto-detect separator)
    try:
        df = pd.read_csv(input_csv, sep='\t')
    except:
        try:
            df = pd.read_csv(input_csv)
        except Exception as e:
            print(f"Error reading CSV: {e}")
            return False
    
    # Check if 'model' column exists
    if 'model' not in df.columns:
        print("Error: CSV doesn't have a 'model' column")
        return False
    
    # Check if param_count already exists
    if 'param_count' in df.columns:
        print("Warning: 'param_count' column already exists. Overwriting.")
    
    # Extract parameter counts
    df['param_count'] = df['model'].apply(extract_param_count)
    
    # Save to new CSV with same separator as input
    df.to_csv(output_csv, sep='\t', index=False)
    print(f"Successfully added param_count column. Saved to {output_csv}")
    return True

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python add_param_count.py input.csv [output.csv]")
        sys.exit(1)
    
    input_csv = sys.argv[1]
    output_csv = sys.argv[2] if len(sys.argv) > 2 else None
    
    success = add_param_count_column(input_csv, output_csv)
    sys.exit(0 if success else 1)