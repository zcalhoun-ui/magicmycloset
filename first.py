import os
import pandas as pd
import numpy as np

# 1. Get the script directory and build file paths
script_dir = os.path.dirname(os.path.abspath(__file__))
identifier_path = os.path.join(script_dir, "identifier.csv")
matrix_path = os.path.join(script_dir, "matrix.csv")
output_path = os.path.join(script_dir, "matrix_numerical.csv")

# 2. Load the CSV files
identifier_df = pd.read_csv(identifier_path)
matrix_df = pd.read_csv(matrix_path)

# 3. FLIPPED MAPPING:
# We now assume column 0 has the numerical targets and column 1 has the alphanumeric codes
target_values = identifier_df.iloc[:, 0] 
matrix_current_codes = identifier_df.iloc[:, 1].astype(str).str.strip().str.upper()

# Create the map where the keys match what is currently in your matrix
id_map = dict(zip(matrix_current_codes, target_values))

# 4. Clean the Matrix Data to ensure a perfect text match with the keys
for col in matrix_df.columns:
    matrix_df[col] = matrix_df[col].astype(str).str.strip().str.upper()

# 5. Convert every value in the matrix using the flipped dictionary map
converted_matrix_df = matrix_df.replace(id_map)

# 6. Save the result
converted_matrix_df.to_csv(output_path, index=False)

# 7. Print confirmation to your VS Code terminal
print("=== CONVERSION SUCCESSFUL WITH FLIPPED MAP ===")
print("New Map Sample (Current Matrix Value -> New Target Value):")
print(list(id_map.items())[:3])
print("\nFirst 3 rows of your successfully converted matrix:")
print(converted_matrix_df.head(3))

# build 3d dense tensor

# 1. Convert columns to integer coordinates
converted_matrix_df['Top'] = converted_matrix_df['Top'].astype(int)
converted_matrix_df['Bottom'] = converted_matrix_df['Bottom'].astype(int)
converted_matrix_df['A/L'] = converted_matrix_df['A/L'].astype(int)

print("Success to here")

# 2. Allocate the dense tensor (initialized with NaN for unrated combinations)
shape = (converted_matrix_df['Top'].max() + 1, converted_matrix_df['Bottom'].max() + 1, converted_matrix_df['A/L'].max() + 1)
scorer_tensor = np.full(shape, np.nan)

print("I allocated the tensor")

# 3. Populate the tensor using NumPy advanced indexing
scorer_tensor[converted_matrix_df['Top'].values, converted_matrix_df['Bottom'].values, converted_matrix_df['A/L'].values] = converted_matrix_df['Rating'].values

print("Successfully populated the tensor")






