import os
import pandas as pd
import numpy as np
import tensorly as tl
from tensorly.decomposition import tucker 

def get_season():
    """Inputs: user input
    Outputs: string"""
    cite = True
    while cite:
        twoplus = str(input("Please choose a season. For simplicity's sake, choose 1 for winter, 2 for spring, 3 for summer, or 4 for fall. "))
        while twoplus not in ["1", "2", "3", "4"]:
            twoplus = str(input("Oops! Please press one of these 4 numbers: "))
        if twoplus == "1":
            cite = False
            file = "Winter"
        elif twoplus == "2": 
            cite = False
            file = "Spring"
        elif twoplus == "3":
            cite = False
            file = "Summer"
        elif twoplus == "4":
            cite = False
            file = "Winter"
        return file

# 1. Get the script directory and build file paths
def get_script_directory():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    return script_dir

# pivot longer the recommendations_df so that the index row has all of the values 
def pivot_longer(df, id_vars, var_name, value_name):
    return df.melt(id_vars=id_vars, var_name=var_name, value_name=value_name)

# Perform a left join to keep all rows and order from long_recommendations
def merge_with_identifier(long_recommendations, identifier_df):
    return pd.merge(
        long_recommendations,
        identifier_df,
        left_on='Index',        # The column name in long_recommendations
        right_on='Matrix ID',   # The matching column name in identifiers_df
        how='left'              # Keeps long_recommendations completely intact
    )

def wider(df, index_col, columns_col, values_col):
    return df.pivot(index=index_col, columns=columns_col, values=values_col).reset_index()

script_dir = get_script_directory()
identifier_path = os.path.join(script_dir, "identifier.csv")
matrix_path = os.path.join(script_dir, "matrix_numerical.csv")

# 2. Load the CSV files
identifier_df = pd.read_csv(identifier_path)
converted_matrix_df = pd.read_csv(matrix_path)
# remove bad outfit connections
converted_matrix_df = converted_matrix_df[converted_matrix_df['Rating'] != -1]

# 3. Convert columns to integer coordinates   
converted_matrix_df['Top'] = converted_matrix_df['Top'].astype(int)
converted_matrix_df['Bottom'] = converted_matrix_df['Bottom'].astype(int)
converted_matrix_df['A/L'] = converted_matrix_df['A/L'].astype(int)

print("Success to here")

# 8. Allocate the dense tensor (initialized with 0.0 for matrix math compatibility)
shape = (
    converted_matrix_df['Top'].max() + 1, 
    converted_matrix_df['Bottom'].max() + 1, 
    converted_matrix_df['A/L'].max() + 1
)
scorer_tensor = np.zeros(shape, dtype=float)  # Use float for standard linear algebra operations

print("I allocated the tensor")

# 9. Populate the tensor using NumPy advanced indexing (assigning 1.0 where present)
scorer_tensor[
    converted_matrix_df['Top'].values, 
    converted_matrix_df['Bottom'].values, 
    converted_matrix_df['A/L'].values
] = 1.0

print("Successfully populated the tensor")

# === Build LRA Matrix ===
# 1. Keep NumPy as the backend since you are using pandas values
tl.set_backend('numpy')

# 2. Use your existing environment variable directly
original_tensor = scorer_tensor

# 3. Dynamically read your true dimensions
dim1, dim2, dim3 = original_tensor.shape
print(f"Detected Data Dimensions: {dim1} x {dim2} x {dim3}")

custom_ranks = [12, 15, 20]

# Compute the decomposition
core, factors = tucker(scorer_tensor, rank=custom_ranks, init='svd')

# Reconstruct the dense low-rank approximation tensor
lra_tensor = tl.tucker_to_tensor((core, factors))

print(f"Reconstructed tensor shape: {lra_tensor.shape}") # Will be 21 x 26 x 42
print(f"Compressed core matrix shape: {core.shape}") 


print(f"Low-Rank Approximation completed successfully!")

# === Extract LRA Recomendations ===

# 1. Isolate known coordinates to prevent repeating your existing dataset links
known_indices = np.where(scorer_tensor == 1.0)
known_set = set(zip(known_indices[0], known_indices[1], known_indices[2]))

# 2. Sort all 3D matrix locations by their reconstructed score value (descending order)
flattened_indices = np.argsort(lra_tensor.ravel())[::-1]
top_coords, bottom_coords, al_coords = np.unravel_index(flattened_indices, lra_tensor.shape)

# 3. Harvest the top 5 brand new combinations
new_recommendations = []
count = 0
for i in range(len(flattened_indices)):
    if count >= 8:  # Absolute cutoff set to 5
        break
    
    coords = (top_coords[i], bottom_coords[i], al_coords[i])
    score = lra_tensor[coords]
    
    # Skip entries you already knew about
    if coords in known_set:
        continue
        
    new_recommendations.append({
        'Top_Idx': coords[0],
        'Bottom_Idx': coords[1],
        'A_L_Idx': coords[2],
        'Match_Confidence_Score': round(float(score), 4)
    })
    count += 1

# 4. Format into a lightweight dataframe profile 
recommendations_df = pd.DataFrame(new_recommendations)

print("=== TOP 5 HIDDEN LINK RECOMMENDATIONS ===")
print(recommendations_df.to_string(index=False))

# === Translations ===
long_recommendations = pivot_longer(
    recommendations_df,
    id_vars=['Match_Confidence_Score'],
    value_name='Index',
    var_name='Category'
    
)
print(long_recommendations.head(10))


decoded_recommendations = merge_with_identifier(long_recommendations, identifier_df)
             # Keeps long_recommendations completely intact


# Display the first 5 rows to verify the "Clothing Item" column was successfully added
print(decoded_recommendations.head(10))

# returns the code to wide format for easier reading 
wide_df = wider(
    decoded_recommendations,
    index_col='Match_Confidence_Score',
    columns_col='Category',
    values_col='Clothing Item'
)

wide_df = wide_df.sort_values(by='Match_Confidence_Score', ascending=False)  # Sort by confidence score

print(wide_df)  # Display the first 8 rows of the wide format dataframe






