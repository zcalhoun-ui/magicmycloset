import os
import pandas as pd
import numpy as np
import tensorly as tl
from tensorly.decomposition import tucker  
import networkx as nx
import community as community_louvain
import matplotlib.pyplot as plt

tl.set_backend('numpy')

# 1. Get the script directory and build file paths
def get_script_directory():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    return script_dir

# 2. Load the CSV file
script_dir = get_script_directory()
matrix_path = os.path.join(script_dir, "matrix_numerical.csv")
matrix_df = pd.read_csv(matrix_path)

print(matrix_df.head(5))

# will take in feedback from website in future
matrix_df = matrix_df.groupby(['Top', 'Bottom', 'A/L'], as_index=False)['Rating'].mean()
# filter out those 0s (total flop clearly)
train_data = matrix_df[matrix_df['Rating'] != 0].copy()
# remove labels and standardize numbers
for col in ['Top', 'Bottom', 'A/L']:
    train_data[col + '_idx'] = train_data[col].astype('category').cat.codes
    # Keep track of mapping to decode recommendations later
    matrix_df[col + '_idx'] = matrix_df[col].map(dict(zip(train_data[col], train_data[col + '_idx'])))
# Isolate the dimensions for matrix sizing
I1 = train_data['Top_idx'].max() + 1
I2 = train_data['Bottom_idx'].max() + 1
I3 = train_data['A/L_idx'].max() + 1


# Establishes hyperparameters to guide the model
RANK = 2           # Low rank acts as a barrier against overfitting on 270 samples
LEARNING_RATE = 0.00001
LAMBDA_REG = 0.80  # Regularization penalty weight
EPOCHS = 60

# --- 3. CP FACTOR MATRIX INITIALIZATION ---
np.random.seed(29)
# creates a matrix of dimensions T1 (how many unique tops there are) and rank (we set to 3). 
# The model will look for 3 unique categories within tops, and give each top the liklihood of 
# belonging to each of these categories.
A1 = np.random.normal(0, 0.1, (I1, RANK)) # Mode 1: Top
A2 = np.random.normal(0, 0.1, (I2, RANK)) # Mode 2: Bottom
A3 = np.random.normal(0, 0.1, (I3, RANK)) # Mode 3: A/L

# Extract active coordinates and targets for the SGD loop
indices = train_data[['Top_idx', 'Bottom_idx', 'A/L_idx']].values
ratings = train_data['Rating'].values

print(f"Training Magicmycloset across {len(train_data)} explicit outfit ratings...")

for epoch in range(EPOCHS):
    total_loss = 0
    
    # Shuffle coordinates each epoch to optimize SGD pathways
    permutation = np.random.permutation(len(ratings))
    indices = indices[permutation]
    ratings = ratings[permutation]
    
    for idx in range(len(ratings)):
        i, j, k = indices[idx]
        actual_rating = ratings[idx]
        # 3D CP Prediction Rule: Element-wise vector multiplication
        pred_rating = np.sum(A1[i] * A2[j] * A3[k])
        
        # Gradient computation step
        error = actual_rating - pred_rating
        total_loss += error ** 2

        # Calculate Gradients with L2 Regularization
        grad_A1 = -2 * error * (A2[j] * A3[k]) + 2 * LAMBDA_REG * A1[i]
        grad_A2 = -2 * error * (A1[i] * A3[k]) + 2 * LAMBDA_REG * A2[j]
        grad_A3 = -2 * error * (A1[i] * A2[j]) + 2 * LAMBDA_REG * A3[k]

        # Update Latent Factors
        A1[i] -= LEARNING_RATE * grad_A1
        A2[j] -= LEARNING_RATE * grad_A2
        A3[k] -= LEARNING_RATE * grad_A3
     # Apply global regularization loss adjustment
    reg_loss = LAMBDA_REG * (np.sum(A1**2) + np.sum(A2**2) + np.sum(A3**2))
    epoch_loss = total_loss + reg_loss

    if (epoch + 1) % 10 == 0 or epoch == 0:
        print(f"Iteration {epoch+1:02d}/{EPOCHS} | Penalized Error Scale: {epoch_loss:.4f}")
print("\n 3D Magicmycloset best matrices chosen.")

def predict_outfit_score(top_idx, bottom_idx, al_idx):
    """Predicts a general compatibility score for any 3-part outfit combination."""
    return np.sum(A1[top_idx] * A2[bottom_idx] * A3[al_idx])

predicted_score = predict_outfit_score(top_idx=0, bottom_idx=0, al_idx=0)
print(f"\nPredicted compatibility score for Outfit trio index (1010.0, 101.0, 10.0): {predicted_score:.4f}")

import itertools

def extract_top_recommendations(n_recommendations=10):
    """
    Finds all untested outfit combinations, predicts their scores, 
    and returns the top ranked recommendations.
    """
    # 1. Create lookup sets of combinations that WERE already tested (rating != 0)
    # This acts as our "Omega Mask" filter during inference to find what is untested
    tested_combinations = set(
        zip(train_data['Top_idx'], train_data['Bottom_idx'], train_data['A/L_idx'])
    )
    
    # 2. Reconstruct original ID mappings to decode the index numbers back to actual IDs
    # (e.g., mapping index 0 back to Top ID 1006.0)
    top_id_map = dict(zip(train_data['Top_idx'], train_data['Top']))
    bottom_id_map = dict(zip(train_data['Bottom_idx'], train_data['Bottom']))
    al_id_map = dict(zip(train_data['A/L_idx'], train_data['A/L']))
    
    # 3. Generate EVERY mathematically possible outfit combination from your inventory
    all_possible_indices = itertools.product(range(I1), range(I2), range(I3))
    
    recommendations = []
    
    # 4. Loop over all combinations and screen out tested ones
    for i, j, k in all_possible_indices:
        if (i, j, k) in tested_combinations:
            continue  # Skip outfits that already have a +1 or -1 rating
            
        # 5. Compute the CP Prediction score for the untested outfit
        # Math: sum of element-wise multiplication of the 3 latent vectors
        score = np.sum(A1[i] * A2[j] * A3[k])
        
        # Translate matrix index positions back to your original item IDs
        original_top = top_id_map[i]
        original_bottom = bottom_id_map[j]
        original_al = al_id_map[k]
        
        recommendations.append({
            'Top': original_top,
            'Bottom': original_bottom,
            'A/L': original_al,
            'Predicted_Score': score
        })
        
    # 6. Convert to a DataFrame and sort by the highest score descending
    rec_df = pd.DataFrame(recommendations)
    
    if rec_df.empty:
        print("No untested combinations found in current matrix limits!")
        return rec_df
        
    rec_df = rec_df.sort_values(by='Predicted_Score', ascending=False).reset_index(drop=True)
    
    # Return only the number of requested recommendations
    return rec_df.head(n_recommendations)

# --- RUN THE EXTRACTION ---
print("\n--- GENERATING TOP UNTESTED OUTFIT RECOMMENDATIONS ---")
top_outfits = extract_top_recommendations(n_recommendations=8)
print(top_outfits)

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

# === Translations ===
long_recommendations = pivot_longer(
    top_outfits,
    id_vars=['Predicted_Score'],
    var_name='Category',
    value_name='Index'
)
identifier_path = os.path.join(script_dir, "identifier.csv")
identifier_df = pd.read_csv(identifier_path)

decoded_recommendations = merge_with_identifier(long_recommendations, identifier_df)
             # Keeps long_recommendations completely intact



# returns the code to wide format for easier reading 
wide_df = wider(
    decoded_recommendations,
    index_col='Predicted_Score',
    columns_col='Category',
    values_col='Clothing Item'
)

wide_df = wide_df.sort_values(by='Predicted_Score', ascending=False)  # Sort by confidence score

print(wide_df)






