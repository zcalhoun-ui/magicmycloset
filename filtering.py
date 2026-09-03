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
            file = "Fall"
        return file

def get_script_directory():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    return script_dir

def get_id_directory():
    script_did = get_script_directory()
    identifier_path = os.path.join(script_did, "identifier.csv")
    return identifier_path

def get_matrix_directory():
    script_did = get_script_directory()
    matrix_path = os.path.join(script_did, "matrix_numerical.csv")
    return matrix_path

def load_id_csv(season):
    identifier_df = pd.read_csv(get_id_directory())
    identifier_df.columns = identifier_df.columns.str.strip()
    identifier_df = identifier_df[identifier_df[season] != 0]
    identifier_df["Matrix ID"] = pd.to_numeric(identifier_df["Matrix ID"], errors='coerce')
    return identifier_df

def load_mat_csv():
    matrix_df = pd.read_csv(get_matrix_directory())
    converted_matrix_df = matrix_df[matrix_df['Rating'] != -1]
    return converted_matrix_df


def filter_matrix(season):
    matrix = load_mat_csv()
    id = load_id_csv(season)
    matrix = matrix[['Top', 'Bottom', 'A/L']].astype(str).apply(lambda x: x.str.strip())
    keep_list = id['Matrix ID'].astype(str).str.strip().to_list()
    filter_matrix = matrix[matrix[['Top', 'Bottom', 'A/L']].isin(keep_list).all(axis=1)]
   
    return filter_matrix

def get_tensor(season):
    converted_matrix_df = filter_matrix(season)
    converted_matrix_df[['Top', 'Bottom', 'A/L']] = (
        converted_matrix_df[['Top', 'Bottom', 'A/L']]
        .astype(float)
        .astype(int)
    )
    shape = tuple(converted_matrix_df[['Top', 'Bottom', 'A/L']].max() + 1)
    scorer_tensor = np.zeros(shape, dtype=float)
    scorer_tensor[
    converted_matrix_df['Top'].values, 
    converted_matrix_df['Bottom'].values, 
    converted_matrix_df['A/L'].values] = 1.0
    return scorer_tensor

def get_lra(season):
    tl.set_backend('numpy')
    original_tensor = get_tensor(season)
    scorer_tensor = get_tensor(season)
    dim1, dim2, dim3 = original_tensor.shape
    print(f"Detected Data Dimensions: {dim1} x {dim2} x {dim3}")
    custom_ranks = [12, 15, 20]
    core, factors = tucker(scorer_tensor, rank=custom_ranks, init='svd')
    lra_tensor = tl.tucker_to_tensor((core, factors))
    print(f"Reconstructed tensor shape: {lra_tensor.shape}") 
    print(f"Compressed core matrix shape: {core.shape}")
    print(f"Low-Rank Approximation completed successfully!")
    return lra_tensor

def get_recommends(season):
    scorer_tensor = get_tensor(season)
    lra_tensor = get_lra(season)
    known_indices = np.where(scorer_tensor == 1.0)
    known_set = set(zip(known_indices[0], known_indices[1], known_indices[2]))
    print(f"Number of known indices: {len(known_set)}")
    flattened_indices = np.argsort(lra_tensor.ravel())[::-1]
    top_coords, bottom_coords, al_coords = np.unravel_index(flattened_indices, lra_tensor.shape)
    new_recommendations = []
    count = 0
    for i in range(len(flattened_indices)):
        if count >= 6:  # Absolute cutoff 
            break
        
        coords = (top_coords[i], bottom_coords[i], al_coords[i])
        score = lra_tensor[coords]
        
        if coords in known_set:
            continue
            
        new_recommendations.append({
            'Top_Idx': coords[0],
            'Bottom_Idx': coords[1],
            'A_L_Idx': coords[2],
            'Match_Confidence_Score': round(float(score), 4)
        })
        count += 1
        recommendations_df = pd.DataFrame(new_recommendations)
    print("I found the recommendations")
    return recommendations_df


# Perform a left join to keep all rows and order from long_recommendations
def merge_with_identifier(long_recommendations, identifier_df):
    return pd.merge(
        long_recommendations,
        identifier_df,
        left_on='Index',        # The column name in long_recommendations
        right_on='Matrix ID',   # The matching column name in identifiers_df
        how='left'              # Keeps long_recommendations completely intact
    )

def translate(season):
    recommendations_df = get_recommends(season)
    identifier_df = pd.read_csv(get_id_directory())
    recommendations_df2 = recommendations_df
    recommendations_df2['unique_id'] = range(len(recommendations_df2))
    long_recommendations = recommendations_df2.melt(id_vars=['unique_id', 'Match_Confidence_Score'], value_vars = ['Top_Idx', 'Bottom_Idx', 'A_L_Idx'], var_name = 'Category', value_name='Index'  )
    decoded_recommendations = merge_with_identifier(long_recommendations, identifier_df)
    wide_df = decoded_recommendations.pivot_table(
        index=['unique_id', 'Match_Confidence_Score'],
        columns='Category',
        values='Clothing Item',
        aggfunc='first'
    ).reset_index()
    wide_df = wide_df.sort_values(by='Match_Confidence_Score', ascending=False)
    clean_df = wide_df.drop(columns='unique_id')
    return clean_df
        

def main():
    print("First test")
    season = get_season()
    print(season)
    print("I allocated the tensor")
    lra_new = translate(season)
    print(f"The new recommendations for {season} based on your existing outfits are:\n{lra_new}")
    
if __name__ == "__main__":
  main()