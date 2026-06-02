import pickle
import numpy as np
import os

def compare_pkls(file1, file2):
    with open(file1, "rb") as f:
        data1 = pickle.load(f)
    with open(file2, "rb") as f:
        data2 = pickle.load(f)
    
    if np.array_equal(data1, data2):
        print(f"FILES ARE IDENTICAL: {file1} == {file2}")
    else:
        diff_count = np.sum(data1 != data2)
        print(f"FILES ARE DIFFERENT: {file1} != {file2}")
        print(f"Total elements: {data1.size}")
        print(f"Different elements: {diff_count}")
        
    # Check if argmax is identical (policy level)
    policy1 = np.argmax(data1, axis=-1)
    policy2 = np.argmax(data2, axis=-1)
    
    # Specific check for the slice visualized in readpkl.py (n_aoa // 2)
    n_aoa = data1.shape[0]
    aoa_idx = n_aoa // 2
    print(f"Visualized AoA slice index: {aoa_idx}")
    
    slice1 = policy1[aoa_idx]
    slice2 = policy2[aoa_idx]
    
    if np.array_equal(slice1, slice2):
        print(f"THE VISUALIZED SLICE (AoA index {aoa_idx}) IS COMPLETELY IDENTICAL.")
    else:
        diff_slice = np.sum(slice1 != slice2)
        print(f"THE VISUALIZED SLICE HAS {diff_slice} DIFFERENCES.")

    if np.array_equal(policy1, policy2):
        print("THE FULL DERIVED POLICIES (argmax) ARE IDENTICAL.")
    else:
        diff_policy = np.sum(policy1 != policy2)
        print(f"FULL POLICIES ARE DIFFERENT at {diff_policy} states.")

if __name__ == "__main__":
    p2500 = r"q_table\q_table_E_2500.pkl"
    p5000 = r"q_table\q_table_E_5000.pkl"
    compare_pkls(p2500, p5000)
