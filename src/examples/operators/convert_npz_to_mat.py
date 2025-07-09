import os
from pathlib import Path

import numpy as np
from scipy.io import savemat

allpath = [
    Path("src", "examples", "operators", "cylinder", "data_output"),
    Path("src", "examples", "operators", "cavity", "data_output"),
    Path("src", "examples", "operators", "lidcavity", "data_output"),
]
allfiles = ["A.npz", "E.npz", "A_coo.npz", "E_coo.npz"]

for path in allpath:
    for file in allfiles:
        npz_file_path = path / file
        mat_file_path = os.path.splitext(npz_file_path)[0] + ".mat"

        mat = np.load(npz_file_path)

        savemat(mat_file_path, mat)
        print("generated ", mat_file_path, "from", npz_file_path)
