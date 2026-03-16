# export_chair.py
import numpy as np
import trimesh
from superdec.inference import SuperDecInference

infer = SuperDecInference(checkpoint="checkpoints/normalized")
results = infer.run("examples/chair.ply")

# Save each superquadric as an OBJ mesh
for i, sq in enumerate(results["meshes"]):  # trimesh objects
    sq.export(f"/tmp/chair_sq_{i}.obj")
    print(f"Saved chair_sq_{i}.obj")

# Also save the raw params
np.savez("/tmp/chair_superquadrics.npz", **results["params"])
