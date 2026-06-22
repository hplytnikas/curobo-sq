# Example data samples

Two real objects in the exact layout the dataloader expects, so you can inspect
the format and smoke-test the pipeline without downloading the full dataset.

```
ShapeNet/
  03642806/818a8f85.../              # ShapeNet laptop (processed)
    pointcloud.npz                   #   surface points (+ normals)
    points.npz                       #   occupancy: points (100k) + packed bits
  shapenet_tabletop/03642806/818a8f85.../models/
    model_normalized.solid.binvox    #   raw source for occupancy preprocessing
  gso/Pokmon_Conquest_Nintendo_DS_Game/   # GSO object (processed)
    pointcloud.npz
    points.npz                       #   occupancy: unpacked uint8 (100k)
```

The two `points.npz` cover both occupancy schemas the dataloader handles:
- **ShapeNet/OccNet** — `occupancies` is bit-packed (length ≈ N/8).
- **GSO** (our `preprocess_gso_occupancy.py`) — `occupancies` is unpacked uint8.

To use them in place of the full dataset, point `--shapenet_root` (or the config
`shapenet.path`) at `data_samples/ShapeNet`.
</content>
