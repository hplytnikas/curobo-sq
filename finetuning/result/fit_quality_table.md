# SuperDec fit-quality: before vs after fine-tuning

ShapeNet test split (tabletop categories), Chamfer measured in the original
object frame. Clean = full point cloud input. Occluded = partial input
(seeded occlusions), scored against the full unoccluded shape.
Scripts: scripts/eval_fit_quality_cluster.sh, scripts/eval_fit_quality_partial_cluster.sh

| Model | Checkpoint | Clean Chamfer-L1 | Clean Chamfer-L2 | Occluded Chamfer-L1 | Occluded Chamfer-L2 | Occluded F-score@1% | avg #prims |
|---|---|---|---|---|---|---|---|
| Baseline (before) | normalized/ckpt.pt | **0.01861** | **0.00065** | 0.02955 | 0.00356 | 0.266 | 4.07 |
| Fine-tune (chamfer) | expocc_tt_chamfer/epoch_500 | 0.01934 | 0.00076 | **0.02331** | **0.00149** | **0.291** | 4.80 |
| Fine-tune (bent) | expocc_tt_bent/epoch_100 | 0.01936 | 0.00077 | 0.02394 | 0.00188 | 0.277 | **2.66** |

Lower is better for Chamfer; higher for F-score. **Bold** = best per column.

Notes: bent is epoch_100 vs chamfer epoch_500 (not like-for-like). Occlusion
sampler is mild (RandomOcclusion/HRPOcclusion each p=0.25).
