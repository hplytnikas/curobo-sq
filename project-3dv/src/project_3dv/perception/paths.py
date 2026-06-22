"""
paths.py — single source of truth for filesystem locations external to this
package (the SuperDec repo clone, its fine-tuned checkpoints, and the
curobo-sq repo).

Every path here can be overridden with an environment variable; the
`_DEFAULT` constants are only the fallback for this cluster layout. If
project-3dv or curobo-sq is ever cloned to a different machine or path, set
the corresponding env var instead of editing this file or any of the ~20
scripts that used to hardcode these strings directly.

These are functions, not module-level constants, so the environment
variable is re-read on every call — matching the previous call-site
behaviour where scripts like run_on_frame.py set os.environ["SUPERDEC_DIR"]
*after* pipeline.py may already have been imported, but *before* the
pipeline functions that consume it actually run. A frozen constant
evaluated once at import time would silently ignore that late override.

Environment variables
----------------------
    SUPERDEC_DIR          SuperDec repo clone (contains superdec/, checkpoints/)
                          — name matches what run_on_frame.py already sets.
    SUPERDEC_CKPT_DIR     Fine-tuned tabletop checkpoint directory
                          — name matches what run_on_frame.py already sets.
    PROJECT_3DV_DATA_DIR  project-3dv/data — datasets (OCID, RGB-D Scenes v2, ...)
    CUROBO_SQ_DIR         curobo-sq repo clone (for curobo/src, curobo/examples)
"""

import os

SUPERDEC_DIR_DEFAULT = "/work/courses/3dv/team15/superdec"
SUPERDEC_CHECKPOINT_DIR_DEFAULT = (
    "/work/courses/3dv/team15/checkpoints/"
    "superdec_tabletop/superdec_tabletop_finetune_v2"
)
PROJECT_3DV_DATA_DIR_DEFAULT = "/work/courses/3dv/team15/project-3dv/data"
CUROBO_SQ_DIR_DEFAULT = "/work/courses/3dv/team15/curobo-sq"


def superdec_dir() -> str:
    return os.environ.get("SUPERDEC_DIR", SUPERDEC_DIR_DEFAULT)


def superdec_checkpoint_dir() -> str:
    return os.environ.get("SUPERDEC_CKPT_DIR", SUPERDEC_CHECKPOINT_DIR_DEFAULT)


def project_3dv_data_dir() -> str:
    return os.environ.get("PROJECT_3DV_DATA_DIR", PROJECT_3DV_DATA_DIR_DEFAULT)


def curobo_sq_dir() -> str:
    return os.environ.get("CUROBO_SQ_DIR", CUROBO_SQ_DIR_DEFAULT)
