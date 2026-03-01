# Local Helpers

This folder contains one-off and reusable maintenance scripts for this fork.

Most scripts are local utilities created while cleaning workflows, checking environment state, or patching model and node references. Several of them contain hardcoded paths and should be reviewed before reuse on another machine.

Current categories:

- Workflow cleanup and translation:
  `clean_workflow.py`, `deep_clean_workflow.py`, `radical_clean.py`, `radical_clean_v2.py`, `sanitize_workflow.py`, `translate_workflow.py`
- Workflow inspection:
  `inspect_nodes.py`, `find_referencing_node.py`, `verify_clean.py`, `remove_group_instances.py`, `remove_videopreview.py`
- Environment and dependency checks:
  `check_libs.py`, `check_orbit_models.py`, `test_import_v2.py`, `test_mediapipe.py`, `switch_attention_mode.py`
- Model and workflow patch helpers:
  `fix_model_names.py`, `fix_triton_lib.py`, `fix_user_download_workflow.py`

These scripts are not part of upstream ComfyUI. They are local fork utilities for workflow experimentation and maintenance.
