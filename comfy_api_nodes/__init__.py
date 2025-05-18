# Filter out warnings about model_name fields and protected namespaces in Pydantic models
#
# Problem: Pydantic <2.10 raises warnings when models contain fields named "model_name"
# due to its protected namespace feature. Several Kling API models use "model_name" fields
# (which indicate which Kling model to use in API requests) triggering these warnings:
#
# UserWarning: Field "model_name" in KlingSingleImageEffectInput has conflict with
# protected namespace "model_".
#
# Affected versions:
# - Pydantic <2.10: Uses ('model_',) as default protected namespace
# - Pydantic ≥2.10: Uses ('model_validate', 'model_dump',) as default (no warnings)
#
# Why not fix this with model_config in each model class?
# - These models are auto-generated from OpenAPI specs, so modifications would be lost
# - Pydantic BaseModel does not actually have a 'model_name' method, it's just a
#   preventative warning to avoid potential future conflicts
#
# For more details, see: https://github.com/comfyanonymous/ComfyUI/issues/8172#issuecomment-2888641795

import warnings
warnings.filterwarnings("ignore", message=".*model_name.*protected namespace.*")
