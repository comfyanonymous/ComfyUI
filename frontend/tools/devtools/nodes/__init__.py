from __future__ import annotations

from .errors import (
    DeprecatedNode,
    ErrorRaiseNode,
    ErrorRaiseNodeWithMessage,
    ExperimentalNode,
    NODE_CLASS_MAPPINGS as errors_class_mappings,
    NODE_DISPLAY_NAME_MAPPINGS as errors_display_name_mappings,
)
from .inputs import (
    LongComboDropdown,
    NodeWithBooleanInput,
    NodeWithDefaultInput,
    NodeWithForceInput,
    NodeWithOptionalComboInput,
    NodeWithOptionalInput,
    NodeWithOnlyOptionalInput,
    NodeWithOutputList,
    NodeWithSeedInput,
    NodeWithStringInput,
    NodeWithUnionInput,
    NodeWithValidation,
    NodeWithV2ComboInput,
    SimpleSlider,
    NODE_CLASS_MAPPINGS as inputs_class_mappings,
    NODE_DISPLAY_NAME_MAPPINGS as inputs_display_name_mappings,
)
from .models import (
    DummyPatch,
    LoadAnimatedImageTest,
    ObjectPatchNode,
    NODE_CLASS_MAPPINGS as models_class_mappings,
    NODE_DISPLAY_NAME_MAPPINGS as models_display_name_mappings,
)
from .remote import (
    MultiSelectNode,
    NodeWithOutputCombo,
    RemoteWidgetNode,
    RemoteWidgetNodeWithControlAfterRefresh,
    RemoteWidgetNodeWithParams,
    RemoteWidgetNodeWithRefresh,
    RemoteWidgetNodeWithRefreshButton,
    NODE_CLASS_MAPPINGS as remote_class_mappings,
    NODE_DISPLAY_NAME_MAPPINGS as remote_display_name_mappings,
)

NODE_CLASS_MAPPINGS = {
    **errors_class_mappings,
    **inputs_class_mappings,
    **remote_class_mappings,
    **models_class_mappings,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    **errors_display_name_mappings,
    **inputs_display_name_mappings,
    **remote_display_name_mappings,
    **models_display_name_mappings,
}

__all__ = [
    "DeprecatedNode",
    "DummyPatch",
    "ErrorRaiseNode",
    "ErrorRaiseNodeWithMessage",
    "ExperimentalNode",
    "LoadAnimatedImageTest",
    "LongComboDropdown",
    "MultiSelectNode",
    "NodeWithBooleanInput",
    "NodeWithDefaultInput",
    "NodeWithForceInput",
    "NodeWithOptionalComboInput",
    "NodeWithOptionalInput",
    "NodeWithOnlyOptionalInput",
    "NodeWithOutputCombo",
    "NodeWithOutputList",
    "NodeWithSeedInput",
    "NodeWithStringInput",
    "NodeWithUnionInput",
    "NodeWithValidation",
    "NodeWithV2ComboInput",
    "ObjectPatchNode",
    "RemoteWidgetNode",
    "RemoteWidgetNodeWithControlAfterRefresh",
    "RemoteWidgetNodeWithParams",
    "RemoteWidgetNodeWithRefresh",
    "RemoteWidgetNodeWithRefreshButton",
    "SimpleSlider",
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
]
