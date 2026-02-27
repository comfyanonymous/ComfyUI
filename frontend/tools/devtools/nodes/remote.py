from __future__ import annotations


class RemoteWidgetNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "remote_widget_value": (
                    "COMBO",
                    {
                        "remote": {
                            "route": "/api/models/checkpoints",
                        },
                    },
                ),
            },
        }

    FUNCTION = "remote_widget"
    CATEGORY = "DevTools"
    DESCRIPTION = "A node that lazily fetches options from a remote endpoint"
    RETURN_TYPES = ("STRING",)

    def remote_widget(self, remote_widget_value: str):
        return (remote_widget_value,)


class RemoteWidgetNodeWithParams:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "remote_widget_value": (
                    "COMBO",
                    {
                        "remote": {
                            "route": "/api/models/checkpoints",
                            "query_params": {
                                "sort": "true",
                            },
                        },
                    },
                ),
            },
        }

    FUNCTION = "remote_widget"
    CATEGORY = "DevTools"
    DESCRIPTION = (
        "A node that lazily fetches options from a remote endpoint with query params"
    )
    RETURN_TYPES = ("STRING",)

    def remote_widget(self, remote_widget_value: str):
        return (remote_widget_value,)


class RemoteWidgetNodeWithRefresh:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "remote_widget_value": (
                    "COMBO",
                    {
                        "remote": {
                            "route": "/api/models/checkpoints",
                            "refresh": 300,
                            "max_retries": 10,
                            "timeout": 256,
                        },
                    },
                ),
            },
        }

    FUNCTION = "remote_widget"
    CATEGORY = "DevTools"
    DESCRIPTION = "A node that lazily fetches options from a remote endpoint and refresh the options every 300 ms"
    RETURN_TYPES = ("STRING",)

    def remote_widget(self, remote_widget_value: str):
        return (remote_widget_value,)


class RemoteWidgetNodeWithRefreshButton:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "remote_widget_value": (
                    "COMBO",
                    {
                        "remote": {
                            "route": "/api/models/checkpoints",
                            "refresh_button": True,
                        },
                    },
                ),
            },
        }

    FUNCTION = "remote_widget"
    CATEGORY = "DevTools"
    DESCRIPTION = "A node that lazily fetches options from a remote endpoint and has a refresh button to manually reload options"
    RETURN_TYPES = ("STRING",)

    def remote_widget(self, remote_widget_value: str):
        return (remote_widget_value,)


class RemoteWidgetNodeWithControlAfterRefresh:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "remote_widget_value": (
                    "COMBO",
                    {
                        "remote": {
                            "route": "/api/models/checkpoints",
                            "refresh_button": True,
                            "control_after_refresh": "first",
                        },
                    },
                ),
            },
        }

    FUNCTION = "remote_widget"
    CATEGORY = "DevTools"
    DESCRIPTION = "A node that lazily fetches options from a remote endpoint and has a refresh button to manually reload options and select the first option on refresh"
    RETURN_TYPES = ("STRING",)

    def remote_widget(self, remote_widget_value: str):
        return (remote_widget_value,)


class NodeWithOutputCombo:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "subset_options": (["A", "B"], {"forceInput": True}),
                "subset_options_v2": (
                    "COMBO",
                    {"options": ["A", "B"], "forceInput": True},
                ),
            }
        }

    RETURN_TYPES = (["A", "B", "C"],)
    FUNCTION = "node_with_output_combo"
    CATEGORY = "DevTools"
    DESCRIPTION = "A node that outputs a combo type"

    def node_with_output_combo(self, subset_options: str):
        return (subset_options,)


class MultiSelectNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "foo": (
                    "COMBO",
                    {
                        "options": ["A", "B", "C"],
                        "multi_select": {
                            "placeholder": "Choose foos",
                            "chip": True,
                        },
                    },
                )
            }
        }

    RETURN_TYPES = ("STRING",)
    OUTPUT_IS_LIST = [True]
    FUNCTION = "multi_select_node"
    CATEGORY = "DevTools"
    DESCRIPTION = "A node that outputs a multi select type"

    def multi_select_node(self, foo: list[str]) -> list[str]:
        return (foo,)


NODE_CLASS_MAPPINGS = {
    "DevToolsRemoteWidgetNode": RemoteWidgetNode,
    "DevToolsRemoteWidgetNodeWithParams": RemoteWidgetNodeWithParams,
    "DevToolsRemoteWidgetNodeWithRefresh": RemoteWidgetNodeWithRefresh,
    "DevToolsRemoteWidgetNodeWithRefreshButton": RemoteWidgetNodeWithRefreshButton,
    "DevToolsRemoteWidgetNodeWithControlAfterRefresh": RemoteWidgetNodeWithControlAfterRefresh,
    "DevToolsNodeWithOutputCombo": NodeWithOutputCombo,
    "DevToolsMultiSelectNode": MultiSelectNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DevToolsRemoteWidgetNode": "Remote Widget Node",
    "DevToolsRemoteWidgetNodeWithParams": "Remote Widget Node With Sort Query Param",
    "DevToolsRemoteWidgetNodeWithRefresh": "Remote Widget Node With 300ms Refresh",
    "DevToolsRemoteWidgetNodeWithRefreshButton": "Remote Widget Node With Refresh Button",
    "DevToolsRemoteWidgetNodeWithControlAfterRefresh": "Remote Widget Node With Refresh Button and Control After Refresh",
    "DevToolsNodeWithOutputCombo": "Node With Output Combo",
    "DevToolsMultiSelectNode": "Multi Select Node",
}

__all__ = [
    "RemoteWidgetNode",
    "RemoteWidgetNodeWithParams",
    "RemoteWidgetNodeWithRefresh",
    "RemoteWidgetNodeWithRefreshButton",
    "RemoteWidgetNodeWithControlAfterRefresh",
    "NodeWithOutputCombo",
    "MultiSelectNode",
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
]
