from comfy_extras import nodes_dataset


def _all_node_classes():
    """Every node class registered by the dataset extension."""
    return [
        cls for cls in vars(nodes_dataset).values()
        if isinstance(cls, type) and issubclass(cls, nodes_dataset.io.ComfyNode)
    ]


def test_is_deprecated_propagates_to_schema():
    """A node's class-level is_deprecated flag must reach define_schema()'s
    io.Schema, since /object_info (and the frontend's deprecated-node filter)
    reads the schema, not the class attribute. See PR that fixed nodes like
    ReplaceTextNode reporting deprecated: false despite is_deprecated = True."""
    for node_cls in _all_node_classes():
        if getattr(node_cls, "node_id", "") is None:
            continue  # abstract base class (ImageProcessingNode/TextProcessingNode), not a real node
        if not hasattr(node_cls, "is_deprecated"):
            continue  # node builds its Schema with is_deprecated inline, nothing to compare
        class_flag = bool(node_cls.is_deprecated)
        schema_flag = bool(node_cls.define_schema().is_deprecated)
        assert schema_flag == class_flag, (
            f"{node_cls.__name__}: class is_deprecated={class_flag} but "
            f"define_schema().is_deprecated={schema_flag}"
        )


def test_known_deprecated_nodes_report_deprecated_in_schema():
    deprecated_nodes = [
        nodes_dataset.ResizeImagesByShorterEdgeNode,
        nodes_dataset.ResizeImagesByLongerEdgeNode,
        nodes_dataset.MergeImageListsNode,
        nodes_dataset.TextToLowercaseNode,
        nodes_dataset.TextToUppercaseNode,
        nodes_dataset.AddTextPrefixNode,
        nodes_dataset.AddTextSuffixNode,
        nodes_dataset.ReplaceTextNode,
        nodes_dataset.StripWhitespaceNode,
        nodes_dataset.MergeTextListsNode,
        nodes_dataset.SaveImageDataSetToFolderNode,
    ]
    for node_cls in deprecated_nodes:
        assert node_cls.define_schema().is_deprecated is True, (
            f"{node_cls.__name__} should report deprecated: true in its schema"
        )
