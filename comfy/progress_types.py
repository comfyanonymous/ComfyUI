from abc import ABCMeta, abstractmethod

from typing_extensions import TypedDict, NotRequired


class PreviewImageMetadata(TypedDict, total=True):
    """
    Metadata associated with a preview image sent to the UI.
    """
    node_id: str
    prompt_id: str
    display_node_id: str
    parent_node_id: str
    real_node_id: str
    image_type: NotRequired[str]


class AbstractProgressRegistry(metaclass=ABCMeta):

    @abstractmethod
    def register_handler(self, handler):
        """Register a progress handler"""
        pass

    @abstractmethod
    def unregister_handler(self, handler_name):
        """Unregister a progress handler"""
        pass

    @abstractmethod
    def enable_handler(self, handler_name):
        """Enable a progress handler"""
        pass

    @abstractmethod
    def disable_handler(self, handler_name):
        """Disable a progress handler"""
        pass

    @abstractmethod
    def ensure_entry(self, node_id):
        """Ensure a node entry exists"""
        pass

    @abstractmethod
    def start_progress(self, node_id):
        """Start progress tracking for a node"""
        pass

    @abstractmethod
    def update_progress(self, node_id, value, max_value, image):
        """Update progress for a node"""
        pass

    @abstractmethod
    def finish_progress(self, node_id):
        """Finish progress tracking for a node"""
        pass

    @abstractmethod
    def reset_handlers(self):
        """Reset all handlers"""
        pass


class ProgressRegistryStub(AbstractProgressRegistry):
    """A stub implementation of AbstractProgressRegistry that performs no operations."""

    def register_handler(self, handler):
        """Register a progress handler"""
        pass

    def unregister_handler(self, handler_name):
        """Unregister a progress handler"""
        pass

    def enable_handler(self, handler_name):
        """Enable a progress handler"""
        pass

    def disable_handler(self, handler_name):
        """Disable a progress handler"""
        pass

    def ensure_entry(self, node_id):
        """Ensure a node entry exists"""
        pass

    def start_progress(self, node_id):
        """Start progress tracking for a node"""
        pass

    def update_progress(self, node_id, value, max_value, image):
        """Update progress for a node"""
        pass

    def finish_progress(self, node_id):
        """Finish progress tracking for a node"""
        pass

    def reset_handlers(self):
        """Reset all handlers"""
        pass
