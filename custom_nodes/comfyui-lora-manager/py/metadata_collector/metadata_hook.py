import sys
import inspect
from .metadata_registry import MetadataRegistry

class MetadataHook:
    """Install hooks for metadata collection"""
    
    @staticmethod
    def install():
        """Install hooks to collect metadata during execution"""
        try:
            # Import ComfyUI's execution module
            execution = None
            try:
                # Try direct import first
                import execution # type: ignore
            except ImportError:
                # Try to locate from system modules
                for module_name in sys.modules:
                    if module_name.endswith('.execution'):
                        execution = sys.modules[module_name]
                        break
                    
            # If we can't find the execution module, we can't install hooks
            if execution is None:
                print("Could not locate ComfyUI execution module, metadata collection disabled")
                return
            
            # Detect whether we're using the new async version of ComfyUI
            is_async = False
            map_node_func_name = '_map_node_over_list'
            
            if hasattr(execution, '_async_map_node_over_list'):
                is_async = inspect.iscoroutinefunction(execution._async_map_node_over_list)
                map_node_func_name = '_async_map_node_over_list'
            elif hasattr(execution, '_map_node_over_list'):
                is_async = inspect.iscoroutinefunction(execution._map_node_over_list)
            
            if is_async:
                print("Detected async ComfyUI execution, installing async metadata hooks")
                MetadataHook._install_async_hooks(execution, map_node_func_name)
            else:
                print("Detected sync ComfyUI execution, installing sync metadata hooks")
                MetadataHook._install_sync_hooks(execution)
            
            print("Metadata collection hooks installed for runtime values")
            
        except Exception as e:
            print(f"Error installing metadata hooks: {str(e)}")
    
    @staticmethod
    def _install_sync_hooks(execution):
        """Install hooks for synchronous execution model"""
        # Store the original _map_node_over_list function
        original_map_node_over_list = execution._map_node_over_list
        
        # Define the wrapped _map_node_over_list function
        def map_node_over_list_with_metadata(obj, input_data_all, func, allow_interrupt=False, execution_block_cb=None, pre_execute_cb=None):
            # Only collect metadata when calling the main function of nodes
            if func == obj.FUNCTION and hasattr(obj, '__class__'):
                try:
                    # Get the current prompt_id from the registry
                    registry = MetadataRegistry()
                    prompt_id = registry.current_prompt_id
                    
                    if prompt_id is not None:
                        # Get node class type
                        class_type = obj.__class__.__name__
                        
                        # Unique ID might be available through the obj if it has a unique_id field
                        node_id = getattr(obj, 'unique_id', None)
                        if node_id is None and pre_execute_cb:
                            # Try to extract node_id through reflection on GraphBuilder.set_default_prefix
                            frame = inspect.currentframe()
                            while frame:
                                if 'unique_id' in frame.f_locals:
                                    node_id = frame.f_locals['unique_id']
                                    break
                                frame = frame.f_back
                        
                        # Record inputs before execution
                        if node_id is not None:
                            registry.record_node_execution(node_id, class_type, input_data_all, None)
                except Exception as e:
                    print(f"Error collecting metadata (pre-execution): {str(e)}")
            
            # Execute the original function
            results = original_map_node_over_list(obj, input_data_all, func, allow_interrupt, execution_block_cb, pre_execute_cb)
            
            # After execution, collect outputs for relevant nodes
            if func == obj.FUNCTION and hasattr(obj, '__class__'):
                try:
                    # Get the current prompt_id from the registry
                    registry = MetadataRegistry()
                    prompt_id = registry.current_prompt_id
                    
                    if prompt_id is not None:
                        # Get node class type
                        class_type = obj.__class__.__name__
                        
                        # Unique ID might be available through the obj if it has a unique_id field
                        node_id = getattr(obj, 'unique_id', None)
                        if node_id is None and pre_execute_cb:
                            # Try to extract node_id through reflection
                            frame = inspect.currentframe()
                            while frame:
                                if 'unique_id' in frame.f_locals:
                                    node_id = frame.f_locals['unique_id']
                                    break
                                frame = frame.f_back
                        
                        # Record outputs after execution
                        if node_id is not None:
                            registry.update_node_execution(node_id, class_type, results)
                except Exception as e:
                    print(f"Error collecting metadata (post-execution): {str(e)}")
            
            return results
            
        # Also hook the execute function to track the current prompt_id
        original_execute = execution.execute
        
        def execute_with_prompt_tracking(*args, **kwargs):
            if len(args) >= 7:  # Check if we have enough arguments
                server, prompt, caches, node_id, extra_data, executed, prompt_id = args[:7]
                registry = MetadataRegistry()
                
                # Start collection if this is a new prompt
                if not registry.current_prompt_id or registry.current_prompt_id != prompt_id:
                    registry.start_collection(prompt_id)
                    
                # Store the dynprompt reference for node lookups
                if hasattr(prompt, 'original_prompt'):
                    registry.set_current_prompt(prompt)
            
            # Execute the original function
            return original_execute(*args, **kwargs)
            
        # Replace the functions
        execution._map_node_over_list = map_node_over_list_with_metadata
        execution.execute = execute_with_prompt_tracking
    
    @staticmethod
    def _install_async_hooks(execution, map_node_func_name='_async_map_node_over_list'):
        """Install hooks for asynchronous execution model"""
        # Store the original _async_map_node_over_list function
        original_map_node_over_list = getattr(execution, map_node_func_name)
        
        # Wrapped async function, compatible with both stable and nightly
        async def async_map_node_over_list_with_metadata(prompt_id, unique_id, obj, input_data_all, func, allow_interrupt=False, execution_block_cb=None, pre_execute_cb=None, *args, **kwargs):
            hidden_inputs = kwargs.get('hidden_inputs', None)
            # Only collect metadata when calling the main function of nodes
            if func == obj.FUNCTION and hasattr(obj, '__class__'):
                try:
                    registry = MetadataRegistry()
                    if prompt_id is not None:
                        class_type = obj.__class__.__name__
                        node_id = unique_id
                        if node_id is not None:
                            registry.record_node_execution(node_id, class_type, input_data_all, None)
                except Exception as e:
                    print(f"Error collecting metadata (pre-execution): {str(e)}")
            
            # Call original function with all args/kwargs
            results = await original_map_node_over_list(
                prompt_id, unique_id, obj, input_data_all, func,
                allow_interrupt, execution_block_cb, pre_execute_cb, *args, **kwargs
            )
            
            if func == obj.FUNCTION and hasattr(obj, '__class__'):
                try:
                    registry = MetadataRegistry()
                    if prompt_id is not None:
                        class_type = obj.__class__.__name__
                        node_id = unique_id
                        if node_id is not None:
                            registry.update_node_execution(node_id, class_type, results)
                except Exception as e:
                    print(f"Error collecting metadata (post-execution): {str(e)}")
            
            return results
        
        # Also hook the execute function to track the current prompt_id
        original_execute = execution.execute
        
        async def async_execute_with_prompt_tracking(*args, **kwargs):
            if len(args) >= 7:  # Check if we have enough arguments
                server, prompt, caches, node_id, extra_data, executed, prompt_id = args[:7]
                registry = MetadataRegistry()
                
                # Start collection if this is a new prompt
                if not registry.current_prompt_id or registry.current_prompt_id != prompt_id:
                    registry.start_collection(prompt_id)
                    
                # Store the dynprompt reference for node lookups
                if hasattr(prompt, 'original_prompt'):
                    registry.set_current_prompt(prompt)
            
            # Execute the original function
            return await original_execute(*args, **kwargs)
            
        # Replace the functions with async versions
        setattr(execution, map_node_func_name, async_map_node_over_list_with_metadata)
        execution.execute = async_execute_with_prompt_tracking
