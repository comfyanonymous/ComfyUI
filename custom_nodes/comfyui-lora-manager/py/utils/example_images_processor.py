import logging
import os
import re
import tempfile
import random
import string
from aiohttp import web
from ..utils.constants import SUPPORTED_MEDIA_EXTENSIONS
from ..services.service_registry import ServiceRegistry
from ..services.settings_manager import settings
from .example_images_metadata import MetadataUpdater
from ..utils.metadata_manager import MetadataManager

logger = logging.getLogger(__name__)

class ExampleImagesProcessor:
    """Processes and manipulates example images"""

    @staticmethod
    def generate_short_id(length=8):
        """Generate a short random alphanumeric identifier"""
        chars = string.ascii_lowercase + string.digits
        return ''.join(random.choice(chars) for _ in range(length))
    
    @staticmethod
    def get_civitai_optimized_url(image_url):
        """Convert Civitai image URL to its optimized WebP version"""
        base_pattern = r'(https://image\.civitai\.com/[^/]+/[^/]+)'
        match = re.match(base_pattern, image_url)
        
        if match:
            base_url = match.group(1)
            return f"{base_url}/optimized=true/image.webp"
        
        return image_url
    
    @staticmethod
    async def download_model_images(model_hash, model_name, model_images, model_dir, optimize, independent_session):
        """Download images for a single model
        
        Returns:
            tuple: (success, is_stale_metadata) - whether download was successful, whether metadata is stale
        """
        model_success = True
        
        for i, image in enumerate(model_images):
            image_url = image.get('url')
            if not image_url:
                continue
            
            # Get image filename from URL
            image_filename = os.path.basename(image_url.split('?')[0])
            image_ext = os.path.splitext(image_filename)[1].lower()
            
            # Handle images and videos
            is_image = image_ext in SUPPORTED_MEDIA_EXTENSIONS['images']
            is_video = image_ext in SUPPORTED_MEDIA_EXTENSIONS['videos']
            
            if not (is_image or is_video):
                logger.debug(f"Skipping unsupported file type: {image_filename}")
                continue
            
            # Use 0-based indexing instead of 1-based indexing
            save_filename = f"image_{i}{image_ext}"
            
            # If optimizing images and this is a Civitai image, use their pre-optimized WebP version
            if is_image and optimize and 'civitai.com' in image_url:
                image_url = ExampleImagesProcessor.get_civitai_optimized_url(image_url)
                save_filename = f"image_{i}.webp"
            
            # Check if already downloaded
            save_path = os.path.join(model_dir, save_filename)
            if os.path.exists(save_path):
                logger.debug(f"File already exists: {save_path}")
                continue
            
            # Download the file
            try:
                logger.debug(f"Downloading {save_filename} for {model_name}")
                
                # Download directly using the independent session
                async with independent_session.get(image_url, timeout=60) as response:
                    if response.status == 200:
                        with open(save_path, 'wb') as f:
                            async for chunk in response.content.iter_chunked(8192):
                                if chunk:
                                    f.write(chunk)
                    elif response.status == 404:
                        error_msg = f"Failed to download file: {image_url}, status code: 404 - Model metadata might be stale"
                        logger.warning(error_msg)
                        model_success = False  # Mark the model as failed due to 404 error
                        # Return early to trigger metadata refresh attempt
                        return False, True  # (success, is_metadata_stale)
                    else:
                        error_msg = f"Failed to download file: {image_url}, status code: {response.status}"
                        logger.warning(error_msg)
                        model_success = False  # Mark the model as failed
            except Exception as e:
                error_msg = f"Error downloading file {image_url}: {str(e)}"
                logger.error(error_msg)
                model_success = False  # Mark the model as failed
        
        return model_success, False  # (success, is_metadata_stale)
    
    @staticmethod
    async def process_local_examples(model_file_path, model_file_name, model_name, model_dir, optimize):
        """Process local example images
        
        Returns:
            bool: True if local images were processed successfully, False otherwise
        """
        try:
            if not model_file_path or not os.path.exists(os.path.dirname(model_file_path)):
                return False
                
            model_dir_path = os.path.dirname(model_file_path)
            local_images = []
            
            # Look for files with pattern: filename.example.*.ext
            if model_file_name:
                example_prefix = f"{model_file_name}.example."
                
                if os.path.exists(model_dir_path):
                    for file in os.listdir(model_dir_path):
                        file_lower = file.lower()
                        if file_lower.startswith(example_prefix.lower()):
                            file_ext = os.path.splitext(file_lower)[1]
                            is_supported = (file_ext in SUPPORTED_MEDIA_EXTENSIONS['images'] or 
                                           file_ext in SUPPORTED_MEDIA_EXTENSIONS['videos'])
                            
                            if is_supported:
                                local_images.append(os.path.join(model_dir_path, file))
            
            # Process local images if found
            if local_images:
                logger.info(f"Found {len(local_images)} local example images for {model_name}")
                
                for local_image_path in local_images:
                    # Extract index from filename
                    file_name = os.path.basename(local_image_path)
                    example_prefix = f"{model_file_name}.example."
                    
                    try:
                        # Extract the part between '.example.' and the file extension
                        index_part = file_name[len(example_prefix):].split('.')[0]
                        # Try to parse it as an integer
                        index = int(index_part)
                        local_ext = os.path.splitext(local_image_path)[1].lower()
                        save_filename = f"image_{index}{local_ext}"
                    except (ValueError, IndexError):
                        # If we can't parse the index, fall back to sequential numbering
                        logger.warning(f"Could not extract index from {file_name}, using sequential numbering")
                        local_ext = os.path.splitext(local_image_path)[1].lower()
                        save_filename = f"image_{len(local_images)}{local_ext}"
                    
                    save_path = os.path.join(model_dir, save_filename)
                    
                    # Skip if already exists in output directory
                    if os.path.exists(save_path):
                        logger.debug(f"File already exists in output: {save_path}")
                        continue
                    
                    # Copy the file
                    with open(local_image_path, 'rb') as src_file:
                        with open(save_path, 'wb') as dst_file:
                            dst_file.write(src_file.read())
                
                return True
            return False
        except Exception as e:
            logger.error(f"Error processing local examples for {model_name}: {str(e)}")
            return False
            
    @staticmethod
    async def import_images(request):
        """
        Import local example images
        
        Accepts:
        - multipart/form-data form with model_hash and files fields
        or
        - JSON request with model_hash and file_paths
        
        Returns:
        - Success status and list of imported files
        """
        try:
            model_hash = None
            files_to_import = []
            temp_files_to_cleanup = []
            
            # Check if it's a multipart form-data request (direct file upload)
            if request.content_type and 'multipart/form-data' in request.content_type:
                reader = await request.multipart()
                
                # First get model_hash
                field = await reader.next()
                if field.name == 'model_hash':
                    model_hash = await field.text()
                
                # Then process all files
                while True:
                    field = await reader.next()
                    if field is None:
                        break
                        
                    if field.name == 'files':
                        # Create a temporary file with appropriate suffix for type detection
                        file_name = field.filename
                        file_ext = os.path.splitext(file_name)[1].lower()
                        
                        with tempfile.NamedTemporaryFile(suffix=file_ext, delete=False) as tmp_file:
                            temp_path = tmp_file.name
                            temp_files_to_cleanup.append(temp_path)  # Track for cleanup
                            
                            # Write chunks to the temporary file
                            while True:
                                chunk = await field.read_chunk()
                                if not chunk:
                                    break
                                tmp_file.write(chunk)
                        
                        # Add to the list of files to process
                        files_to_import.append(temp_path)
            else:
                # Parse JSON request (legacy method using file paths)
                data = await request.json()
                model_hash = data.get('model_hash')
                files_to_import = data.get('file_paths', [])
            
            if not model_hash:
                return web.json_response({
                    'success': False,
                    'error': 'Missing model_hash parameter'
                }, status=400)
            
            if not files_to_import:
                return web.json_response({
                    'success': False,
                    'error': 'No files provided to import'
                }, status=400)
            
            # Get example images path
            example_images_path = settings.get('example_images_path')
            if not example_images_path:
                return web.json_response({
                    'success': False,
                    'error': 'No example images path configured'
                }, status=400)
            
            # Find the model and get current metadata
            lora_scanner = await ServiceRegistry.get_lora_scanner()
            checkpoint_scanner = await ServiceRegistry.get_checkpoint_scanner()
            embedding_scanner = await ServiceRegistry.get_embedding_scanner()
            
            model_data = None
            scanner = None
            
            # Check both scanners to find the model
            for scan_obj in [lora_scanner, checkpoint_scanner, embedding_scanner]:
                cache = await scan_obj.get_cached_data()
                for item in cache.raw_data:
                    if item.get('sha256') == model_hash:
                        model_data = item
                        scanner = scan_obj
                        break
                if model_data:
                    break
            
            if not model_data:
                return web.json_response({
                    'success': False,
                    'error': f"Model with hash {model_hash} not found in cache"
                }, status=404)
            
            # Create model folder
            model_folder = os.path.join(example_images_path, model_hash)
            os.makedirs(model_folder, exist_ok=True)
            
            imported_files = []
            errors = []
            newly_imported_paths = []
            
            # Process each file path
            for file_path in files_to_import:
                try:
                    # Ensure the file exists
                    if not os.path.isfile(file_path):
                        errors.append(f"File not found: {file_path}")
                        continue
                    
                    # Check if file type is supported
                    file_ext = os.path.splitext(file_path)[1].lower()
                    if not (file_ext in SUPPORTED_MEDIA_EXTENSIONS['images'] or 
                            file_ext in SUPPORTED_MEDIA_EXTENSIONS['videos']):
                        errors.append(f"Unsupported file type: {file_path}")
                        continue
                    
                    # Generate new filename using short ID instead of UUID
                    short_id = ExampleImagesProcessor.generate_short_id()
                    new_filename = f"custom_{short_id}{file_ext}"
                    
                    dest_path = os.path.join(model_folder, new_filename)
                    
                    # Copy the file
                    import shutil
                    shutil.copy2(file_path, dest_path)
                    # Store both the dest_path and the short_id
                    newly_imported_paths.append((dest_path, short_id))
                    
                    # Add to imported files list
                    imported_files.append({
                        'name': new_filename,
                        'path': f'/example_images_static/{model_hash}/{new_filename}',
                        'extension': file_ext,
                        'is_video': file_ext in SUPPORTED_MEDIA_EXTENSIONS['videos']
                    })
                except Exception as e:
                    errors.append(f"Error importing {file_path}: {str(e)}")
            
            # Update metadata with new example images
            regular_images, custom_images = await MetadataUpdater.update_metadata_after_import(
                model_hash, 
                model_data,
                scanner,
                newly_imported_paths
            )
            
            return web.json_response({
                'success': len(imported_files) > 0,
                'message': f'Successfully imported {len(imported_files)} files' + 
                        (f' with {len(errors)} errors' if errors else ''),
                'files': imported_files,
                'errors': errors,
                'regular_images': regular_images,
                'custom_images': custom_images,
                "model_file_path": model_data.get('file_path', ''),
            })
                
        except Exception as e:
            logger.error(f"Failed to import example images: {e}", exc_info=True)
            return web.json_response({
                'success': False,
                'error': str(e)
            }, status=500)
        finally:
            # Clean up temporary files
            for temp_file in temp_files_to_cleanup:
                try:
                    os.remove(temp_file)
                except Exception as e:
                    logger.error(f"Failed to remove temporary file {temp_file}: {e}")

    @staticmethod
    async def delete_custom_image(request):
        """
        Delete a custom example image for a model
        
        Accepts:
        - JSON request with model_hash and short_id
        
        Returns:
        - Success status and updated image lists
        """
        try:
            # Parse request data
            data = await request.json()
            model_hash = data.get('model_hash')
            short_id = data.get('short_id')
            
            if not model_hash or not short_id:
                return web.json_response({
                    'success': False,
                    'error': 'Missing required parameters: model_hash and short_id'
                }, status=400)
            
            # Get example images path
            example_images_path = settings.get('example_images_path')
            if not example_images_path:
                return web.json_response({
                    'success': False,
                    'error': 'No example images path configured'
                }, status=400)
            
            # Find the model and get current metadata
            lora_scanner = await ServiceRegistry.get_lora_scanner()
            checkpoint_scanner = await ServiceRegistry.get_checkpoint_scanner()
            embedding_scanner = await ServiceRegistry.get_embedding_scanner()
            
            model_data = None
            scanner = None
            
            # Check both scanners to find the model
            for scan_obj in [lora_scanner, checkpoint_scanner, embedding_scanner]:
                if scan_obj.has_hash(model_hash):
                    cache = await scan_obj.get_cached_data()
                    for item in cache.raw_data:
                        if item.get('sha256') == model_hash:
                            model_data = item
                            scanner = scan_obj
                            break
                if model_data:
                    break
            
            if not model_data:
                return web.json_response({
                    'success': False,
                    'error': f"Model with hash {model_hash} not found in cache"
                }, status=404)
            
            # Check if model has custom images
            if not model_data.get('civitai', {}).get('customImages'):
                return web.json_response({
                    'success': False,
                    'error': f"Model has no custom images"
                }, status=404)
            
            # Find the custom image with matching short_id
            custom_images = model_data['civitai']['customImages']
            matching_image = None
            new_custom_images = []
            
            for image in custom_images:
                if image.get('id') == short_id:
                    matching_image = image
                else:
                    new_custom_images.append(image)
            
            if not matching_image:
                return web.json_response({
                    'success': False,
                    'error': f"Custom image with id {short_id} not found"
                }, status=404)
            
            # Find and delete the actual file
            model_folder = os.path.join(example_images_path, model_hash)
            file_deleted = False
            
            if os.path.exists(model_folder):
                for filename in os.listdir(model_folder):
                    if f"custom_{short_id}" in filename:
                        file_path = os.path.join(model_folder, filename)
                        try:
                            os.remove(file_path)
                            file_deleted = True
                            logger.info(f"Deleted custom example file: {file_path}")
                            break
                        except Exception as e:
                            return web.json_response({
                                'success': False,
                                'error': f"Failed to delete file: {str(e)}"
                            }, status=500)
            
            if not file_deleted:
                logger.warning(f"File for custom example with id {short_id} not found, but metadata will still be updated")
            
            # Update metadata
            model_data['civitai']['customImages'] = new_custom_images
            
            # Save updated metadata to file
            file_path = model_data.get('file_path')
            if file_path:
                try:
                    # Create a copy of model data without 'folder' field
                    model_copy = model_data.copy()
                    model_copy.pop('folder', None)
                    
                    # Write metadata to file
                    await MetadataManager.save_metadata(file_path, model_copy)
                    logger.debug(f"Saved updated metadata for {model_data.get('model_name')}")
                except Exception as e:
                    logger.error(f"Failed to save metadata: {str(e)}")
                    return web.json_response({
                        'success': False,
                        'error': f"Failed to save metadata: {str(e)}"
                    }, status=500)
            
                # Update cache
                await scanner.update_single_model_cache(file_path, file_path, model_data)
            
            # Get regular images array (might be None)
            regular_images = model_data['civitai'].get('images', [])
            
            return web.json_response({
                'success': True,
                'regular_images': regular_images,
                'custom_images': new_custom_images,
                'model_file_path': model_data.get('file_path', '')
            })
                
        except Exception as e:
            logger.error(f"Failed to delete custom example image: {e}", exc_info=True)
            return web.json_response({
                'success': False,
                'error': str(e)
            }, status=500)