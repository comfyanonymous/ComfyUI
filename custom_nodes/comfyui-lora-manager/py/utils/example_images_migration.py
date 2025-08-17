import asyncio
import logging
import os
import re
import json
from ..services.settings_manager import settings
from ..services.service_registry import ServiceRegistry
from ..utils.metadata_manager import MetadataManager
from ..utils.example_images_processor import ExampleImagesProcessor
from ..utils.constants import SUPPORTED_MEDIA_EXTENSIONS

logger = logging.getLogger(__name__)

CURRENT_NAMING_VERSION = 2  # Increment this when naming conventions change

class ExampleImagesMigration:
    """Handles migrations for example images naming conventions"""
    
    @staticmethod
    async def check_and_run_migrations():
        """Check if migrations are needed and run them in background"""
        example_images_path = settings.get('example_images_path')
        if not example_images_path or not os.path.exists(example_images_path):
            logger.debug("No example images path configured or path doesn't exist, skipping migrations")
            return
        
        # Check current version from progress file
        current_version = 0
        progress_file = os.path.join(example_images_path, '.download_progress.json')
        if os.path.exists(progress_file):
            try:
                with open(progress_file, 'r', encoding='utf-8') as f:
                    progress_data = json.load(f)
                    current_version = progress_data.get('naming_version', 0)
            except Exception as e:
                logger.error(f"Failed to load progress file for migration check: {e}")
        
        # If current version is less than target version, start migration
        if current_version < CURRENT_NAMING_VERSION:
            logger.info(f"Starting example images naming migration from v{current_version} to v{CURRENT_NAMING_VERSION}")
            # Start migration in background task
            asyncio.create_task(
                ExampleImagesMigration.run_migrations(example_images_path, current_version, CURRENT_NAMING_VERSION)
            )
    
    @staticmethod
    async def run_migrations(example_images_path, from_version, to_version):
        """Run necessary migrations based on version difference"""
        try:
            # Get all model folders
            model_folders = []
            for item in os.listdir(example_images_path):
                item_path = os.path.join(example_images_path, item)
                if os.path.isdir(item_path) and len(item) == 64:  # SHA256 hash is 64 chars
                    model_folders.append(item_path)
            
            logger.info(f"Found {len(model_folders)} model folders to check for migration")
            
            # Apply migrations sequentially
            if from_version < 1 and to_version >= 1:
                await ExampleImagesMigration._migrate_to_v1(model_folders)
            
            if from_version < 2 and to_version >= 2:
                await ExampleImagesMigration._migrate_to_v2(model_folders)
            
            # Update version in progress file
            progress_file = os.path.join(example_images_path, '.download_progress.json')
            try:
                progress_data = {}
                if os.path.exists(progress_file):
                    with open(progress_file, 'r', encoding='utf-8') as f:
                        progress_data = json.load(f)
                
                progress_data['naming_version'] = to_version
                
                with open(progress_file, 'w', encoding='utf-8') as f:
                    json.dump(progress_data, f, indent=2)
                    
                logger.info(f"Example images naming migration to v{to_version} completed")
                
            except Exception as e:
                logger.error(f"Failed to update version in progress file: {e}")
                
        except Exception as e:
            logger.error(f"Error during migration: {e}", exc_info=True)
    
    @staticmethod
    async def _migrate_to_v1(model_folders):
        """Migrate from 1-based to 0-based indexing"""
        count = 0
        for folder in model_folders:
            has_one_based = False
            has_zero_based = False
            files_to_rename = []
            
            # Check naming pattern in this folder
            for file in os.listdir(folder):
                if re.match(r'image_1\.\w+$', file):
                    has_one_based = True
                if re.match(r'image_0\.\w+$', file):
                    has_zero_based = True
            
            # Only migrate folders with 1-based indexing and no 0-based
            if has_one_based and not has_zero_based:
                # Create rename mapping
                for file in os.listdir(folder):
                    match = re.match(r'image_(\d+)\.(\w+)$', file)
                    if match:
                        index = int(match.group(1))
                        ext = match.group(2)
                        if index > 0:  # Only rename if index is positive
                            files_to_rename.append((
                                file,
                                f"image_{index-1}.{ext}"
                            ))
                
                # Use temporary names to avoid conflicts
                for old_name, new_name in files_to_rename:
                    old_path = os.path.join(folder, old_name)
                    temp_path = os.path.join(folder, f"temp_{old_name}")
                    try:
                        os.rename(old_path, temp_path)
                    except Exception as e:
                        logger.error(f"Failed to rename {old_path} to {temp_path}: {e}")
                
                # Rename from temporary names to final names
                for old_name, new_name in files_to_rename:
                    temp_path = os.path.join(folder, f"temp_{old_name}")
                    new_path = os.path.join(folder, new_name)
                    try:
                        os.rename(temp_path, new_path)
                        logger.debug(f"Renamed {old_name} to {new_name} in {folder}")
                    except Exception as e:
                        logger.error(f"Failed to rename {temp_path} to {new_path}: {e}")
                
                count += 1
                
                # Give other tasks a chance to run
                if count % 10 == 0:
                    await asyncio.sleep(0)
        
        logger.info(f"Migrated {count} folders from 1-based to 0-based indexing")
    
    @staticmethod
    async def _migrate_to_v2(model_folders):
        """
        Migrate to v2 naming scheme:
        - Move custom examples from images array to customImages array
        - Rename files from image_<index>.<ext> to custom_<short_id>.<ext>
        - Add id field to each custom image entry
        """
        count = 0
        updated_models = 0
        migration_errors = 0
        
        # Get scanner instances
        lora_scanner = await ServiceRegistry.get_lora_scanner()
        checkpoint_scanner = await ServiceRegistry.get_checkpoint_scanner()
        
        # Wait until scanners are initialized
        scanners = [lora_scanner, checkpoint_scanner]
        for scanner in scanners:
            if scanner.is_initializing():
                logger.info("Waiting for scanners to complete initialization before starting migration...")
                initialized = False
                retry_count = 0
                while not initialized and retry_count < 120:  # Wait up to 120 seconds
                    await asyncio.sleep(1)
                    initialized = not scanner.is_initializing()
                    retry_count += 1
                
                if not initialized:
                    logger.warning("Scanner initialization timeout - proceeding with migration anyway")
        
        logger.info(f"Starting migration to v2 naming scheme for {len(model_folders)} model folders")
        
        for folder in model_folders:
            try:
                # Extract model hash from folder name
                model_hash = os.path.basename(folder)
                if not model_hash or len(model_hash) != 64:
                    continue
                
                # Find the model in scanner cache
                model_data = None
                scanner = None
                
                for scan_obj in scanners:
                  if scan_obj.has_hash(model_hash):
                    cache = await scan_obj.get_cached_data()
                    for item in cache.raw_data:
                      if item.get('sha256') == model_hash:
                        model_data = item
                        scanner = scan_obj
                        break
                  if model_data:
                    break
                
                if not model_data or not scanner:
                    logger.debug(f"Model with hash {model_hash} not found in cache, skipping migration")
                    continue
                
                # Clone model data to avoid modifying the cache directly
                model_metadata = model_data.copy()
                
                # Check if model has civitai metadata
                if not model_metadata.get('civitai'):
                    continue
                
                # Get images array
                images = model_metadata.get('civitai', {}).get('images', [])
                if not images:
                    continue
                
                # Initialize customImages array if it doesn't exist
                if not model_metadata['civitai'].get('customImages'):
                    model_metadata['civitai']['customImages'] = []
                
                # Find custom examples (entries with empty url)
                custom_indices = []
                for i, image in enumerate(images):
                    if image.get('url') == "":
                        custom_indices.append(i)
                
                if not custom_indices:
                    continue
                    
                logger.debug(f"Found {len(custom_indices)} custom examples in {model_hash}")
                
                # Process each custom example
                for index in custom_indices:
                    try:
                        image_entry = images[index]
                        
                        # Determine media type based on the entry type
                        media_type = 'videos' if image_entry.get('type') == 'video' else 'images'
                        extensions_to_try = SUPPORTED_MEDIA_EXTENSIONS[media_type]
                        
                        # Find the image file by trying possible extensions
                        old_path = None
                        old_filename = None
                        found = False
                        
                        for ext in extensions_to_try:
                            test_path = os.path.join(folder, f"image_{index}{ext}")
                            if os.path.exists(test_path):
                                old_path = test_path
                                old_filename = f"image_{index}{ext}"
                                found = True
                                break
                                
                        if not found:
                            logger.warning(f"Could not find file for index {index} in {model_hash}, skipping")
                            continue
                        
                        # Generate short ID for the custom example
                        short_id = ExampleImagesProcessor.generate_short_id()
                        
                        # Get file extension
                        file_ext = os.path.splitext(old_path)[1]
                        
                        # Create new filename
                        new_filename = f"custom_{short_id}{file_ext}"
                        new_path = os.path.join(folder, new_filename)
                        
                        # Rename the file
                        try:
                            os.rename(old_path, new_path)
                            logger.debug(f"Renamed {old_filename} to {new_filename} in {folder}")
                        except Exception as e:
                            logger.error(f"Failed to rename {old_path} to {new_path}: {e}")
                            continue
                        
                        # Create a copy of the image entry with the id field
                        custom_entry = image_entry.copy()
                        custom_entry['id'] = short_id
                        
                        # Add to customImages array
                        model_metadata['civitai']['customImages'].append(custom_entry)
                        
                        count += 1
                        
                    except Exception as e:
                        logger.error(f"Error migrating custom example at index {index} for {model_hash}: {e}")
                
                # Remove custom examples from the original images array
                model_metadata['civitai']['images'] = [
                    img for i, img in enumerate(images) if i not in custom_indices
                ]
                
                # Save the updated metadata
                file_path = model_data.get('file_path')
                if file_path:
                    try:
                        # Create a copy of model data without 'folder' field
                        model_copy = model_metadata.copy()
                        model_copy.pop('folder', None)
                        
                        # Save metadata to file
                        await MetadataManager.save_metadata(file_path, model_copy)
                        
                        # Update scanner cache
                        await scanner.update_single_model_cache(file_path, file_path, model_metadata)
                        
                        updated_models += 1
                    except Exception as e:
                        logger.error(f"Failed to save metadata for {model_hash}: {e}")
                        migration_errors += 1
                
                # Give other tasks a chance to run
                if count % 10 == 0:
                    await asyncio.sleep(0)
                    
            except Exception as e:
                logger.error(f"Error migrating folder {folder}: {e}")
                migration_errors += 1
        
        logger.info(f"Migration to v2 complete: migrated {count} custom examples across {updated_models} models with {migration_errors} errors")