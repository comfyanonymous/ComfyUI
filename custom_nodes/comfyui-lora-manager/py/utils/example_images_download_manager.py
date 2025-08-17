import logging
import os
import asyncio
import json
import time
import aiohttp
from aiohttp import web
from ..services.service_registry import ServiceRegistry
from .example_images_processor import ExampleImagesProcessor
from .example_images_metadata import MetadataUpdater

logger = logging.getLogger(__name__)

# Download status tracking
download_task = None
is_downloading = False
download_progress = {
    'total': 0,
    'completed': 0,
    'current_model': '',
    'status': 'idle',  # idle, running, paused, completed, error
    'errors': [],
    'last_error': None,
    'start_time': None,
    'end_time': None,
    'processed_models': set(),  # Track models that have been processed
    'refreshed_models': set(),  # Track models that had metadata refreshed
    'failed_models': set()  # Track models that failed to download after metadata refresh
}

class DownloadManager:
    """Manages downloading example images for models"""
    
    @staticmethod
    async def start_download(request):
        """
        Start downloading example images for models
        
        Expects a JSON body with:
        {
            "output_dir": "path/to/output",  # Base directory to save example images
            "optimize": true,                # Whether to optimize images (default: true)
            "model_types": ["lora", "checkpoint"], # Model types to process (default: both)
            "delay": 1.0                     # Delay between downloads to avoid rate limiting (default: 1.0)
        }
        """
        global download_task, is_downloading, download_progress
        
        if is_downloading:
            # Create a copy for JSON serialization
            response_progress = download_progress.copy()
            response_progress['processed_models'] = list(download_progress['processed_models'])
            response_progress['refreshed_models'] = list(download_progress['refreshed_models'])
            response_progress['failed_models'] = list(download_progress['failed_models'])
            
            return web.json_response({
                'success': False,
                'error': 'Download already in progress',
                'status': response_progress
            }, status=400)
        
        try:
            # Parse the request body
            data = await request.json()
            output_dir = data.get('output_dir')
            optimize = data.get('optimize', True)
            model_types = data.get('model_types', ['lora', 'checkpoint'])
            delay = float(data.get('delay', 0.2)) # Default to 0.2 seconds
            
            if not output_dir:
                return web.json_response({
                    'success': False,
                    'error': 'Missing output_dir parameter'
                }, status=400)
            
            # Create the output directory
            os.makedirs(output_dir, exist_ok=True)
            
            # Initialize progress tracking
            download_progress['total'] = 0
            download_progress['completed'] = 0
            download_progress['current_model'] = ''
            download_progress['status'] = 'running'
            download_progress['errors'] = []
            download_progress['last_error'] = None
            download_progress['start_time'] = time.time()
            download_progress['end_time'] = None
            
            # Get the processed models list from a file if it exists
            progress_file = os.path.join(output_dir, '.download_progress.json')
            if os.path.exists(progress_file):
                try:
                    with open(progress_file, 'r', encoding='utf-8') as f:
                        saved_progress = json.load(f)
                        download_progress['processed_models'] = set(saved_progress.get('processed_models', []))
                        download_progress['failed_models'] = set(saved_progress.get('failed_models', []))
                        logger.debug(f"Loaded previous progress, {len(download_progress['processed_models'])} models already processed, {len(download_progress['failed_models'])} models marked as failed")
                except Exception as e:
                    logger.error(f"Failed to load progress file: {e}")
                    download_progress['processed_models'] = set()
                    download_progress['failed_models'] = set()
            else:
                download_progress['processed_models'] = set()
                download_progress['failed_models'] = set()
            
            # Start the download task
            is_downloading = True
            download_task = asyncio.create_task(
                DownloadManager._download_all_example_images(
                    output_dir, 
                    optimize, 
                    model_types,
                    delay
                )
            )
            
            # Create a copy for JSON serialization
            response_progress = download_progress.copy()
            response_progress['processed_models'] = list(download_progress['processed_models'])
            response_progress['refreshed_models'] = list(download_progress['refreshed_models'])
            response_progress['failed_models'] = list(download_progress['failed_models'])
            
            return web.json_response({
                'success': True,
                'message': 'Download started',
                'status': response_progress
            })
            
        except Exception as e:
            logger.error(f"Failed to start example images download: {e}", exc_info=True)
            return web.json_response({
                'success': False,
                'error': str(e)
            }, status=500)
            
    @staticmethod
    async def get_status(request):
        """Get the current status of example images download"""
        global download_progress
        
        # Create a copy of the progress dict with the set converted to a list for JSON serialization
        response_progress = download_progress.copy()
        response_progress['processed_models'] = list(download_progress['processed_models'])
        response_progress['refreshed_models'] = list(download_progress['refreshed_models'])
        response_progress['failed_models'] = list(download_progress['failed_models'])
        
        return web.json_response({
            'success': True,
            'is_downloading': is_downloading,
            'status': response_progress
        })

    @staticmethod
    async def pause_download(request):
        """Pause the example images download"""
        global download_progress
        
        if not is_downloading:
            return web.json_response({
                'success': False,
                'error': 'No download in progress'
            }, status=400)
        
        download_progress['status'] = 'paused'
        
        return web.json_response({
            'success': True,
            'message': 'Download paused'
        })

    @staticmethod
    async def resume_download(request):
        """Resume the example images download"""
        global download_progress
        
        if not is_downloading:
            return web.json_response({
                'success': False,
                'error': 'No download in progress'
            }, status=400)
        
        if download_progress['status'] == 'paused':
            download_progress['status'] = 'running'
            
            return web.json_response({
                'success': True,
                'message': 'Download resumed'
            })
        else:
            return web.json_response({
                'success': False,
                'error': f"Download is in '{download_progress['status']}' state, cannot resume"
            }, status=400)
    
    @staticmethod
    async def _download_all_example_images(output_dir, optimize, model_types, delay):
        """Download example images for all models"""
        global is_downloading, download_progress
        
        # Create independent download session
        connector = aiohttp.TCPConnector(
            ssl=True,
            limit=3,
            force_close=False,
            enable_cleanup_closed=True
        )
        timeout = aiohttp.ClientTimeout(total=None, connect=60, sock_read=60)
        independent_session = aiohttp.ClientSession(
            connector=connector,
            trust_env=True,
            timeout=timeout
        )
        
        try:
            # Get scanners
            scanners = []
            if 'lora' in model_types:
                lora_scanner = await ServiceRegistry.get_lora_scanner()
                scanners.append(('lora', lora_scanner))
            
            if 'checkpoint' in model_types:
                checkpoint_scanner = await ServiceRegistry.get_checkpoint_scanner()
                scanners.append(('checkpoint', checkpoint_scanner))

            if 'embedding' in model_types:
                embedding_scanner = await ServiceRegistry.get_embedding_scanner()
                scanners.append(('embedding', embedding_scanner))
            
            # Get all models
            all_models = []
            for scanner_type, scanner in scanners:
                cache = await scanner.get_cached_data()
                if cache and cache.raw_data:
                    for model in cache.raw_data:
                        if model.get('sha256'):
                            all_models.append((scanner_type, model, scanner))
            
            # Update total count
            download_progress['total'] = len(all_models)
            logger.debug(f"Found {download_progress['total']} models to process")
            
            # Process each model
            for i, (scanner_type, model, scanner) in enumerate(all_models):
                # Main logic for processing model is here, but actual operations are delegated to other classes
                was_remote_download = await DownloadManager._process_model(
                    scanner_type, model, scanner, 
                    output_dir, optimize, independent_session
                )
                
                # Update progress
                download_progress['completed'] += 1
                
                # Only add delay after remote download of models, and not after processing the last model
                if was_remote_download and i < len(all_models) - 1 and download_progress['status'] == 'running':
                    await asyncio.sleep(delay)
            
            # Mark as completed
            download_progress['status'] = 'completed'
            download_progress['end_time'] = time.time()
            logger.debug(f"Example images download completed: {download_progress['completed']}/{download_progress['total']} models processed")
            
        except Exception as e:
            error_msg = f"Error during example images download: {str(e)}"
            logger.error(error_msg, exc_info=True)
            download_progress['errors'].append(error_msg)
            download_progress['last_error'] = error_msg
            download_progress['status'] = 'error'
            download_progress['end_time'] = time.time()
        
        finally:
            # Close the independent session
            try:
                await independent_session.close()
            except Exception as e:
                logger.error(f"Error closing download session: {e}")
                
            # Save final progress to file
            try:
                DownloadManager._save_progress(output_dir)
            except Exception as e:
                logger.error(f"Failed to save progress file: {e}")
            
            # Set download status to not downloading
            is_downloading = False
    
    @staticmethod
    async def _process_model(scanner_type, model, scanner, output_dir, optimize, independent_session):
        """Process a single model download"""
        global download_progress
        
        # Check if download is paused
        while download_progress['status'] == 'paused':
            await asyncio.sleep(1)
        
        # Check if download should continue
        if download_progress['status'] != 'running':
            logger.info(f"Download stopped: {download_progress['status']}")
            return False  # Return False to indicate no remote download happened
        
        model_hash = model.get('sha256', '').lower()
        model_name = model.get('model_name', 'Unknown')
        model_file_path = model.get('file_path', '')
        model_file_name = model.get('file_name', '')
        
        try:
            # Update current model info
            download_progress['current_model'] = f"{model_name} ({model_hash[:8]})"
            
            # Skip if already in failed models
            if model_hash in download_progress['failed_models']:
                logger.debug(f"Skipping known failed model: {model_name}")
                return False
            
            # Skip if already processed AND directory exists with files
            if model_hash in download_progress['processed_models']:
                model_dir = os.path.join(output_dir, model_hash)
                has_files = os.path.exists(model_dir) and any(os.listdir(model_dir))
                if has_files:
                    logger.debug(f"Skipping already processed model: {model_name}")
                    return False
                else:
                    logger.info(f"Model {model_name} marked as processed but folder empty or missing, reprocessing")
                    # Remove from processed models since we need to reprocess
                    download_progress['processed_models'].discard(model_hash)
            
            # Create model directory
            model_dir = os.path.join(output_dir, model_hash)
            os.makedirs(model_dir, exist_ok=True)
            
            # First check for local example images - local processing doesn't need delay
            local_images_processed = await ExampleImagesProcessor.process_local_examples(
                model_file_path, model_file_name, model_name, model_dir, optimize
            )
            
            # If we processed local images, update metadata
            if local_images_processed:
                await MetadataUpdater.update_metadata_from_local_examples(
                    model_hash, model, scanner_type, scanner, model_dir
                )
                download_progress['processed_models'].add(model_hash)
                return False  # Return False to indicate no remote download happened
            
            # If no local images, try to download from remote
            elif model.get('civitai') and model.get('civitai', {}).get('images'):
                images = model.get('civitai', {}).get('images', [])
                
                success, is_stale = await ExampleImagesProcessor.download_model_images(
                    model_hash, model_name, images, model_dir, optimize, independent_session
                )
                
                # If metadata is stale, try to refresh it
                if is_stale and model_hash not in download_progress['refreshed_models']:
                    await MetadataUpdater.refresh_model_metadata(
                        model_hash, model_name, scanner_type, scanner
                    )
                    
                    # Get the updated model data
                    updated_model = await MetadataUpdater.get_updated_model(
                        model_hash, scanner
                    )
                    
                    if updated_model and updated_model.get('civitai', {}).get('images'):
                        # Retry download with updated metadata
                        updated_images = updated_model.get('civitai', {}).get('images', [])
                        success, _ = await ExampleImagesProcessor.download_model_images(
                            model_hash, model_name, updated_images, model_dir, optimize, independent_session
                        )
                    
                    download_progress['refreshed_models'].add(model_hash)
                
                # Mark as processed if successful, or as failed if unsuccessful after refresh
                if success:
                    download_progress['processed_models'].add(model_hash)
                else:
                    # If we refreshed metadata and still failed, mark as permanently failed
                    if model_hash in download_progress['refreshed_models']:
                        download_progress['failed_models'].add(model_hash)
                        logger.info(f"Marking model {model_name} as failed after metadata refresh")
                    
                return True  # Return True to indicate a remote download happened
            else:
                # No civitai data or images available, mark as failed to avoid future attempts
                download_progress['failed_models'].add(model_hash)
                logger.debug(f"No civitai images available for model {model_name}, marking as failed")
            
            # Save progress periodically
            if download_progress['completed'] % 10 == 0 or download_progress['completed'] == download_progress['total'] - 1:
                DownloadManager._save_progress(output_dir)
                
            return False  # Default return if no conditions met
                
        except Exception as e:
            error_msg = f"Error processing model {model.get('model_name')}: {str(e)}"
            logger.error(error_msg, exc_info=True)
            download_progress['errors'].append(error_msg)
            download_progress['last_error'] = error_msg
            return False  # Return False on exception
    
    @staticmethod
    def _save_progress(output_dir):
        """Save download progress to file"""
        global download_progress
        try:
            progress_file = os.path.join(output_dir, '.download_progress.json')
            
            # Read existing progress file if it exists
            existing_data = {}
            if os.path.exists(progress_file):
                try:
                    with open(progress_file, 'r', encoding='utf-8') as f:
                        existing_data = json.load(f)
                except Exception as e:
                    logger.warning(f"Failed to read existing progress file: {e}")
            
            # Create new progress data
            progress_data = {
                'processed_models': list(download_progress['processed_models']),
                'refreshed_models': list(download_progress['refreshed_models']),
                'failed_models': list(download_progress['failed_models']),
                'completed': download_progress['completed'],
                'total': download_progress['total'],
                'last_update': time.time()
            }
            
            # Preserve existing fields (especially naming_version)
            for key, value in existing_data.items():
                if key not in progress_data:
                    progress_data[key] = value
            
            # Write updated progress data
            with open(progress_file, 'w', encoding='utf-8') as f:
                json.dump(progress_data, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save progress file: {e}")