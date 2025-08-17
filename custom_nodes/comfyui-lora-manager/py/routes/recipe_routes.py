import os
import time
import base64
import jinja2
import numpy as np
from PIL import Image
import io
import logging
from aiohttp import web
from typing import Dict
import tempfile
import json
import asyncio
import sys
from ..utils.exif_utils import ExifUtils
from ..recipes import RecipeParserFactory
from ..utils.constants import CARD_PREVIEW_WIDTH

from ..services.settings_manager import settings
from ..config import config

# Check if running in standalone mode
standalone_mode = 'nodes' not in sys.modules

from ..services.service_registry import ServiceRegistry  # Add ServiceRegistry import

# Only import MetadataRegistry in non-standalone mode
if not standalone_mode:
    # Import metadata_collector functions and classes conditionally
    from ..metadata_collector import get_metadata  # Add MetadataCollector import
    from ..metadata_collector.metadata_processor import MetadataProcessor  # Add MetadataProcessor import
    from ..metadata_collector.metadata_registry import MetadataRegistry

logger = logging.getLogger(__name__)

class RecipeRoutes:
    """API route handlers for Recipe management"""

    def __init__(self):
        # Initialize service references as None, will be set during async init
        self.recipe_scanner = None
        self.civitai_client = None
        self.template_env = jinja2.Environment(
            loader=jinja2.FileSystemLoader(config.templates_path),
            autoescape=True
        )
        
        # Pre-warm the cache
        self._init_cache_task = None

    async def init_services(self):
        """Initialize services from ServiceRegistry"""
        self.recipe_scanner = await ServiceRegistry.get_recipe_scanner()
        self.civitai_client = await ServiceRegistry.get_civitai_client()

    @classmethod
    def setup_routes(cls, app: web.Application):
        """Register API routes"""
        routes = cls()
        app.router.add_get('/loras/recipes', routes.handle_recipes_page)

        app.router.add_get('/api/recipes', routes.get_recipes)
        app.router.add_get('/api/recipe/{recipe_id}', routes.get_recipe_detail)
        app.router.add_post('/api/recipes/analyze-image', routes.analyze_recipe_image)
        app.router.add_post('/api/recipes/analyze-local-image', routes.analyze_local_image)
        app.router.add_post('/api/recipes/save', routes.save_recipe)
        app.router.add_delete('/api/recipe/{recipe_id}', routes.delete_recipe)
        
        # Add new filter-related endpoints
        app.router.add_get('/api/recipes/top-tags', routes.get_top_tags)
        app.router.add_get('/api/recipes/base-models', routes.get_base_models)
        
        # Add new sharing endpoints
        app.router.add_get('/api/recipe/{recipe_id}/share', routes.share_recipe)
        app.router.add_get('/api/recipe/{recipe_id}/share/download', routes.download_shared_recipe)
        
        # Add new endpoint for getting recipe syntax
        app.router.add_get('/api/recipe/{recipe_id}/syntax', routes.get_recipe_syntax)
        
        # Add new endpoint for updating recipe metadata (name, tags and source_path)
        app.router.add_put('/api/recipe/{recipe_id}/update', routes.update_recipe)
        
        # Add new endpoint for reconnecting deleted LoRAs
        app.router.add_post('/api/recipe/lora/reconnect', routes.reconnect_lora)
        
        # Add new endpoint for finding duplicate recipes
        app.router.add_get('/api/recipes/find-duplicates', routes.find_duplicates)
        
        # Add new endpoint for bulk deletion of recipes
        app.router.add_post('/api/recipes/bulk-delete', routes.bulk_delete)
        
        # Start cache initialization
        app.on_startup.append(routes._init_cache)
        
        app.router.add_post('/api/recipes/save-from-widget', routes.save_recipe_from_widget)
        
        # Add route to get recipes for a specific Lora
        app.router.add_get('/api/recipes/for-lora', routes.get_recipes_for_lora)
        
        # Add new endpoint for scanning and rebuilding the recipe cache
        app.router.add_get('/api/recipes/scan', routes.scan_recipes)
    
    async def _init_cache(self, app):
        """Initialize cache on startup"""
        try:
            # Initialize services first
            await self.init_services()
            
            # Now that services are initialized, get the lora scanner
            lora_scanner = self.recipe_scanner._lora_scanner
            
            # Get lora cache to ensure it's initialized
            lora_cache = await lora_scanner.get_cached_data()
            
            # Verify hash index is built
            if hasattr(lora_scanner, '_hash_index'):
                hash_index_size = len(lora_scanner._hash_index._hash_to_path) if hasattr(lora_scanner._hash_index, '_hash_to_path') else 0
            
            # Now that lora scanner is initialized, initialize recipe cache
            await self.recipe_scanner.get_cached_data(force_refresh=True)
        except Exception as e:
            logger.error(f"Error pre-warming recipe cache: {e}", exc_info=True)

    async def handle_recipes_page(self, request: web.Request) -> web.Response:
        """Handle GET /loras/recipes request"""
        try:
            # Ensure services are initialized
            await self.init_services()
            
            # Skip initialization check and directly try to get cached data
            try:
                # Recipe scanner will initialize cache if needed
                await self.recipe_scanner.get_cached_data(force_refresh=False)
                template = self.template_env.get_template('recipes.html')
                rendered = template.render(
                    recipes=[],  # Frontend will load recipes via API
                    is_initializing=False,
                    settings=settings,
                    request=request
                )
            except Exception as cache_error:
                logger.error(f"Error loading recipe cache data: {cache_error}")
                # Still keep error handling - show initializing page on error
                template = self.template_env.get_template('recipes.html')
                rendered = template.render(
                    is_initializing=True,
                    settings=settings,
                    request=request
                )
                logger.info("Recipe cache error, returning initialization page")
            
            return web.Response(
                text=rendered,
                content_type='text/html'
            )
            
        except Exception as e:
            logger.error(f"Error handling recipes request: {e}", exc_info=True)
            return web.Response(
                text="Error loading recipes page",
                status=500
            )
    
    async def get_recipes(self, request: web.Request) -> web.Response:
        """API endpoint for getting paginated recipes"""
        try:
            # Ensure services are initialized
            await self.init_services()
            
            # Get query parameters with defaults
            page = int(request.query.get('page', '1'))
            page_size = int(request.query.get('page_size', '20'))
            sort_by = request.query.get('sort_by', 'date')
            search = request.query.get('search', None)
            
            # Get search options (renamed for better clarity)
            search_title = request.query.get('search_title', 'true').lower() == 'true'
            search_tags = request.query.get('search_tags', 'true').lower() == 'true'  
            search_lora_name = request.query.get('search_lora_name', 'true').lower() == 'true'
            search_lora_model = request.query.get('search_lora_model', 'true').lower() == 'true'
            
            # Get filter parameters
            base_models = request.query.get('base_models', None)
            tags = request.query.get('tags', None)
            
            # New parameter: get LoRA hash filter
            lora_hash = request.query.get('lora_hash', None)
            
            # Parse filter parameters
            filters = {}
            if base_models:
                filters['base_model'] = base_models.split(',')
            if tags:
                filters['tags'] = tags.split(',')
            
            # Add search options to filters
            search_options = {
                'title': search_title,
                'tags': search_tags,
                'lora_name': search_lora_name,
                'lora_model': search_lora_model
            }

            # Get paginated data with the new lora_hash parameter
            result = await self.recipe_scanner.get_paginated_data(
                page=page,
                page_size=page_size,
                sort_by=sort_by,
                search=search,
                filters=filters,
                search_options=search_options,
                lora_hash=lora_hash
            )
            
            # Format the response data with static URLs for file paths
            for item in result['items']:
                # Always ensure file_url is set
                if 'file_path' in item:
                    item['file_url'] = self._format_recipe_file_url(item['file_path'])
                else:
                    item['file_url'] = '/loras_static/images/no-preview.png'
                
                # 确保 loras 数组存在
                if 'loras' not in item:
                    item['loras'] = []
                    
                # 确保有 base_model 字段
                if 'base_model' not in item:
                    item['base_model'] = ""
            
            return web.json_response(result)
        except Exception as e:
            logger.error(f"Error retrieving recipes: {e}", exc_info=True)
            return web.json_response({"error": str(e)}, status=500)

    async def get_recipe_detail(self, request: web.Request) -> web.Response:
        """Get detailed information about a specific recipe"""
        try:
            # Ensure services are initialized
            await self.init_services()
            
            recipe_id = request.match_info['recipe_id']
            
            # Use the new get_recipe_by_id method from recipe_scanner
            recipe = await self.recipe_scanner.get_recipe_by_id(recipe_id)
            
            if not recipe:
                return web.json_response({"error": "Recipe not found"}, status=404)
            
            return web.json_response(recipe)
        except Exception as e:
            logger.error(f"Error retrieving recipe details: {e}", exc_info=True)
            return web.json_response({"error": str(e)}, status=500)
    
    def _format_recipe_file_url(self, file_path: str) -> str:
        """Format file path for recipe image as a URL"""
        try:
            # Return the file URL directly for the first lora root's preview
            recipes_dir = os.path.join(config.loras_roots[0], "recipes").replace(os.sep, '/')
            if file_path.replace(os.sep, '/').startswith(recipes_dir):
                relative_path = os.path.relpath(file_path, config.loras_roots[0]).replace(os.sep, '/')
                return f"/loras_static/root1/preview/{relative_path}" 
            
            # If not in recipes dir, try to create a valid URL from the file path
            file_name = os.path.basename(file_path)
            return f"/loras_static/root1/preview/recipes/{file_name}"
        except Exception as e:
            logger.error(f"Error formatting recipe file URL: {e}", exc_info=True)
            return '/loras_static/images/no-preview.png'  # Return default image on error
    
    def _format_recipe_data(self, recipe: Dict) -> Dict:
        """Format recipe data for API response"""
        formatted = {**recipe}  # Copy all fields
        
        # Format file paths to URLs
        if 'file_path' in formatted:
            formatted['file_url'] = self._format_recipe_file_url(formatted['file_path'])
        
        # Format dates for display
        for date_field in ['created_date', 'modified']:
            if date_field in formatted:
                formatted[f"{date_field}_formatted"] = self._format_timestamp(formatted[date_field])
        
        return formatted
    
    def _format_timestamp(self, timestamp: float) -> str:
        """Format timestamp for display"""
        from datetime import datetime
        return datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S') 

    async def analyze_recipe_image(self, request: web.Request) -> web.Response:
        """Analyze an uploaded image or URL for recipe metadata"""
        temp_path = None
        try:
            # Ensure services are initialized
            await self.init_services()
            
            # Check if request contains multipart data (image) or JSON data (url)
            content_type = request.headers.get('Content-Type', '')
            
            is_url_mode = False
            metadata = None  # Initialize metadata variable
            
            if 'multipart/form-data' in content_type:
                # Handle image upload
                reader = await request.multipart()
                field = await reader.next()
                
                if field.name != 'image':
                    return web.json_response({
                        "error": "No image field found",
                        "loras": []
                    }, status=400)
                
                # Create a temporary file to store the uploaded image
                with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as temp_file:
                    while True:
                        chunk = await field.read_chunk()
                        if not chunk:
                            break
                        temp_file.write(chunk)
                    temp_path = temp_file.name
                    
            elif 'application/json' in content_type:
                # Handle URL input
                data = await request.json()
                url = data.get('url')
                is_url_mode = True
                
                if not url:
                    return web.json_response({
                        "error": "No URL provided",
                        "loras": []
                    }, status=400)
                
                # Check if this is a Civitai image URL
                import re
                civitai_image_match = re.match(r'https://civitai\.com/images/(\d+)', url)
                
                if civitai_image_match:
                    # Extract image ID and fetch image info using get_image_info
                    image_id = civitai_image_match.group(1)
                    image_info = await self.civitai_client.get_image_info(image_id)
                    
                    if not image_info:
                        return web.json_response({
                            "error": "Failed to fetch image information from Civitai",
                            "loras": []
                        }, status=400)
                    
                    # Get image URL from response
                    image_url = image_info.get('url')
                    if not image_url:
                        return web.json_response({
                            "error": "No image URL found in Civitai response",
                            "loras": []
                        }, status=400)
                    
                    # Download image directly from URL
                    session = await self.civitai_client.session
                    # Create a temporary file to save the downloaded image
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as temp_file:
                        temp_path = temp_file.name
                    
                    async with session.get(image_url) as response:
                        if response.status != 200:
                            return web.json_response({
                                "error": f"Failed to download image from URL: HTTP {response.status}",
                                "loras": []
                            }, status=400)
                        
                        with open(temp_path, 'wb') as f:
                            f.write(await response.read())
                    
                    # Use meta field from image_info as metadata
                    if 'meta' in image_info:
                        metadata = image_info['meta']
            
            # If metadata wasn't obtained from Civitai API, extract it from the image
            if metadata is None:
                # Extract metadata from the image using ExifUtils
                metadata = ExifUtils.extract_image_metadata(temp_path)
            
            # If no metadata found, return a more specific error
            if not metadata:
                result = {
                    "error": "No metadata found in this image",
                    "loras": []  # Return empty loras array to prevent client-side errors
                }
                
                # For URL mode, include the image data as base64
                if is_url_mode and temp_path:
                    with open(temp_path, "rb") as image_file:
                        result["image_base64"] = base64.b64encode(image_file.read()).decode('utf-8')
                    
                return web.json_response(result, status=200)
            
            # Use the parser factory to get the appropriate parser
            parser = RecipeParserFactory.create_parser(metadata)

            if parser is None:
                result = {
                    "error": "No parser found for this image",
                    "loras": []  # Return empty loras array to prevent client-side errors
                }
                
                # For URL mode, include the image data as base64
                if is_url_mode and temp_path:
                    with open(temp_path, "rb") as image_file:
                        result["image_base64"] = base64.b64encode(image_file.read()).decode('utf-8')
                    
                return web.json_response(result, status=200)
            
            # Parse the metadata
            result = await parser.parse_metadata(
                metadata, 
                recipe_scanner=self.recipe_scanner, 
                civitai_client=self.civitai_client
            )
            
            # For URL mode, include the image data as base64
            if is_url_mode and temp_path:
                with open(temp_path, "rb") as image_file:
                    result["image_base64"] = base64.b64encode(image_file.read()).decode('utf-8')
            
            # Check for errors
            if "error" in result and not result.get("loras"):
                return web.json_response(result, status=200)
            
            # Calculate fingerprint from parsed loras
            from ..utils.utils import calculate_recipe_fingerprint
            fingerprint = calculate_recipe_fingerprint(result.get("loras", []))
            
            # Add fingerprint to result
            result["fingerprint"] = fingerprint
            
            # Find matching recipes with the same fingerprint
            matching_recipes = []
            if fingerprint:
                matching_recipes = await self.recipe_scanner.find_recipes_by_fingerprint(fingerprint)
                
            # Add matching recipes to result
            result["matching_recipes"] = matching_recipes
            
            return web.json_response(result)
            
        except Exception as e:
            logger.error(f"Error analyzing recipe image: {e}", exc_info=True)
            return web.json_response({
                "error": str(e),
                "loras": []  # Return empty loras array to prevent client-side errors
            }, status=500)
        finally:
            # Clean up the temporary file in the finally block
            if temp_path and os.path.exists(temp_path):
                try:
                    os.unlink(temp_path)
                except Exception as e:
                    logger.error(f"Error deleting temporary file: {e}")
    
    async def analyze_local_image(self, request: web.Request) -> web.Response:
        """Analyze a local image file for recipe metadata"""
        try:
            # Ensure services are initialized 
            await self.init_services()
            
            # Get JSON data from request
            data = await request.json()
            file_path = data.get('path')
            
            if not file_path:
                return web.json_response({
                    'error': 'No file path provided',
                    'loras': []
                }, status=400)
                
            # Normalize file path for cross-platform compatibility
            file_path = os.path.normpath(file_path.strip('"').strip("'"))
            
            # Validate that the file exists
            if not os.path.isfile(file_path):
                return web.json_response({
                    'error': 'File not found',
                    'loras': []
                }, status=404)
                
            # Extract metadata from the image using ExifUtils
            metadata = ExifUtils.extract_image_metadata(file_path)
            
            # If no metadata found, return error
            if not metadata:
                # Get base64 image data
                with open(file_path, "rb") as image_file:
                    image_base64 = base64.b64encode(image_file.read()).decode('utf-8')
                    
                return web.json_response({
                    "error": "No metadata found in this image",
                    "loras": [],  # Return empty loras array to prevent client-side errors
                    "image_base64": image_base64
                }, status=200)
            
            # Use the parser factory to get the appropriate parser
            parser = RecipeParserFactory.create_parser(metadata)

            if parser is None:
                # Get base64 image data
                with open(file_path, "rb") as image_file:
                    image_base64 = base64.b64encode(image_file.read()).decode('utf-8')
                    
                return web.json_response({
                    "error": "No parser found for this image",
                    "loras": [],  # Return empty loras array to prevent client-side errors
                    "image_base64": image_base64
                }, status=200)
            
            # Parse the metadata
            result = await parser.parse_metadata(
                metadata, 
                recipe_scanner=self.recipe_scanner, 
                civitai_client=self.civitai_client
            )
            
            # Add base64 image data to result
            with open(file_path, "rb") as image_file:
                result["image_base64"] = base64.b64encode(image_file.read()).decode('utf-8')
            
            # Check for errors
            if "error" in result and not result.get("loras"):
                return web.json_response(result, status=200)
            
            # Calculate fingerprint from parsed loras
            from ..utils.utils import calculate_recipe_fingerprint
            fingerprint = calculate_recipe_fingerprint(result.get("loras", []))
            
            # Add fingerprint to result
            result["fingerprint"] = fingerprint
            
            # Find matching recipes with the same fingerprint
            matching_recipes = []
            if fingerprint:
                matching_recipes = await self.recipe_scanner.find_recipes_by_fingerprint(fingerprint)
                
            # Add matching recipes to result
            result["matching_recipes"] = matching_recipes
            
            return web.json_response(result)
            
        except Exception as e:
            logger.error(f"Error analyzing local image: {e}", exc_info=True)
            return web.json_response({
                'error': str(e),
                'loras': []  # Return empty loras array to prevent client-side errors
            }, status=500)

    async def save_recipe(self, request: web.Request) -> web.Response:
        """Save a recipe to the recipes folder"""
        try:
            # Ensure services are initialized
            await self.init_services()
            
            reader = await request.multipart()
            
            # Process form data
            image = None
            image_base64 = None
            image_url = None
            name = None
            tags = []
            metadata = None
            
            while True:
                field = await reader.next()
                if field is None:
                    break
                
                if field.name == 'image':
                    # Read image data
                    image_data = b''
                    while True:
                        chunk = await field.read_chunk()
                        if not chunk:
                            break
                        image_data += chunk
                    image = image_data
                    
                elif field.name == 'image_base64':
                    # Get base64 image data
                    image_base64 = await field.text()
                    
                elif field.name == 'image_url':
                    # Get image URL
                    image_url = await field.text()
                    
                elif field.name == 'name':
                    name = await field.text()
                    
                elif field.name == 'tags':
                    tags_text = await field.text()
                    try:
                        tags = json.loads(tags_text)
                    except:
                        tags = []
                    
                elif field.name == 'metadata':
                    metadata_text = await field.text()
                    try:
                        metadata = json.loads(metadata_text)
                    except:
                        metadata = {}
            
            missing_fields = []
            if not name:
                missing_fields.append("name")
            if not metadata:
                missing_fields.append("metadata")
            if missing_fields:
                return web.json_response({"error": f"Missing required fields: {', '.join(missing_fields)}"}, status=400)
            
            # Handle different image sources
            if not image:
                if image_base64:
                    # Convert base64 to binary
                    try:
                        # Remove potential data URL prefix
                        if ',' in image_base64:
                            image_base64 = image_base64.split(',', 1)[1]
                        image = base64.b64decode(image_base64)
                    except Exception as e:
                        return web.json_response({"error": f"Invalid base64 image data: {str(e)}"}, status=400)
                else:
                    return web.json_response({"error": "No image data provided"}, status=400)
            
            # Create recipes directory if it doesn't exist
            recipes_dir = self.recipe_scanner.recipes_dir
            os.makedirs(recipes_dir, exist_ok=True)
            
            # Generate UUID for the recipe
            import uuid
            recipe_id = str(uuid.uuid4())
            
            # Optimize the image (resize and convert to WebP)
            optimized_image, extension = ExifUtils.optimize_image(
                image_data=image,
                target_width=CARD_PREVIEW_WIDTH,
                format='webp',
                quality=85,
                preserve_metadata=True
            )
            
            # Save the optimized image
            image_filename = f"{recipe_id}{extension}"
            image_path = os.path.join(recipes_dir, image_filename)
            with open(image_path, 'wb') as f:
                f.write(optimized_image)
            
            # Create the recipe data structure
            current_time = time.time()
            
            # Format loras data according to the recipe.json format
            loras_data = []
            for lora in metadata.get("loras", []):
                # Modified: Always include deleted LoRAs in the recipe metadata
                # Even if they're marked to be excluded, we still keep their identifying information
                # The exclude flag will only be used to determine if they should be included in recipe syntax
                
                # Convert frontend lora format to recipe format
                lora_entry = {
                    "file_name": lora.get("file_name", "") or os.path.splitext(os.path.basename(lora.get("localPath", "")))[0] if lora.get("localPath") else "",
                    "hash": lora.get("hash", "").lower() if lora.get("hash") else "",
                    "strength": float(lora.get("weight", 1.0)),
                    "modelVersionId": lora.get("id", 0),
                    "modelName": lora.get("name", ""),
                    "modelVersionName": lora.get("version", ""),
                    "isDeleted": lora.get("isDeleted", False),  # Preserve deletion status in saved recipe
                    "exclude": lora.get("exclude", False)  # Add exclude flag to the recipe
                }
                loras_data.append(lora_entry)
            
            # Format gen_params according to the recipe.json format
            gen_params = metadata.get("gen_params", {})
            if not gen_params and "raw_metadata" in metadata:
                # Extract from raw metadata if available
                raw_metadata = metadata.get("raw_metadata", {})
                gen_params = {
                    "prompt": raw_metadata.get("prompt", ""),
                    "negative_prompt": raw_metadata.get("negative_prompt", ""),
                    "checkpoint": raw_metadata.get("checkpoint", {}),
                    "steps": raw_metadata.get("steps", ""),
                    "sampler": raw_metadata.get("sampler", ""),
                    "cfg_scale": raw_metadata.get("cfg_scale", ""),
                    "seed": raw_metadata.get("seed", ""),
                    "size": raw_metadata.get("size", ""),
                    "clip_skip": raw_metadata.get("clip_skip", "")
                }
            
            # Calculate recipe fingerprint 
            from ..utils.utils import calculate_recipe_fingerprint
            fingerprint = calculate_recipe_fingerprint(loras_data)
            
            # Create the recipe data structure
            recipe_data = {
                "id": recipe_id,
                "file_path": image_path,
                "title": name,
                "modified": current_time,
                "created_date": current_time,
                "base_model": metadata.get("base_model", ""),
                "loras": loras_data,
                "gen_params": gen_params,
                "fingerprint": fingerprint
            }
            
            # Add tags if provided
            if tags:
                recipe_data["tags"] = tags
            
            # Add source_path if provided in metadata
            if metadata.get("source_path"):
                recipe_data["source_path"] = metadata.get("source_path")
            
            # Save the recipe JSON
            json_filename = f"{recipe_id}.recipe.json"
            json_path = os.path.join(recipes_dir, json_filename)
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(recipe_data, f, indent=4, ensure_ascii=False)

            # Add recipe metadata to the image
            ExifUtils.append_recipe_metadata(image_path, recipe_data)
            
            # Check for duplicates
            matching_recipes = []
            if fingerprint:
                matching_recipes = await self.recipe_scanner.find_recipes_by_fingerprint(fingerprint)
                # Remove current recipe from matches
                if recipe_id in matching_recipes:
                    matching_recipes.remove(recipe_id)
            
            # Simplified cache update approach
            # Instead of trying to update the cache directly, just set it to None
            # to force a refresh on the next get_cached_data call
            if self.recipe_scanner._cache is not None:
                # Add the recipe to the raw data if the cache exists
                # This is a simple direct update without locks or timeouts
                self.recipe_scanner._cache.raw_data.append(recipe_data)
                # Schedule a background task to resort the cache
                asyncio.create_task(self.recipe_scanner._cache.resort())
                logger.info(f"Added recipe {recipe_id} to cache")
            
            return web.json_response({
                'success': True,
                'recipe_id': recipe_id,
                'image_path': image_path,
                'json_path': json_path,
                'matching_recipes': matching_recipes
            })
            
        except Exception as e:
            logger.error(f"Error saving recipe: {e}", exc_info=True)
            return web.json_response({"error": str(e)}, status=500) 

    async def delete_recipe(self, request: web.Request) -> web.Response:
        """Delete a recipe by ID"""
        try:
            # Ensure services are initialized
            await self.init_services()
            
            recipe_id = request.match_info['recipe_id']
            
            # Get recipes directory
            recipes_dir = self.recipe_scanner.recipes_dir
            if not recipes_dir or not os.path.exists(recipes_dir):
                return web.json_response({"error": "Recipes directory not found"}, status=404)
            
            # Find recipe JSON file
            recipe_json_path = os.path.join(recipes_dir, f"{recipe_id}.recipe.json")
            if not os.path.exists(recipe_json_path):
                return web.json_response({"error": "Recipe not found"}, status=404)
            
            # Load recipe data to get image path
            with open(recipe_json_path, 'r', encoding='utf-8') as f:
                recipe_data = json.load(f)
            
            # Get image path
            image_path = recipe_data.get('file_path')
            
            # Delete recipe JSON file
            os.remove(recipe_json_path)
            logger.info(f"Deleted recipe JSON file: {recipe_json_path}")
            
            # Delete recipe image if it exists
            if image_path and os.path.exists(image_path):
                os.remove(image_path)
                logger.info(f"Deleted recipe image: {image_path}")
            
            # Simplified cache update approach
            if self.recipe_scanner._cache is not None:
                # Remove the recipe from raw_data if it exists
                self.recipe_scanner._cache.raw_data = [
                    r for r in self.recipe_scanner._cache.raw_data 
                    if str(r.get('id', '')) != recipe_id
                ]
                # Schedule a background task to resort the cache
                asyncio.create_task(self.recipe_scanner._cache.resort())
                logger.info(f"Removed recipe {recipe_id} from cache")
            
            return web.json_response({"success": True, "message": "Recipe deleted successfully"})
        except Exception as e:
            logger.error(f"Error deleting recipe: {e}", exc_info=True)
            return web.json_response({"error": str(e)}, status=500) 

    async def get_top_tags(self, request: web.Request) -> web.Response:
        """Get top tags used in recipes"""
        try:
            # Ensure services are initialized
            await self.init_services()
            
            # Get limit parameter with default
            limit = int(request.query.get('limit', '20'))
            
            # Get all recipes from cache
            cache = await self.recipe_scanner.get_cached_data()
            
            # Count tag occurrences
            tag_counts = {}
            for recipe in cache.raw_data:
                if 'tags' in recipe and recipe['tags']:
                    for tag in recipe['tags']:
                        tag_counts[tag] = tag_counts.get(tag, 0) + 1
            
            # Sort tags by count and limit results
            sorted_tags = [{'tag': tag, 'count': count} for tag, count in tag_counts.items()]
            sorted_tags.sort(key=lambda x: x['count'], reverse=True)
            top_tags = sorted_tags[:limit]
            
            return web.json_response({
                'success': True,
                'tags': top_tags
            })
        except Exception as e:
            logger.error(f"Error retrieving top tags: {e}", exc_info=True)
            return web.json_response({
                'success': False,
                'error': str(e)
            }, status=500)
    
    async def get_base_models(self, request: web.Request) -> web.Response:
        """Get base models used in recipes"""
        try:
            # Ensure services are initialized
            await self.init_services()
            
            # Get all recipes from cache
            cache = await self.recipe_scanner.get_cached_data()
            
            # Count base model occurrences
            base_model_counts = {}
            for recipe in cache.raw_data:
                if 'base_model' in recipe and recipe['base_model']:
                    base_model = recipe['base_model']
                    base_model_counts[base_model] = base_model_counts.get(base_model, 0) + 1
            
            # Sort base models by count
            sorted_models = [{'name': model, 'count': count} for model, count in base_model_counts.items()]
            sorted_models.sort(key=lambda x: x['count'], reverse=True)
            
            return web.json_response({
                'success': True,
                'base_models': sorted_models
            })
        except Exception as e:
            logger.error(f"Error retrieving base models: {e}", exc_info=True)
            return web.json_response({
                'success': False,
                'error': str(e)}
            , status=500) 

    async def share_recipe(self, request: web.Request) -> web.Response:
        """Process a recipe image for sharing by adding metadata to EXIF"""
        try:
            # Ensure services are initialized
            await self.init_services()
            
            recipe_id = request.match_info['recipe_id']
            
            # Get all recipes from cache
            cache = await self.recipe_scanner.get_cached_data()
            
            # Find the specific recipe
            recipe = next((r for r in cache.raw_data if str(r.get('id', '')) == recipe_id), None)
            
            if not recipe:
                return web.json_response({"error": "Recipe not found"}, status=404)
            
            # Get the image path
            image_path = recipe.get('file_path')
            if not image_path or not os.path.exists(image_path):
                return web.json_response({"error": "Recipe image not found"}, status=404)
            
            # Create a temporary copy of the image to modify
            import tempfile
            import shutil
            
            # Create temp file with same extension
            ext = os.path.splitext(image_path)[1]
            with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as temp_file:
                temp_path = temp_file.name
            
            # Copy the original image to temp file
            shutil.copy2(image_path, temp_path)
            processed_path = temp_path
            
            # Create a URL for the processed image
            # Use a timestamp to prevent caching
            timestamp = int(time.time())
            url_path = f"/api/recipe/{recipe_id}/share/download?t={timestamp}"
            
            # Store the temp path in a dictionary to serve later
            if not hasattr(self, '_shared_recipes'):
                self._shared_recipes = {}
            
            self._shared_recipes[recipe_id] = {
                'path': processed_path,
                'timestamp': timestamp,
                'expires': time.time() + 300  # Expire after 5 minutes
            }
            
            # Clean up old entries
            self._cleanup_shared_recipes()
            
            return web.json_response({
                'success': True,
                'download_url': url_path,
                'filename': f"recipe_{recipe.get('title', '').replace(' ', '_').lower()}{ext}"
            })
        except Exception as e:
            logger.error(f"Error sharing recipe: {e}", exc_info=True)
            return web.json_response({"error": str(e)}, status=500)

    async def download_shared_recipe(self, request: web.Request) -> web.Response:
        """Serve a processed recipe image for download"""
        try:
            # Ensure services are initialized
            await self.init_services()
            
            recipe_id = request.match_info['recipe_id']
            
            # Check if we have this shared recipe
            if not hasattr(self, '_shared_recipes') or recipe_id not in self._shared_recipes:
                return web.json_response({"error": "Shared recipe not found or expired"}, status=404)
            
            shared_info = self._shared_recipes[recipe_id]
            file_path = shared_info['path']
            
            if not os.path.exists(file_path):
                return web.json_response({"error": "Shared recipe file not found"}, status=404)
            
            # Get recipe to determine filename
            cache = await self.recipe_scanner.get_cached_data()
            recipe = next((r for r in cache.raw_data if str(r.get('id', '')) == recipe_id), None)
            
            # Set filename for download
            filename = f"recipe_{recipe.get('title', '').replace(' ', '_').lower() if recipe else recipe_id}"
            ext = os.path.splitext(file_path)[1]
            download_filename = f"{filename}{ext}"
            
            # Serve the file
            return web.FileResponse(
                file_path,
                headers={
                    'Content-Disposition': f'attachment; filename="{download_filename}"'
                }
            )
        except Exception as e:
            logger.error(f"Error downloading shared recipe: {e}", exc_info=True)
            return web.json_response({"error": str(e)}, status=500)

    def _cleanup_shared_recipes(self):
        """Clean up expired shared recipes"""
        if not hasattr(self, '_shared_recipes'):
            return
        
        current_time = time.time()
        expired_ids = [rid for rid, info in self._shared_recipes.items() 
                      if current_time > info.get('expires', 0)]
        
        for rid in expired_ids:
            try:
                # Delete the temporary file
                file_path = self._shared_recipes[rid]['path']
                if os.path.exists(file_path):
                    os.unlink(file_path)
                
                # Remove from dictionary
                del self._shared_recipes[rid]
            except Exception as e:
                logger.error(f"Error cleaning up shared recipe {rid}: {e}")

    async def save_recipe_from_widget(self, request: web.Request) -> web.Response:
        """Save a recipe from the LoRAs widget"""
        try:
            # Ensure services are initialized
            await self.init_services()
            
            # Get metadata using the metadata collector instead of workflow parsing
            raw_metadata = get_metadata()
            metadata_dict = MetadataProcessor.to_dict(raw_metadata)
            
            # Check if we have valid metadata
            if not metadata_dict:
                return web.json_response({"error": "No generation metadata found"}, status=400)
            
            # Get the most recent image from metadata registry instead of temp directory
            if not standalone_mode:
                metadata_registry = MetadataRegistry()
                latest_image = metadata_registry.get_first_decoded_image()
            else:
                latest_image = None
            
            if latest_image is None:
                return web.json_response({"error": "No recent images found to use for recipe. Try generating an image first."}, status=400)
            
            # Convert the image data to bytes - handle tuple and tensor cases
            logger.debug(f"Image type: {type(latest_image)}")
            
            try:
                # Handle the tuple case first
                if isinstance(latest_image, tuple):
                    # Extract the tensor from the tuple
                    if len(latest_image) > 0:
                        tensor_image = latest_image[0]
                    else:
                        return web.json_response({"error": "Empty image tuple received"}, status=400)
                else:
                    tensor_image = latest_image
                
                # Get the shape info for debugging
                if hasattr(tensor_image, 'shape'):
                    shape_info = tensor_image.shape
                    logger.debug(f"Tensor shape: {shape_info}, dtype: {tensor_image.dtype}")
                
                import torch
                
                # Convert tensor to numpy array
                if isinstance(tensor_image, torch.Tensor):
                    image_np = tensor_image.cpu().numpy()
                else:
                    image_np = np.array(tensor_image)
                
                # Handle different tensor shapes
                # Case: (1, 1, H, W, 3) or (1, H, W, 3) - batch or multi-batch
                if len(image_np.shape) > 3:
                    # Remove batch dimensions until we get to (H, W, 3)
                    while len(image_np.shape) > 3:
                        image_np = image_np[0]
                
                # If values are in [0, 1] range, convert to [0, 255]
                if image_np.dtype == np.float32 or image_np.dtype == np.float64:
                    if image_np.max() <= 1.0:
                        image_np = (image_np * 255).astype(np.uint8)
                
                # Ensure image is in the right format (HWC with RGB channels)
                if len(image_np.shape) == 3 and image_np.shape[2] == 3:
                    pil_image = Image.fromarray(image_np)
                    img_byte_arr = io.BytesIO()
                    pil_image.save(img_byte_arr, format='PNG')
                    image = img_byte_arr.getvalue()
                else:
                    return web.json_response({"error": f"Cannot handle this data shape: {image_np.shape}, {image_np.dtype}"}, status=400)
            except Exception as e:
                logger.error(f"Error processing image data: {str(e)}", exc_info=True)
                return web.json_response({"error": f"Error processing image: {str(e)}"}, status=400)
            
            # Get the lora stack from the metadata
            lora_stack = metadata_dict.get("loras", "")
            
            # Parse the lora stack format: "<lora:name:strength> <lora:name2:strength2> ..."
            import re
            lora_matches = re.findall(r'<lora:([^:]+):([^>]+)>', lora_stack)
            
            # Check if any loras were found
            if not lora_matches:
                return web.json_response({"error": "No LoRAs found in the generation metadata"}, status=400)
            
            # Generate recipe name from the first 3 loras (or less if fewer are available)
            loras_for_name = lora_matches[:3]  # Take at most 3 loras for the name
            
            recipe_name_parts = []
            for lora_name, lora_strength in loras_for_name:
                # Get the basename without path or extension
                basename = os.path.basename(lora_name)
                basename = os.path.splitext(basename)[0]
                recipe_name_parts.append(f"{basename}:{lora_strength}")
            
            recipe_name = " ".join(recipe_name_parts)
            
            # Create recipes directory if it doesn't exist
            recipes_dir = self.recipe_scanner.recipes_dir
            os.makedirs(recipes_dir, exist_ok=True)
            
            # Generate UUID for the recipe
            import uuid
            recipe_id = str(uuid.uuid4())
            
            # Optimize the image (resize and convert to WebP)
            optimized_image, extension = ExifUtils.optimize_image(
                image_data=image,
                target_width=CARD_PREVIEW_WIDTH,
                format='webp',
                quality=85,
                preserve_metadata=True
            )
            
            # Save the optimized image
            image_filename = f"{recipe_id}{extension}"
            image_path = os.path.join(recipes_dir, image_filename)
            with open(image_path, 'wb') as f:
                f.write(optimized_image)
            
            # Format loras data from the lora stack
            loras_data = []
            
            for lora_name, lora_strength in lora_matches:
                try:
                    # Get lora info from scanner
                    lora_info = await self.recipe_scanner._lora_scanner.get_model_info_by_name(lora_name)
                    
                    # Create lora entry
                    lora_entry = {
                        "file_name": lora_name,
                        "hash": lora_info.get("sha256", "").lower() if lora_info else "",
                        "strength": float(lora_strength),
                        "modelVersionId": lora_info.get("civitai", {}).get("id", 0) if lora_info else 0,
                        "modelName": lora_info.get("civitai", {}).get("model", {}).get("name", "") if lora_info else lora_name,
                        "modelVersionName": lora_info.get("civitai", {}).get("name", "") if lora_info else "",
                        "isDeleted": False
                    }
                    loras_data.append(lora_entry)
                except Exception as e:
                    logger.warning(f"Error processing LoRA {lora_name}: {e}")
            
            # Get base model from lora scanner for the available loras
            base_model_counts = {}
            for lora in loras_data:
                lora_info = await self.recipe_scanner._lora_scanner.get_model_info_by_name(lora.get("file_name", ""))
                if lora_info and "base_model" in lora_info:
                    base_model = lora_info["base_model"]
                    base_model_counts[base_model] = base_model_counts.get(base_model, 0) + 1
            
            # Get most common base model
            most_common_base_model = ""
            if base_model_counts:
                most_common_base_model = max(base_model_counts.items(), key=lambda x: x[1])[0]
            
            # Create the recipe data structure
            recipe_data = {
                "id": recipe_id,
                "file_path": image_path,
                "title": recipe_name,  # Use generated recipe name
                "modified": time.time(),
                "created_date": time.time(),
                "base_model": most_common_base_model,
                "loras": loras_data,
                "checkpoint": metadata_dict.get("checkpoint", ""),
                "gen_params": {key: value for key, value in metadata_dict.items() 
                               if key not in ['checkpoint', 'loras']},
                "loras_stack": lora_stack  # Include the original lora stack string
            }
            
            # Save the recipe JSON
            json_filename = f"{recipe_id}.recipe.json"
            json_path = os.path.join(recipes_dir, json_filename)
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(recipe_data, f, indent=4, ensure_ascii=False)

            # Add recipe metadata to the image
            ExifUtils.append_recipe_metadata(image_path, recipe_data)
            
            # Update cache
            if self.recipe_scanner._cache is not None:
                # Add the recipe to the raw data if the cache exists
                self.recipe_scanner._cache.raw_data.append(recipe_data)
                # Schedule a background task to resort the cache
                asyncio.create_task(self.recipe_scanner._cache.resort())
                logger.info(f"Added recipe {recipe_id} to cache")
            
            return web.json_response({
                'success': True,
                'recipe_id': recipe_id,
                'image_path': image_path,
                'json_path': json_path,
                'recipe_name': recipe_name  # Include the generated recipe name in the response
            })
            
        except Exception as e:
            logger.error(f"Error saving recipe from widget: {e}", exc_info=True)
            return web.json_response({"error": str(e)}, status=500)

    async def get_recipe_syntax(self, request: web.Request) -> web.Response:
        """Generate recipe syntax for LoRAs in the recipe, looking up proper file names using hash_index"""
        try:
            # Ensure services are initialized
            await self.init_services()
            
            recipe_id = request.match_info['recipe_id']
            
            # Get all recipes from cache
            cache = await self.recipe_scanner.get_cached_data()
            
            # Find the specific recipe
            recipe = next((r for r in cache.raw_data if str(r.get('id', '')) == recipe_id), None)
            
            if not recipe:
                return web.json_response({"error": "Recipe not found"}, status=404)
            
            # Get the loras from the recipe
            loras = recipe.get('loras', [])
            
            if not loras:
                return web.json_response({"error": "No LoRAs found in this recipe"}, status=400)
            
            # Generate recipe syntax for all LoRAs that:
            # 1. Are in the library (not deleted) OR
            # 2. Are deleted but not marked for exclusion
            lora_syntax_parts = []
            
            # Access the hash_index from lora_scanner
            hash_index = self.recipe_scanner._lora_scanner._hash_index
            
            for lora in loras:
                # Skip loras that are deleted AND marked for exclusion
                if lora.get("isDeleted", False):
                    continue

                if not self.recipe_scanner._lora_scanner.has_hash(lora.get("hash", "")):
                    continue
                
                # Get the strength
                strength = lora.get("strength", 1.0)
                
                # Try to find the actual file name for this lora
                file_name = None
                hash_value = lora.get("hash", "").lower()
                
                if hash_value and hasattr(hash_index, "_hash_to_path"):
                    # Look up the file path from the hash
                    file_path = hash_index._hash_to_path.get(hash_value)
                    
                    if file_path:
                        # Extract the file name without extension from the path
                        file_name = os.path.splitext(os.path.basename(file_path))[0]
                
                # If hash lookup failed, fall back to modelVersionId lookup
                if not file_name and lora.get("modelVersionId"):
                    # Search for files with matching modelVersionId
                    all_loras = await self.recipe_scanner._lora_scanner.get_cached_data()
                    for cached_lora in all_loras.raw_data:
                        if not cached_lora.get("civitai"):
                            continue
                        if cached_lora.get("civitai", {}).get("id") == lora.get("modelVersionId"):
                            file_name = os.path.splitext(os.path.basename(cached_lora["path"]))[0]
                            break
                
                # If all lookups failed, use the file_name from the recipe
                if not file_name:
                    file_name = lora.get("file_name", "unknown-lora")
                
                # Add to syntax parts
                lora_syntax_parts.append(f"<lora:{file_name}:{strength}>")
            
            # Join the LoRA syntax parts
            lora_syntax = " ".join(lora_syntax_parts)
            
            return web.json_response({
                'success': True,
                'syntax': lora_syntax
            })
        except Exception as e:
            logger.error(f"Error generating recipe syntax: {e}", exc_info=True)
            return web.json_response({"error": str(e)}, status=500)

    async def update_recipe(self, request: web.Request) -> web.Response:
        """Update recipe metadata (name and tags)"""
        try:
            # Ensure services are initialized
            await self.init_services()
            
            recipe_id = request.match_info['recipe_id']
            data = await request.json()
            
            # Validate required fields
            if 'title' not in data and 'tags' not in data and 'source_path' not in data and 'preview_nsfw_level' not in data:
                return web.json_response({
                    "error": "At least one field to update must be provided (title or tags or source_path or preview_nsfw_level)"
                }, status=400)
            
            # Use the recipe scanner's update method
            success = await self.recipe_scanner.update_recipe_metadata(recipe_id, data)
            
            if not success:
                return web.json_response({"error": "Recipe not found or update failed"}, status=404)
            
            return web.json_response({
                "success": True,
                "recipe_id": recipe_id,
                "updates": data
            })
        except Exception as e:
            logger.error(f"Error updating recipe: {e}", exc_info=True)
            return web.json_response({"error": str(e)}, status=500)

    async def reconnect_lora(self, request: web.Request) -> web.Response:
        """Reconnect a deleted LoRA in a recipe to a local LoRA file"""
        try:
            # Ensure services are initialized
            await self.init_services()
            
            # Parse request data
            data = await request.json()
            
            # Validate required fields
            required_fields = ['recipe_id', 'lora_index', 'target_name']
            for field in required_fields:
                if field not in data:
                    return web.json_response({
                        "error": f"Missing required field: {field}"
                    }, status=400)
            
            recipe_id = data['recipe_id']
            lora_index = int(data['lora_index'])
            target_name = data['target_name']
            
            # Get recipe scanner
            scanner = self.recipe_scanner
            lora_scanner = scanner._lora_scanner
            
            # Check if recipe exists
            recipe_path = os.path.join(scanner.recipes_dir, f"{recipe_id}.recipe.json")
            if not os.path.exists(recipe_path):
                return web.json_response({"error": "Recipe not found"}, status=404)
                
            # Find target LoRA by name
            target_lora = await lora_scanner.get_model_info_by_name(target_name)
            if not target_lora:
                return web.json_response({"error": f"Local LoRA not found with name: {target_name}"}, status=404)
                
            # Load recipe data
            with open(recipe_path, 'r', encoding='utf-8') as f:
                recipe_data = json.load(f)

            lora = recipe_data.get("loras", [])[lora_index] if lora_index < len(recipe_data.get('loras', [])) else None

            if lora is None:
                return web.json_response({"error": "LoRA index out of range in recipe"}, status=404)

            # Update LoRA data
            lora['isDeleted'] = False
            lora['exclude'] = False
            lora['file_name'] = target_name
            
            # Update with information from the target LoRA
            if 'sha256' in target_lora:
                lora['hash'] = target_lora['sha256'].lower()
            if target_lora.get("civitai"):
                lora['modelName'] = target_lora['civitai']['model']['name']
                lora['modelVersionName'] = target_lora['civitai']['name']
                lora['modelVersionId'] = target_lora['civitai']['id']
            
            updated_lora = dict(lora)  # Make a copy for response

            # Recalculate recipe fingerprint after updating LoRA
            from ..utils.utils import calculate_recipe_fingerprint
            recipe_data['fingerprint'] = calculate_recipe_fingerprint(recipe_data.get('loras', []))
                
            # Save updated recipe
            with open(recipe_path, 'w', encoding='utf-8') as f:
                json.dump(recipe_data, f, indent=4, ensure_ascii=False)

            updated_lora['inLibrary'] = True
            updated_lora['preview_url'] = config.get_preview_static_url(target_lora['preview_url'])
            updated_lora['localPath'] = target_lora['file_path']
                
            # Update in cache if it exists
            if scanner._cache is not None:
                for cache_item in scanner._cache.raw_data:
                    if cache_item.get('id') == recipe_id:
                        # Replace loras array with updated version
                        cache_item['loras'] = recipe_data['loras']
                        # Update fingerprint in cache
                        cache_item['fingerprint'] = recipe_data['fingerprint']
                        
                        # Resort the cache
                        asyncio.create_task(scanner._cache.resort())
                        break
                        
            # Update EXIF metadata if image exists
            image_path = recipe_data.get('file_path')
            if image_path and os.path.exists(image_path):
                from ..utils.exif_utils import ExifUtils
                ExifUtils.append_recipe_metadata(image_path, recipe_data)
            
            # Find other recipes with the same fingerprint
            matching_recipes = []
            if 'fingerprint' in recipe_data:
                matching_recipes = await scanner.find_recipes_by_fingerprint(recipe_data['fingerprint'])
                # Remove current recipe from matches
                if recipe_id in matching_recipes:
                    matching_recipes.remove(recipe_id)
                
            return web.json_response({
                "success": True,
                "recipe_id": recipe_id,
                "updated_lora": updated_lora,
                "matching_recipes": matching_recipes
            })
            
        except Exception as e:
            logger.error(f"Error reconnecting LoRA: {e}", exc_info=True)
            return web.json_response({"error": str(e)}, status=500)

    async def get_recipes_for_lora(self, request: web.Request) -> web.Response:
        """Get recipes that use a specific Lora"""
        try:
            # Ensure services are initialized
            await self.init_services()
            
            lora_hash = request.query.get('hash')
            
            # Hash is required
            if not lora_hash:
                return web.json_response({'success': False, 'error': 'Lora hash is required'}, status=400)
            
            # Log the search parameters
            logger.debug(f"Getting recipes for Lora by hash: {lora_hash}")
            
            # Get all recipes from cache
            cache = await self.recipe_scanner.get_cached_data()
            
            # Filter recipes that use this Lora by hash
            matching_recipes = []
            for recipe in cache.raw_data:
                # Check if any of the recipe's loras match this hash
                loras = recipe.get('loras', [])
                for lora in loras:
                    if lora.get('hash', '').lower() == lora_hash.lower():
                        matching_recipes.append(recipe)
                        break  # No need to check other loras in this recipe
            
            # Process the recipes similar to get_paginated_data to ensure all needed data is available
            for recipe in matching_recipes:
                # Add inLibrary information for each lora
                if 'loras' in recipe:
                    for lora in recipe['loras']:
                        if 'hash' in lora and lora['hash']:
                            lora['inLibrary'] = self.recipe_scanner._lora_scanner.has_hash(lora['hash'].lower())
                            lora['preview_url'] = self.recipe_scanner._lora_scanner.get_preview_url_by_hash(lora['hash'].lower())
                            lora['localPath'] = self.recipe_scanner._lora_scanner.get_path_by_hash(lora['hash'].lower())
                
                # Ensure file_url is set (needed by frontend)
                if 'file_path' in recipe:
                    recipe['file_url'] = self._format_recipe_file_url(recipe['file_path'])
                else:
                    recipe['file_url'] = '/loras_static/images/no-preview.png'
            
            return web.json_response({'success': True, 'recipes': matching_recipes})
        except Exception as e:
            logger.error(f"Error getting recipes for Lora: {str(e)}")
            return web.json_response({'success': False, 'error': str(e)}, status=500)

    async def scan_recipes(self, request: web.Request) -> web.Response:
        """API endpoint for scanning and rebuilding the recipe cache"""
        try:
            # Ensure services are initialized
            await self.init_services()
            
            # Force refresh the recipe cache
            logger.info("Manually triggering recipe cache rebuild")
            await self.recipe_scanner.get_cached_data(force_refresh=True)
            
            return web.json_response({
                'success': True,
                'message': 'Recipe cache refreshed successfully'
            })
        except Exception as e:
            logger.error(f"Error refreshing recipe cache: {e}", exc_info=True)
            return web.json_response({
                'success': False,
                'error': str(e)
            }, status=500)

    async def find_duplicates(self, request: web.Request) -> web.Response:
        """Find all duplicate recipes based on fingerprints"""
        try:
            # Ensure services are initialized
            await self.init_services()
            
            # Get all duplicate recipes
            duplicate_groups = await self.recipe_scanner.find_all_duplicate_recipes()
            
            # Create response data with additional recipe information
            response_data = []
            
            for fingerprint, recipe_ids in duplicate_groups.items():
                # Skip groups with only one recipe (not duplicates)
                if len(recipe_ids) <= 1:
                    continue
                    
                # Get recipe details for each recipe in the group
                recipes = []
                for recipe_id in recipe_ids:
                    recipe = await self.recipe_scanner.get_recipe_by_id(recipe_id)
                    if recipe:
                        # Add only needed fields to keep response size manageable
                        recipes.append({
                            'id': recipe.get('id'),
                            'title': recipe.get('title'),
                            'file_url': recipe.get('file_url') or self._format_recipe_file_url(recipe.get('file_path', '')),
                            'modified': recipe.get('modified'),
                            'created_date': recipe.get('created_date'),
                            'lora_count': len(recipe.get('loras', [])),
                        })
                        
                # Only include groups with at least 2 valid recipes
                if len(recipes) >= 2:
                    # Sort recipes by modified date (newest first)
                    recipes.sort(key=lambda x: x.get('modified', 0), reverse=True)
                    
                    response_data.append({
                        'fingerprint': fingerprint,
                        'count': len(recipes),
                        'recipes': recipes
                    })
            
            # Sort groups by count (highest first)
            response_data.sort(key=lambda x: x['count'], reverse=True)
            
            return web.json_response({
                'success': True,
                'duplicate_groups': response_data
            })
            
        except Exception as e:
            logger.error(f"Error finding duplicate recipes: {e}", exc_info=True)
            return web.json_response({
                'success': False,
                'error': str(e)
            }, status=500)
    
    async def bulk_delete(self, request: web.Request) -> web.Response:
        """Delete multiple recipes by ID"""
        try:
            # Ensure services are initialized
            await self.init_services()
            
            # Parse request data
            data = await request.json()
            recipe_ids = data.get('recipe_ids', [])
            
            if not recipe_ids:
                return web.json_response({
                    'success': False,
                    'error': 'No recipe IDs provided'
                }, status=400)
                
            # Get recipes directory
            recipes_dir = self.recipe_scanner.recipes_dir
            if not recipes_dir or not os.path.exists(recipes_dir):
                return web.json_response({
                    'success': False,
                    'error': 'Recipes directory not found'
                }, status=404)
            
            # Track deleted and failed recipes
            deleted_recipes = []
            failed_recipes = []
            
            # Process each recipe ID
            for recipe_id in recipe_ids:
                # Find recipe JSON file
                recipe_json_path = os.path.join(recipes_dir, f"{recipe_id}.recipe.json")
                
                if not os.path.exists(recipe_json_path):
                    failed_recipes.append({
                        'id': recipe_id,
                        'reason': 'Recipe not found'
                    })
                    continue
                
                try:
                    # Load recipe data to get image path
                    with open(recipe_json_path, 'r', encoding='utf-8') as f:
                        recipe_data = json.load(f)
                    
                    # Get image path
                    image_path = recipe_data.get('file_path')
                    
                    # Delete recipe JSON file
                    os.remove(recipe_json_path)
                    
                    # Delete recipe image if it exists
                    if image_path and os.path.exists(image_path):
                        os.remove(image_path)
                        
                    deleted_recipes.append(recipe_id)
                    
                except Exception as e:
                    failed_recipes.append({
                        'id': recipe_id,
                        'reason': str(e)
                    })
            
            # Update cache if any recipes were deleted
            if deleted_recipes and self.recipe_scanner._cache is not None:
                # Remove deleted recipes from raw_data
                self.recipe_scanner._cache.raw_data = [
                    r for r in self.recipe_scanner._cache.raw_data 
                    if r.get('id') not in deleted_recipes
                ]
                # Resort the cache
                asyncio.create_task(self.recipe_scanner._cache.resort())
                logger.info(f"Removed {len(deleted_recipes)} recipes from cache")
            
            return web.json_response({
                'success': True,
                'deleted': deleted_recipes,
                'failed': failed_recipes,
                'total_deleted': len(deleted_recipes),
                'total_failed': len(failed_recipes)
            })
            
        except Exception as e:
            logger.error(f"Error performing bulk delete: {e}", exc_info=True)
            return web.json_response({
                'success': False,
                'error': str(e)
            }, status=500)
