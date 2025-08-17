import piexif
import json
import logging
from typing import Optional
from io import BytesIO
import os
from PIL import Image

logger = logging.getLogger(__name__)

class ExifUtils:
    """Utility functions for working with EXIF data in images"""
    
    @staticmethod
    def extract_image_metadata(image_path: str) -> Optional[str]:
        """Extract metadata from image including UserComment or parameters field
        
        Args:
            image_path (str): Path to the image file
            
        Returns:
            Optional[str]: Extracted metadata or None if not found
        """
        try:
            # First try to open the image
            with Image.open(image_path) as img:
                # Method 1: Check for parameters in image info
                if hasattr(img, 'info') and 'parameters' in img.info:
                    return img.info['parameters']
                
                # Method 2: Check EXIF UserComment field
                if img.format not in ['JPEG', 'TIFF', 'WEBP']:
                    # For non-JPEG/TIFF/WEBP images, try to get EXIF through PIL
                    exif = img.getexif()
                    if exif and piexif.ExifIFD.UserComment in exif:
                        user_comment = exif[piexif.ExifIFD.UserComment]
                        if isinstance(user_comment, bytes):
                            if user_comment.startswith(b'UNICODE\0'):
                                return user_comment[8:].decode('utf-16be')
                            return user_comment.decode('utf-8', errors='ignore')
                        return user_comment
                
                # For JPEG/TIFF/WEBP, use piexif
                try:
                    exif_dict = piexif.load(image_path)
                    
                    if piexif.ExifIFD.UserComment in exif_dict.get('Exif', {}):
                        user_comment = exif_dict['Exif'][piexif.ExifIFD.UserComment]
                        if isinstance(user_comment, bytes):
                            if user_comment.startswith(b'UNICODE\0'):
                                user_comment = user_comment[8:].decode('utf-16be')
                            else:
                                user_comment = user_comment.decode('utf-8', errors='ignore')
                        return user_comment
                except Exception as e:
                    logger.debug(f"Error loading EXIF data: {e}")
                
                # Method 3: Check PNG metadata for workflow info (for ComfyUI images)
                if img.format == 'PNG':
                    # Look for workflow or prompt metadata in PNG chunks
                    for key in img.info:
                        if key in ['workflow', 'prompt', 'parameters']:
                            return img.info[key]
                
                return None
                
        except Exception as e:
            logger.error(f"Error extracting image metadata: {e}", exc_info=True)
            return None
    
    @staticmethod
    def update_image_metadata(image_path: str, metadata: str) -> str:
        """Update metadata in image's EXIF data or parameters fields
        
        Args:
            image_path (str): Path to the image file
            metadata (str): Metadata string to save
            
        Returns:
            str: Path to the updated image
        """
        try:
            # Load the image and check its format
            with Image.open(image_path) as img:
                img_format = img.format
                
                # For PNG, try to update parameters directly
                if img_format == 'PNG':
                    # We'll save with parameters in the PNG info
                    info_dict = {'parameters': metadata}
                    img.save(image_path, format='PNG', pnginfo=info_dict)
                    return image_path
                
                # For WebP format, use PIL's exif parameter directly
                elif img_format == 'WEBP':
                    exif_dict = {'Exif': {piexif.ExifIFD.UserComment: b'UNICODE\0' + metadata.encode('utf-16be')}}
                    exif_bytes = piexif.dump(exif_dict)
                    
                    # Save with the exif data
                    img.save(image_path, format='WEBP', exif=exif_bytes, quality=85)
                    return image_path
                
                # For other formats, use standard EXIF approach
                else:
                    try:
                        exif_dict = piexif.load(img.info.get('exif', b''))
                    except:
                        exif_dict = {'0th':{}, 'Exif':{}, 'GPS':{}, 'Interop':{}, '1st':{}}
                    
                    # If no Exif dictionary exists, create one
                    if 'Exif' not in exif_dict:
                        exif_dict['Exif'] = {}
                    
                    # Update the UserComment field - use UNICODE format
                    unicode_bytes = metadata.encode('utf-16be')
                    metadata_bytes = b'UNICODE\0' + unicode_bytes
                    
                    exif_dict['Exif'][piexif.ExifIFD.UserComment] = metadata_bytes
                    
                    # Convert EXIF dict back to bytes
                    exif_bytes = piexif.dump(exif_dict)
                    
                    # Save the image with updated EXIF data
                    img.save(image_path, exif=exif_bytes)
                    
            return image_path
        except Exception as e:
            logger.error(f"Error updating metadata in {image_path}: {e}")
            return image_path
            
    @staticmethod
    def append_recipe_metadata(image_path, recipe_data) -> str:
        """Append recipe metadata to an image's EXIF data"""
        try:
            # First, extract existing metadata
            metadata = ExifUtils.extract_image_metadata(image_path)
            
            # Check if there's already recipe metadata
            if metadata:
                # Remove any existing recipe metadata
                metadata = ExifUtils.remove_recipe_metadata(metadata)
            
            # Prepare simplified loras data
            simplified_loras = []
            for lora in recipe_data.get("loras", []):
                simplified_lora = {
                    "file_name": lora.get("file_name", ""),
                    "hash": lora.get("hash", "").lower() if lora.get("hash") else "",
                    "strength": float(lora.get("strength", 1.0)),
                    "modelVersionId": lora.get("modelVersionId", 0),
                    "modelName": lora.get("modelName", ""),
                    "modelVersionName": lora.get("modelVersionName", ""),
                }
                simplified_loras.append(simplified_lora)            
            
            # Create recipe metadata JSON
            recipe_metadata = {
                'title': recipe_data.get('title', ''),
                'base_model': recipe_data.get('base_model', ''),
                'loras': simplified_loras,
                'gen_params': recipe_data.get('gen_params', {}),
                'tags': recipe_data.get('tags', [])
            }
            
            # Convert to JSON string
            recipe_metadata_json = json.dumps(recipe_metadata)
            
            # Create the recipe metadata marker
            recipe_metadata_marker = f"Recipe metadata: {recipe_metadata_json}"
            
            # Append to existing metadata or create new one
            new_metadata = f"{metadata} \n {recipe_metadata_marker}" if metadata else recipe_metadata_marker
            
            # Write back to the image
            return ExifUtils.update_image_metadata(image_path, new_metadata)
        except Exception as e:
            logger.error(f"Error appending recipe metadata: {e}", exc_info=True)
            return image_path

    @staticmethod
    def remove_recipe_metadata(user_comment):
        """Remove recipe metadata from user comment"""
        if not user_comment:
            return ""
        
        # Find the recipe metadata marker
        recipe_marker_index = user_comment.find("Recipe metadata: ")
        if recipe_marker_index == -1:
            return user_comment
        
        # If recipe metadata is not at the start, remove the preceding ", "
        if recipe_marker_index >= 2 and user_comment[recipe_marker_index-2:recipe_marker_index] == ", ":
            recipe_marker_index -= 2
        
        # Remove the recipe metadata part
        # First, find where the metadata ends (next line or end of string)
        next_line_index = user_comment.find("\n", recipe_marker_index)
        if next_line_index == -1:
            # Metadata is at the end of the string
            return user_comment[:recipe_marker_index].rstrip()
        else:
            # Metadata is in the middle of the string
            return user_comment[:recipe_marker_index] + user_comment[next_line_index:]
            
    @staticmethod
    def optimize_image(image_data, target_width=250, format='webp', quality=85, preserve_metadata=False):
        """
        Optimize an image by resizing and converting to WebP format
        
        Args:
            image_data: Binary image data or path to image file
            target_width: Width to resize the image to (preserves aspect ratio)
            format: Output format (default: webp)
            quality: Output quality (0-100)
            preserve_metadata: Whether to preserve EXIF metadata
            
        Returns:
            Tuple of (optimized_image_data, extension)
        """
        try:
            # First validate the image data is usable
            img = None
            if isinstance(image_data, str) and os.path.exists(image_data):
                # It's a file path - validate file
                try:
                    with Image.open(image_data) as test_img:
                        # Verify the image can be fully loaded by accessing its size
                        width, height = test_img.size
                    # If we got here, the image is valid
                    img = Image.open(image_data)
                except (IOError, OSError) as e:
                    logger.error(f"Invalid or corrupt image file: {image_data}: {e}")
                    raise ValueError(f"Cannot process corrupt image: {e}")
            else:
                # It's binary data - validate data
                try:
                    with BytesIO(image_data) as temp_buf:
                        test_img = Image.open(temp_buf)
                        # Verify the image can be fully loaded
                        width, height = test_img.size
                    # If successful, reopen for processing
                    img = Image.open(BytesIO(image_data))
                except Exception as e:
                    logger.error(f"Invalid binary image data: {e}")
                    raise ValueError(f"Cannot process corrupt image data: {e}")

            # Extract metadata if needed and valid
            metadata = None
            if preserve_metadata:
                try:
                    if isinstance(image_data, str) and os.path.exists(image_data):
                        # For file path, extract directly
                        metadata = ExifUtils.extract_image_metadata(image_data)
                    else:
                        # For binary data, save to temp file first
                        import tempfile
                        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as temp_file:
                            temp_path = temp_file.name
                            temp_file.write(image_data)
                        try:
                            metadata = ExifUtils.extract_image_metadata(temp_path)
                        except Exception as e:
                            logger.warning(f"Failed to extract metadata from temp file: {e}")
                        finally:
                            # Clean up temp file
                            try:
                                os.unlink(temp_path)
                            except Exception:
                                pass
                except Exception as e:
                    logger.warning(f"Failed to extract metadata, continuing without it: {e}")
                    # Continue without metadata

            # Calculate new height to maintain aspect ratio
            width, height = img.size
            new_height = int(height * (target_width / width))
            
            # Resize the image with error handling
            try:
                resized_img = img.resize((target_width, new_height), Image.LANCZOS)
            except Exception as e:
                logger.error(f"Failed to resize image: {e}")
                # Return original image if resize fails
                return image_data, '.jpg' if not isinstance(image_data, str) else os.path.splitext(image_data)[1]
            
            # Save to BytesIO in the specified format
            output = BytesIO()
            
            # Set format and extension
            if format.lower() == 'webp':
                save_format, extension = 'WEBP', '.webp'
            elif format.lower() in ('jpg', 'jpeg'):
                save_format, extension = 'JPEG', '.jpg'
            elif format.lower() == 'png':
                save_format, extension = 'PNG', '.png'
            else:
                save_format, extension = 'WEBP', '.webp'
            
            # Save with error handling
            try:
                if save_format == 'PNG':
                    resized_img.save(output, format=save_format, optimize=True)
                else:
                    resized_img.save(output, format=save_format, quality=quality)
            except Exception as e:
                logger.error(f"Failed to save optimized image: {e}")
                # Return original image if save fails
                return image_data, '.jpg' if not isinstance(image_data, str) else os.path.splitext(image_data)[1]
            
            # Get the optimized image data
            optimized_data = output.getvalue()
            
            # Handle metadata preservation if requested and available
            if preserve_metadata and metadata:
                try:
                    if save_format == 'WEBP':
                        # For WebP format, directly save with metadata
                        try:
                            output_with_metadata = BytesIO()
                            exif_dict = {'Exif': {piexif.ExifIFD.UserComment: b'UNICODE\0' + metadata.encode('utf-16be')}}
                            exif_bytes = piexif.dump(exif_dict)
                            resized_img.save(output_with_metadata, format='WEBP', exif=exif_bytes, quality=quality)
                            optimized_data = output_with_metadata.getvalue()
                        except Exception as e:
                            logger.warning(f"Failed to add metadata to WebP, continuing without it: {e}")
                    else:
                        # For other formats, use temporary file
                        import tempfile
                        with tempfile.NamedTemporaryFile(suffix=extension, delete=False) as temp_file:
                            temp_path = temp_file.name
                            temp_file.write(optimized_data)
                        
                        try:
                            # Add metadata
                            ExifUtils.update_image_metadata(temp_path, metadata)
                            # Read back the file
                            with open(temp_path, 'rb') as f:
                                optimized_data = f.read()
                        except Exception as e:
                            logger.warning(f"Failed to add metadata to image, continuing without it: {e}")
                        finally:
                            # Clean up temp file
                            try:
                                os.unlink(temp_path)
                            except Exception:
                                pass
                except Exception as e:
                    logger.warning(f"Failed to preserve metadata: {e}, continuing with unmodified output")
            
            return optimized_data, extension
            
        except Exception as e:
            logger.error(f"Error optimizing image: {e}", exc_info=True)
            # Return original data if optimization completely fails
            if isinstance(image_data, str) and os.path.exists(image_data):
                try:
                    with open(image_data, 'rb') as f:
                        return f.read(), os.path.splitext(image_data)[1]
                except Exception:
                    return image_data, '.jpg'  # Last resort fallback
            return image_data, '.jpg'