import os
import json
import sys
import time
import asyncio
import logging
import datetime
import shutil
from typing import Dict, Set

from ..config import config
from ..services.service_registry import ServiceRegistry

# Check if running in standalone mode
standalone_mode = 'nodes' not in sys.modules

if not standalone_mode:
    from ..metadata_collector.metadata_registry import MetadataRegistry
    from ..metadata_collector.constants import MODELS, LORAS

logger = logging.getLogger(__name__)

class UsageStats:
    """Track usage statistics for models and save to JSON"""
    
    _instance = None
    _lock = asyncio.Lock()  # For thread safety
    
    # Default stats file name
    STATS_FILENAME = "lora_manager_stats.json"
    BACKUP_SUFFIX = ".backup"
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
            
        # Initialize stats storage
        self.stats = {
            "checkpoints": {},  # sha256 -> { total: count, history: { date: count } }
            "loras": {},        # sha256 -> { total: count, history: { date: count } }
            "total_executions": 0,
            "last_save_time": 0
        }
        
        # Queue for prompt_ids to process
        self.pending_prompt_ids = set()
        
        # Load existing stats if available
        self._stats_file_path = self._get_stats_file_path()
        self._load_stats()
        
        # Save interval in seconds
        self.save_interval = 90  # 1.5 minutes
        
        # Start background task to process queued prompt_ids
        self._bg_task = asyncio.create_task(self._background_processor())
        
        self._initialized = True
        logger.info("Usage statistics tracker initialized")
    
    def _get_stats_file_path(self) -> str:
        """Get the path to the stats JSON file"""
        if not config.loras_roots or len(config.loras_roots) == 0:
            # Fallback to temporary directory if no lora roots
            return os.path.join(config.temp_directory, self.STATS_FILENAME)
        
        # Use the first lora root
        return os.path.join(config.loras_roots[0], self.STATS_FILENAME)
    
    def _backup_old_stats(self):
        """Backup the old stats file before conversion"""
        if os.path.exists(self._stats_file_path):
            backup_path = f"{self._stats_file_path}{self.BACKUP_SUFFIX}"
            try:
                shutil.copy2(self._stats_file_path, backup_path)
                logger.info(f"Backed up old stats file to {backup_path}")
                return True
            except Exception as e:
                logger.error(f"Failed to backup stats file: {e}")
        return False

    def _convert_old_format(self, old_stats):
        """Convert old stats format to new format with history"""
        new_stats = {
            "checkpoints": {},
            "loras": {},
            "total_executions": old_stats.get("total_executions", 0),
            "last_save_time": old_stats.get("last_save_time", time.time())
        }
        
        # Get today's date in YYYY-MM-DD format
        today = datetime.datetime.now().strftime("%Y-%m-%d")
        
        # Convert checkpoint stats
        if "checkpoints" in old_stats and isinstance(old_stats["checkpoints"], dict):
            for hash_id, count in old_stats["checkpoints"].items():
                new_stats["checkpoints"][hash_id] = {
                    "total": count,
                    "history": {
                        today: count
                    }
                }
        
        # Convert lora stats
        if "loras" in old_stats and isinstance(old_stats["loras"], dict):
            for hash_id, count in old_stats["loras"].items():
                new_stats["loras"][hash_id] = {
                    "total": count,
                    "history": {
                        today: count
                    }
                }
        
        logger.info("Successfully converted stats from old format to new format with history")
        return new_stats
    
    def _is_old_format(self, stats):
        """Check if the stats are in the old format (direct count values)"""
        # Check if any lora or checkpoint entry is a direct number instead of an object
        if "loras" in stats and isinstance(stats["loras"], dict):
            for hash_id, data in stats["loras"].items():
                if isinstance(data, (int, float)):
                    return True
                
        if "checkpoints" in stats and isinstance(stats["checkpoints"], dict):
            for hash_id, data in stats["checkpoints"].items():
                if isinstance(data, (int, float)):
                    return True
        
        return False
    
    def _load_stats(self):
        """Load existing statistics from file"""
        try:
            if os.path.exists(self._stats_file_path):
                with open(self._stats_file_path, 'r', encoding='utf-8') as f:
                    loaded_stats = json.load(f)
                
                # Check if old format and needs conversion
                if self._is_old_format(loaded_stats):
                    logger.info("Detected old stats format, performing conversion")
                    self._backup_old_stats()
                    self.stats = self._convert_old_format(loaded_stats)
                else:
                    # Update our stats with loaded data (already in new format)
                    if isinstance(loaded_stats, dict):
                        # Update individual sections to maintain structure
                        if "checkpoints" in loaded_stats and isinstance(loaded_stats["checkpoints"], dict):
                            self.stats["checkpoints"] = loaded_stats["checkpoints"]
                        
                        if "loras" in loaded_stats and isinstance(loaded_stats["loras"], dict):
                            self.stats["loras"] = loaded_stats["loras"]
                        
                        if "total_executions" in loaded_stats:
                            self.stats["total_executions"] = loaded_stats["total_executions"]
                        
                        if "last_save_time" in loaded_stats:
                            self.stats["last_save_time"] = loaded_stats["last_save_time"]
                
                logger.info(f"Loaded usage statistics from {self._stats_file_path}")
        except Exception as e:
            logger.error(f"Error loading usage statistics: {e}")
    
    async def save_stats(self, force=False):
        """Save statistics to file"""
        try:
            # Only save if it's been at least save_interval since last save or force is True
            current_time = time.time()
            if not force and (current_time - self.stats.get("last_save_time", 0)) < self.save_interval:
                return False
                
            # Use a lock to prevent concurrent writes
            async with self._lock:
                # Update last save time
                self.stats["last_save_time"] = current_time
                
                # Create directory if it doesn't exist
                os.makedirs(os.path.dirname(self._stats_file_path), exist_ok=True)
                
                # Write to a temporary file first, then move it to avoid corruption
                temp_path = f"{self._stats_file_path}.tmp"
                with open(temp_path, 'w', encoding='utf-8') as f:
                    json.dump(self.stats, f, indent=2, ensure_ascii=False)
                
                # Replace the old file with the new one
                os.replace(temp_path, self._stats_file_path)
                
                logger.debug(f"Saved usage statistics to {self._stats_file_path}")
                return True
        except Exception as e:
            logger.error(f"Error saving usage statistics: {e}", exc_info=True)
            return False
    
    def register_execution(self, prompt_id):
        """Register a completed execution by prompt_id for later processing"""
        if prompt_id:
            self.pending_prompt_ids.add(prompt_id)
    
    async def _background_processor(self):
        """Background task to process queued prompt_ids"""
        try:
            while True:
                # Wait a short interval before checking for new prompt_ids
                await asyncio.sleep(5)  # Check every 5 seconds
                
                # Process any pending prompt_ids
                if self.pending_prompt_ids:
                    async with self._lock:
                        # Get a copy of the set and clear original
                        prompt_ids = self.pending_prompt_ids.copy()
                        self.pending_prompt_ids.clear()
                    
                    # Process each prompt_id
                    registry = MetadataRegistry()
                    for prompt_id in prompt_ids:
                        try:
                            metadata = registry.get_metadata(prompt_id)
                            await self._process_metadata(metadata)
                        except Exception as e:
                            logger.error(f"Error processing prompt_id {prompt_id}: {e}")
                
                # Periodically save stats
                await self.save_stats()
        except asyncio.CancelledError:
            # Task was cancelled, clean up
            await self.save_stats(force=True)
        except Exception as e:
            logger.error(f"Error in background processing task: {e}", exc_info=True)
            # Restart the task after a delay if it fails
            asyncio.create_task(self._restart_background_task())
    
    async def _restart_background_task(self):
        """Restart the background task after a delay"""
        await asyncio.sleep(30)  # Wait 30 seconds before restarting
        self._bg_task = asyncio.create_task(self._background_processor())
    
    async def _process_metadata(self, metadata):
        """Process metadata from an execution"""
        if not metadata or not isinstance(metadata, dict):
            return
            
        # Increment total executions count
        self.stats["total_executions"] += 1
        
        # Get today's date in YYYY-MM-DD format
        today = datetime.datetime.now().strftime("%Y-%m-%d")
        
        # Process checkpoints
        if MODELS in metadata and isinstance(metadata[MODELS], dict):
            await self._process_checkpoints(metadata[MODELS], today)
        
        # Process loras
        if LORAS in metadata and isinstance(metadata[LORAS], dict):
            await self._process_loras(metadata[LORAS], today)
    
    async def _process_checkpoints(self, models_data, today_date):
        """Process checkpoint models from metadata"""
        try:
            # Get checkpoint scanner service
            checkpoint_scanner = await ServiceRegistry.get_checkpoint_scanner()
            if not checkpoint_scanner:
                logger.warning("Checkpoint scanner not available for usage tracking")
                return
            
            for node_id, model_info in models_data.items():
                if not isinstance(model_info, dict):
                    continue
                    
                # Check if this is a checkpoint model
                model_type = model_info.get("type")
                if model_type == "checkpoint":
                    model_name = model_info.get("name")
                    if not model_name:
                        continue
                    
                    # Clean up filename (remove extension if present)
                    model_filename = os.path.splitext(os.path.basename(model_name))[0]
                    
                    # Get hash for this checkpoint
                    model_hash = checkpoint_scanner.get_hash_by_filename(model_filename)
                    if model_hash:
                        # Update stats for this checkpoint with date tracking
                        if model_hash not in self.stats["checkpoints"]:
                            self.stats["checkpoints"][model_hash] = {
                                "total": 0,
                                "history": {}
                            }
                        
                        # Increment total count
                        self.stats["checkpoints"][model_hash]["total"] += 1
                        
                        # Increment today's count
                        if today_date not in self.stats["checkpoints"][model_hash]["history"]:
                            self.stats["checkpoints"][model_hash]["history"][today_date] = 0
                        self.stats["checkpoints"][model_hash]["history"][today_date] += 1
        except Exception as e:
            logger.error(f"Error processing checkpoint usage: {e}", exc_info=True)
    
    async def _process_loras(self, loras_data, today_date):
        """Process LoRA models from metadata"""
        try:
            # Get LoRA scanner service
            lora_scanner = await ServiceRegistry.get_lora_scanner()
            if not lora_scanner:
                logger.warning("LoRA scanner not available for usage tracking")
                return
            
            for node_id, lora_info in loras_data.items():
                if not isinstance(lora_info, dict):
                    continue
                
                # Get the list of LoRAs from standardized format
                lora_list = lora_info.get("lora_list", [])
                for lora in lora_list:
                    if not isinstance(lora, dict):
                        continue
                        
                    lora_name = lora.get("name")
                    if not lora_name:
                        continue
                    
                    # Get hash for this LoRA
                    lora_hash = lora_scanner.get_hash_by_filename(lora_name)
                    if lora_hash:
                        # Update stats for this LoRA with date tracking
                        if lora_hash not in self.stats["loras"]:
                            self.stats["loras"][lora_hash] = {
                                "total": 0,
                                "history": {}
                            }
                        
                        # Increment total count
                        self.stats["loras"][lora_hash]["total"] += 1
                        
                        # Increment today's count
                        if today_date not in self.stats["loras"][lora_hash]["history"]:
                            self.stats["loras"][lora_hash]["history"][today_date] = 0
                        self.stats["loras"][lora_hash]["history"][today_date] += 1
        except Exception as e:
            logger.error(f"Error processing LoRA usage: {e}", exc_info=True)
    
    async def get_stats(self):
        """Get current usage statistics"""
        return self.stats
    
    async def get_model_usage_count(self, model_type, sha256):
        """Get usage count for a specific model by hash"""
        if model_type == "checkpoint":
            if sha256 in self.stats["checkpoints"]:
                return self.stats["checkpoints"][sha256]["total"]
        elif model_type == "lora":
            if sha256 in self.stats["loras"]:
                return self.stats["loras"][sha256]["total"]
        return 0
    
    async def process_execution(self, prompt_id):
        """Process a prompt execution immediately (synchronous approach)"""
        if not prompt_id:
            return
            
        try:
            # Process metadata for this prompt_id
            registry = MetadataRegistry()
            metadata = registry.get_metadata(prompt_id)
            if metadata:
                await self._process_metadata(metadata)
                # Save stats if needed
                await self.save_stats()
        except Exception as e:
            logger.error(f"Error processing prompt_id {prompt_id}: {e}", exc_info=True)
