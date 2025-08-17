import os
import json
import jinja2
from aiohttp import web
import logging
from datetime import datetime, timedelta
from collections import defaultdict, Counter
from typing import Dict, List, Any

from ..config import config
from ..services.settings_manager import settings
from ..services.service_registry import ServiceRegistry
from ..utils.usage_stats import UsageStats

logger = logging.getLogger(__name__)

class StatsRoutes:
    """Route handlers for Statistics page and API endpoints"""
    
    def __init__(self):
        self.lora_scanner = None
        self.checkpoint_scanner = None
        self.embedding_scanner = None
        self.usage_stats = None
        self.template_env = jinja2.Environment(
            loader=jinja2.FileSystemLoader(config.templates_path),
            autoescape=True
        )

    async def init_services(self):
        """Initialize services from ServiceRegistry"""
        self.lora_scanner = await ServiceRegistry.get_lora_scanner()
        self.checkpoint_scanner = await ServiceRegistry.get_checkpoint_scanner()
        self.embedding_scanner = await ServiceRegistry.get_embedding_scanner()
        self.usage_stats = UsageStats()

    async def handle_stats_page(self, request: web.Request) -> web.Response:
        """Handle GET /statistics request"""
        try:
            # Ensure services are initialized
            await self.init_services()
            
            # Check if scanners are initializing
            lora_initializing = (
                self.lora_scanner._cache is None or 
                (hasattr(self.lora_scanner, 'is_initializing') and self.lora_scanner.is_initializing())
            )
            
            checkpoint_initializing = (
                self.checkpoint_scanner._cache is None or
                (hasattr(self.checkpoint_scanner, '_is_initializing') and self.checkpoint_scanner._is_initializing)
            )
            
            embedding_initializing = (
                self.embedding_scanner._cache is None or
                (hasattr(self.embedding_scanner, 'is_initializing') and self.embedding_scanner.is_initializing())
            )
            
            is_initializing = lora_initializing or checkpoint_initializing or embedding_initializing

            template = self.template_env.get_template('statistics.html')
            rendered = template.render(
                is_initializing=is_initializing,
                settings=settings,
                request=request
            )
            
            return web.Response(
                text=rendered,
                content_type='text/html'
            )
            
        except Exception as e:
            logger.error(f"Error handling statistics request: {e}", exc_info=True)
            return web.Response(
                text="Error loading statistics page",
                status=500
            )

    async def get_collection_overview(self, request: web.Request) -> web.Response:
        """Get collection overview statistics"""
        try:
            await self.init_services()
            
            # Get LoRA statistics
            lora_cache = await self.lora_scanner.get_cached_data()
            lora_count = len(lora_cache.raw_data)
            lora_size = sum(lora.get('size', 0) for lora in lora_cache.raw_data)
            
            # Get Checkpoint statistics
            checkpoint_cache = await self.checkpoint_scanner.get_cached_data()
            checkpoint_count = len(checkpoint_cache.raw_data)
            checkpoint_size = sum(cp.get('size', 0) for cp in checkpoint_cache.raw_data)
            
            # Get Embedding statistics
            embedding_cache = await self.embedding_scanner.get_cached_data()
            embedding_count = len(embedding_cache.raw_data)
            embedding_size = sum(emb.get('size', 0) for emb in embedding_cache.raw_data)
            
            # Get usage statistics
            usage_data = await self.usage_stats.get_stats()
            
            return web.json_response({
                'success': True,
                'data': {
                    'total_models': lora_count + checkpoint_count + embedding_count,
                    'lora_count': lora_count,
                    'checkpoint_count': checkpoint_count,
                    'embedding_count': embedding_count,
                    'total_size': lora_size + checkpoint_size + embedding_size,
                    'lora_size': lora_size,
                    'checkpoint_size': checkpoint_size,
                    'embedding_size': embedding_size,
                    'total_generations': usage_data.get('total_executions', 0),
                    'unused_loras': self._count_unused_models(lora_cache.raw_data, usage_data.get('loras', {})),
                    'unused_checkpoints': self._count_unused_models(checkpoint_cache.raw_data, usage_data.get('checkpoints', {})),
                    'unused_embeddings': self._count_unused_models(embedding_cache.raw_data, usage_data.get('embeddings', {}))
                }
            })
            
        except Exception as e:
            logger.error(f"Error getting collection overview: {e}", exc_info=True)
            return web.json_response({
                'success': False,
                'error': str(e)
            }, status=500)

    async def get_usage_analytics(self, request: web.Request) -> web.Response:
        """Get usage analytics data"""
        try:
            await self.init_services()
            
            # Get usage statistics
            usage_data = await self.usage_stats.get_stats()
            
            # Get model data for enrichment
            lora_cache = await self.lora_scanner.get_cached_data()
            checkpoint_cache = await self.checkpoint_scanner.get_cached_data()
            embedding_cache = await self.embedding_scanner.get_cached_data()
            
            # Create hash to model mapping
            lora_map = {lora['sha256']: lora for lora in lora_cache.raw_data}
            checkpoint_map = {cp['sha256']: cp for cp in checkpoint_cache.raw_data}
            embedding_map = {emb['sha256']: emb for emb in embedding_cache.raw_data}
            
            # Prepare top used models
            top_loras = self._get_top_used_models(usage_data.get('loras', {}), lora_map, 10)
            top_checkpoints = self._get_top_used_models(usage_data.get('checkpoints', {}), checkpoint_map, 10)
            top_embeddings = self._get_top_used_models(usage_data.get('embeddings', {}), embedding_map, 10)
            
            # Prepare usage timeline (last 30 days)
            timeline = self._get_usage_timeline(usage_data, 30)
            
            return web.json_response({
                'success': True,
                'data': {
                    'top_loras': top_loras,
                    'top_checkpoints': top_checkpoints,
                    'top_embeddings': top_embeddings,
                    'usage_timeline': timeline,
                    'total_executions': usage_data.get('total_executions', 0)
                }
            })
            
        except Exception as e:
            logger.error(f"Error getting usage analytics: {e}", exc_info=True)
            return web.json_response({
                'success': False,
                'error': str(e)
            }, status=500)

    async def get_base_model_distribution(self, request: web.Request) -> web.Response:
        """Get base model distribution statistics"""
        try:
            await self.init_services()
            
            # Get model data
            lora_cache = await self.lora_scanner.get_cached_data()
            checkpoint_cache = await self.checkpoint_scanner.get_cached_data()
            embedding_cache = await self.embedding_scanner.get_cached_data()
            
            # Count by base model
            lora_base_models = Counter(lora.get('base_model', 'Unknown') for lora in lora_cache.raw_data)
            checkpoint_base_models = Counter(cp.get('base_model', 'Unknown') for cp in checkpoint_cache.raw_data)
            embedding_base_models = Counter(emb.get('base_model', 'Unknown') for emb in embedding_cache.raw_data)
            
            return web.json_response({
                'success': True,
                'data': {
                    'loras': dict(lora_base_models),
                    'checkpoints': dict(checkpoint_base_models),
                    'embeddings': dict(embedding_base_models)
                }
            })
            
        except Exception as e:
            logger.error(f"Error getting base model distribution: {e}", exc_info=True)
            return web.json_response({
                'success': False,
                'error': str(e)
            }, status=500)

    async def get_tag_analytics(self, request: web.Request) -> web.Response:
        """Get tag usage analytics"""
        try:
            await self.init_services()
            
            # Get model data
            lora_cache = await self.lora_scanner.get_cached_data()
            checkpoint_cache = await self.checkpoint_scanner.get_cached_data()
            embedding_cache = await self.embedding_scanner.get_cached_data()
            
            # Count tag frequencies
            all_tags = []
            for lora in lora_cache.raw_data:
                all_tags.extend(lora.get('tags', []))
            for cp in checkpoint_cache.raw_data:
                all_tags.extend(cp.get('tags', []))
            for emb in embedding_cache.raw_data:
                all_tags.extend(emb.get('tags', []))
            
            tag_counts = Counter(all_tags)
            
            # Get top 50 tags
            top_tags = [{'tag': tag, 'count': count} for tag, count in tag_counts.most_common(50)]
            
            return web.json_response({
                'success': True,
                'data': {
                    'top_tags': top_tags,
                    'total_unique_tags': len(tag_counts)
                }
            })
            
        except Exception as e:
            logger.error(f"Error getting tag analytics: {e}", exc_info=True)
            return web.json_response({
                'success': False,
                'error': str(e)
            }, status=500)

    async def get_storage_analytics(self, request: web.Request) -> web.Response:
        """Get storage usage analytics"""
        try:
            await self.init_services()
            
            # Get usage statistics
            usage_data = await self.usage_stats.get_stats()
            
            # Get model data
            lora_cache = await self.lora_scanner.get_cached_data()
            checkpoint_cache = await self.checkpoint_scanner.get_cached_data()
            embedding_cache = await self.embedding_scanner.get_cached_data()
            
            # Create models with usage data
            lora_storage = []
            for lora in lora_cache.raw_data:
                usage_count = 0
                if lora['sha256'] in usage_data.get('loras', {}):
                    usage_count = usage_data['loras'][lora['sha256']].get('total', 0)
                
                lora_storage.append({
                    'name': lora['model_name'],
                    'size': lora.get('size', 0),
                    'usage_count': usage_count,
                    'folder': lora.get('folder', ''),
                    'base_model': lora.get('base_model', 'Unknown')
                })
            
            checkpoint_storage = []
            for cp in checkpoint_cache.raw_data:
                usage_count = 0
                if cp['sha256'] in usage_data.get('checkpoints', {}):
                    usage_count = usage_data['checkpoints'][cp['sha256']].get('total', 0)
                
                checkpoint_storage.append({
                    'name': cp['model_name'],
                    'size': cp.get('size', 0),
                    'usage_count': usage_count,
                    'folder': cp.get('folder', ''),
                    'base_model': cp.get('base_model', 'Unknown')
                })
            
            embedding_storage = []
            for emb in embedding_cache.raw_data:
                usage_count = 0
                if emb['sha256'] in usage_data.get('embeddings', {}):
                    usage_count = usage_data['embeddings'][emb['sha256']].get('total', 0)
                
                embedding_storage.append({
                    'name': emb['model_name'],
                    'size': emb.get('size', 0),
                    'usage_count': usage_count,
                    'folder': emb.get('folder', ''),
                    'base_model': emb.get('base_model', 'Unknown')
                })
            
            # Sort by size
            lora_storage.sort(key=lambda x: x['size'], reverse=True)
            checkpoint_storage.sort(key=lambda x: x['size'], reverse=True)
            embedding_storage.sort(key=lambda x: x['size'], reverse=True)
            
            return web.json_response({
                'success': True,
                'data': {
                    'loras': lora_storage[:20],  # Top 20 by size
                    'checkpoints': checkpoint_storage[:20],
                    'embeddings': embedding_storage[:20]
                }
            })
            
        except Exception as e:
            logger.error(f"Error getting storage analytics: {e}", exc_info=True)
            return web.json_response({
                'success': False,
                'error': str(e)
            }, status=500)

    async def get_insights(self, request: web.Request) -> web.Response:
        """Get smart insights about the collection"""
        try:
            await self.init_services()
            
            # Get usage statistics
            usage_data = await self.usage_stats.get_stats()
            
            # Get model data
            lora_cache = await self.lora_scanner.get_cached_data()
            checkpoint_cache = await self.checkpoint_scanner.get_cached_data()
            embedding_cache = await self.embedding_scanner.get_cached_data()
            
            insights = []
            
            # Calculate unused models
            unused_loras = self._count_unused_models(lora_cache.raw_data, usage_data.get('loras', {}))
            unused_checkpoints = self._count_unused_models(checkpoint_cache.raw_data, usage_data.get('checkpoints', {}))
            unused_embeddings = self._count_unused_models(embedding_cache.raw_data, usage_data.get('embeddings', {}))
            
            total_loras = len(lora_cache.raw_data)
            total_checkpoints = len(checkpoint_cache.raw_data)
            total_embeddings = len(embedding_cache.raw_data)
            
            if total_loras > 0:
                unused_lora_percent = (unused_loras / total_loras) * 100
                if unused_lora_percent > 50:
                    insights.append({
                        'type': 'warning',
                        'title': 'High Number of Unused LoRAs',
                        'description': f'{unused_lora_percent:.1f}% of your LoRAs ({unused_loras}/{total_loras}) have never been used.',
                        'suggestion': 'Consider organizing or archiving unused models to free up storage space.'
                    })
            
            if total_checkpoints > 0:
                unused_checkpoint_percent = (unused_checkpoints / total_checkpoints) * 100
                if unused_checkpoint_percent > 30:
                    insights.append({
                        'type': 'warning',
                        'title': 'Unused Checkpoints Detected',
                        'description': f'{unused_checkpoint_percent:.1f}% of your checkpoints ({unused_checkpoints}/{total_checkpoints}) have never been used.',
                        'suggestion': 'Review and consider removing checkpoints you no longer need.'
                    })
            
            if total_embeddings > 0:
                unused_embedding_percent = (unused_embeddings / total_embeddings) * 100
                if unused_embedding_percent > 50:
                    insights.append({
                        'type': 'warning',
                        'title': 'High Number of Unused Embeddings',
                        'description': f'{unused_embedding_percent:.1f}% of your embeddings ({unused_embeddings}/{total_embeddings}) have never been used.',
                        'suggestion': 'Consider organizing or archiving unused embeddings to optimize your collection.'
                    })
            
            # Storage insights
            total_size = sum(lora.get('size', 0) for lora in lora_cache.raw_data) + \
                        sum(cp.get('size', 0) for cp in checkpoint_cache.raw_data) + \
                        sum(emb.get('size', 0) for emb in embedding_cache.raw_data)
            
            if total_size > 100 * 1024 * 1024 * 1024:  # 100GB
                insights.append({
                    'type': 'info',
                    'title': 'Large Collection Detected',
                    'description': f'Your model collection is using {self._format_size(total_size)} of storage.',
                    'suggestion': 'Consider using external storage or cloud solutions for better organization.'
                })
            
            # Recent activity insight
            if usage_data.get('total_executions', 0) > 100:
                insights.append({
                    'type': 'success',
                    'title': 'Active User',
                    'description': f'You\'ve completed {usage_data["total_executions"]} generations so far!',
                    'suggestion': 'Keep exploring and creating amazing content with your models.'
                })
            
            return web.json_response({
                'success': True,
                'data': {
                    'insights': insights
                }
            })
            
        except Exception as e:
            logger.error(f"Error getting insights: {e}", exc_info=True)
            return web.json_response({
                'success': False,
                'error': str(e)
            }, status=500)

    def _count_unused_models(self, models: List[Dict], usage_data: Dict) -> int:
        """Count models that have never been used"""
        used_hashes = set(usage_data.keys())
        unused_count = 0
        
        for model in models:
            if model.get('sha256') not in used_hashes:
                unused_count += 1
                
        return unused_count

    def _get_top_used_models(self, usage_data: Dict, model_map: Dict, limit: int) -> List[Dict]:
        """Get top used models with their metadata"""
        sorted_usage = sorted(usage_data.items(), key=lambda x: x[1].get('total', 0), reverse=True)
        
        top_models = []
        for sha256, usage_info in sorted_usage[:limit]:
            if sha256 in model_map:
                model = model_map[sha256]
                top_models.append({
                    'name': model['model_name'],
                    'usage_count': usage_info.get('total', 0),
                    'base_model': model.get('base_model', 'Unknown'),
                    'preview_url': config.get_preview_static_url(model.get('preview_url', '')),
                    'folder': model.get('folder', '')
                })
        
        return top_models

    def _get_usage_timeline(self, usage_data: Dict, days: int) -> List[Dict]:
        """Get usage timeline for the past N days"""
        timeline = []
        today = datetime.now()
        
        for i in range(days):
            date = today - timedelta(days=i)
            date_str = date.strftime('%Y-%m-%d')
            
            lora_usage = 0
            checkpoint_usage = 0
            embedding_usage = 0
            
            # Count usage for this date
            for model_usage in usage_data.get('loras', {}).values():
                if isinstance(model_usage, dict) and 'history' in model_usage:
                    lora_usage += model_usage['history'].get(date_str, 0)
            
            for model_usage in usage_data.get('checkpoints', {}).values():
                if isinstance(model_usage, dict) and 'history' in model_usage:
                    checkpoint_usage += model_usage['history'].get(date_str, 0)
            
            for model_usage in usage_data.get('embeddings', {}).values():
                if isinstance(model_usage, dict) and 'history' in model_usage:
                    embedding_usage += model_usage['history'].get(date_str, 0)
            
            timeline.append({
                'date': date_str,
                'lora_usage': lora_usage,
                'checkpoint_usage': checkpoint_usage,
                'embedding_usage': embedding_usage,
                'total_usage': lora_usage + checkpoint_usage + embedding_usage
            })
        
        return list(reversed(timeline))  # Oldest to newest

    def _format_size(self, size_bytes: int) -> str:
        """Format file size in human readable format"""
        for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
            if size_bytes < 1024.0:
                return f"{size_bytes:.1f} {unit}"
            size_bytes /= 1024.0
        return f"{size_bytes:.1f} PB"

    def setup_routes(self, app: web.Application):
        """Register routes with the application"""
        # Add an app startup handler to initialize services
        app.on_startup.append(self._on_startup)
        
        # Register page route
        app.router.add_get('/statistics', self.handle_stats_page)
        
        # Register API routes
        app.router.add_get('/api/stats/collection-overview', self.get_collection_overview)
        app.router.add_get('/api/stats/usage-analytics', self.get_usage_analytics)
        app.router.add_get('/api/stats/base-model-distribution', self.get_base_model_distribution)
        app.router.add_get('/api/stats/tag-analytics', self.get_tag_analytics)
        app.router.add_get('/api/stats/storage-analytics', self.get_storage_analytics)
        app.router.add_get('/api/stats/insights', self.get_insights)
        
    async def _on_startup(self, app):
        """Initialize services when the app starts"""
        await self.init_services()