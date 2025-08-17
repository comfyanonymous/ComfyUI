import logging
from aiohttp import web
from typing import Set, Dict, Optional
from uuid import uuid4
import asyncio
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class WebSocketManager:
    """Manages WebSocket connections and broadcasts"""
    
    def __init__(self):
        self._websockets: Set[web.WebSocketResponse] = set()
        self._init_websockets: Set[web.WebSocketResponse] = set()  # New set for initialization progress clients
        self._download_websockets: Dict[str, web.WebSocketResponse] = {}  # New dict for download-specific clients
        # Add progress tracking dictionary
        self._download_progress: Dict[str, Dict] = {}
        
    async def handle_connection(self, request: web.Request) -> web.WebSocketResponse:
        """Handle new WebSocket connection"""
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        self._websockets.add(ws)
        
        try:
            async for msg in ws:
                if msg.type == web.WSMsgType.ERROR:
                    logger.error(f'WebSocket error: {ws.exception()}')
        finally:
            self._websockets.discard(ws)
        return ws
    
    async def handle_init_connection(self, request: web.Request) -> web.WebSocketResponse:
        """Handle new WebSocket connection for initialization progress"""
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        self._init_websockets.add(ws)
        
        try:
            async for msg in ws:
                if msg.type == web.WSMsgType.ERROR:
                    logger.error(f'Init WebSocket error: {ws.exception()}')
        finally:
            self._init_websockets.discard(ws)
        return ws
    
    async def handle_download_connection(self, request: web.Request) -> web.WebSocketResponse:
        """Handle new WebSocket connection for download progress"""
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        
        # Get download_id from query parameters
        download_id = request.query.get('id')
        
        if not download_id:
            # Generate a new download ID if not provided
            download_id = str(uuid4())
        
        # Store the websocket with its download ID
        self._download_websockets[download_id] = ws
        
        try:
            # Send the download ID back to the client
            await ws.send_json({
                'type': 'download_id',
                'download_id': download_id
            })
            
            async for msg in ws:
                if msg.type == web.WSMsgType.ERROR:
                    logger.error(f'Download WebSocket error: {ws.exception()}')
        finally:
            if download_id in self._download_websockets:
                del self._download_websockets[download_id]
            
            # Schedule cleanup of completed downloads after WebSocket disconnection
            asyncio.create_task(self._delayed_cleanup(download_id))
        return ws
    
    async def _delayed_cleanup(self, download_id: str, delay_seconds: int = 300):
        """Clean up download progress after a delay (5 minutes by default)"""
        await asyncio.sleep(delay_seconds)
        progress_data = self._download_progress.get(download_id)
        if progress_data and progress_data.get('progress', 0) >= 100:
            self.cleanup_download_progress(download_id)
            logger.debug(f"Delayed cleanup completed for download {download_id}")
    
    async def broadcast(self, data: Dict):
        """Broadcast message to all connected clients"""
        if not self._websockets:
            return
            
        for ws in self._websockets:
            try:
                await ws.send_json(data)
            except Exception as e:
                logger.error(f"Error sending progress: {e}")
    
    async def broadcast_init_progress(self, data: Dict):
        """Broadcast initialization progress to connected clients"""
        if not self._init_websockets:
            return
            
        # Ensure data has all required fields
        if 'stage' not in data:
            data['stage'] = 'processing'
        if 'progress' not in data:
            data['progress'] = 0
        if 'details' not in data:
            data['details'] = 'Processing...'
            
        for ws in self._init_websockets:
            try:
                await ws.send_json(data)
            except Exception as e:
                logger.error(f"Error sending initialization progress: {e}")
    
    async def broadcast_download_progress(self, download_id: str, data: Dict):
        """Send progress update to specific download client"""
        # Store simplified progress data in memory (only progress percentage)
        self._download_progress[download_id] = {
            'progress': data.get('progress', 0),
            'timestamp': datetime.now()
        }
        
        if download_id not in self._download_websockets:
            logger.debug(f"No WebSocket found for download ID: {download_id}")
            return
            
        ws = self._download_websockets[download_id]
        try:
            await ws.send_json(data)
        except Exception as e:
            logger.error(f"Error sending download progress: {e}")
            
    def get_download_progress(self, download_id: str) -> Optional[Dict]:
        """Get progress information for a specific download"""
        return self._download_progress.get(download_id)
    
    def cleanup_download_progress(self, download_id: str):
        """Remove progress info for a specific download"""
        self._download_progress.pop(download_id, None)
    
    def cleanup_old_downloads(self, max_age_hours: int = 24):
        """Clean up old download progress entries"""
        cutoff_time = datetime.now() - timedelta(hours=max_age_hours)
        to_remove = []
        
        for download_id, progress_data in self._download_progress.items():
            if progress_data.get('timestamp', datetime.now()) < cutoff_time:
                to_remove.append(download_id)
        
        for download_id in to_remove:
            self._download_progress.pop(download_id, None)
            logger.debug(f"Cleaned up old download progress for {download_id}")
            
    def get_connected_clients_count(self) -> int:
        """Get number of connected clients"""
        return len(self._websockets)

    def get_init_clients_count(self) -> int:
        """Get number of initialization progress clients"""
        return len(self._init_websockets)
        
    def get_download_clients_count(self) -> int:
        """Get number of download progress clients"""
        return len(self._download_websockets)
        
    def generate_download_id(self) -> str:
        """Generate a unique download ID"""
        return str(uuid4())

# Global instance
ws_manager = WebSocketManager()