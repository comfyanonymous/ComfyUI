import pytest
from aiohttp import web
from aiohttp.test_utils import AioHTTPTestCase, unittest_run_loop
import json
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from api_server.routes.internal.internal_routes import InternalRoutes


class TestInternalRoutesAuthRequired(AioHTTPTestCase):
    """Invariant: Protected internal endpoints must reject unauthenticated requests"""

    async def get_application(self):
        app = web.Application()
        internal_routes = InternalRoutes()
        internal_routes.setup_routes(app)
        return app

    @unittest_run_loop
    async def test_unauthenticated_requests_rejected(self):
        """Internal endpoints should reject requests without valid authentication"""
        test_cases = [
            # No auth header at all
            {"headers": {}, "desc": "missing_auth"},
            # Malformed token
            {"headers": {"Authorization": "Bearer invalid.malformed.token"}, "desc": "malformed_token"},
            # Empty bearer token
            {"headers": {"Authorization": "Bearer "}, "desc": "empty_token"},
        ]
        
        endpoints = [
            ("/internal/status", "GET"),
            ("/internal/config", "POST"),
        ]
        
        for endpoint, method in endpoints:
            for case in test_cases:
                if method == "GET":
                    resp = await self.client.get(endpoint, headers=case["headers"])
                else:
                    resp = await self.client.post(
                        endpoint, 
                        headers=case["headers"],
                        json={"test": "data"}
                    )
                
                assert resp.status in (401, 403), (
                    f"Endpoint {endpoint} accepted unauthenticated request "
                    f"({case['desc']}): got {resp.status}, expected 401/403"
                )


@pytest.mark.asyncio
async def test_internal_routes_require_auth():
    """Standalone test: internal routes must require authentication"""
    from aiohttp.test_utils import TestClient, TestServer
    
    app = web.Application()
    internal_routes = InternalRoutes()
    internal_routes.setup_routes(app)
    
    async with TestClient(TestServer(app)) as client:
        # Test without any authentication
        resp = await client.post("/internal/config", json={"action": "get_config"})
        assert resp.status in (401, 403), f"Expected 401/403, got {resp.status}"