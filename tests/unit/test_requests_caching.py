import pytest
import time
import requests
import logging

from comfy.node_requests_caching import use_requests_caching

logger = logging.getLogger(__name__)

MIN_NETWORK_TIME_SEC = 0.1


@pytest.mark.parametrize("test_url", [
    "https://fonts.gstatic.com/s/lato/v23/S6uyw4BMUTPHjxAwXiWtFCfQ7A.woff2"
])
def test_caching_context_manager_works(test_url, tmp_path):
    logger.info(f"\n[Test] Call 1 (Inside Context): Fetching... {test_url}")
    start_time_1 = time.time()
    with use_requests_caching(cache_name=tmp_path):
        r1 = requests.get(test_url, timeout=10)
    duration_1 = time.time() - start_time_1

    logger.info(f"Call 1 took: {duration_1:.3f}s")
    assert r1.status_code == 200
    assert r1.from_cache is False
    assert "Cache-Control" in r1.headers, "Response must have 'Cache-Control' header for this test to be valid"

    logger.info(f"[Test] Call 2 (Inside Context): From cache... {test_url}")
    start_time_2 = time.time()
    with use_requests_caching(cache_name=tmp_path):
        r2 = requests.get(test_url, timeout=10)
    duration_2 = time.time() - start_time_2

    logger.info(f"Call 2 took: {duration_2:.3f}s")
    assert r2.status_code == 200
    assert r2.from_cache is True

    logger.info(f"[Test] Call 3 (Outside Context): Fetching again... {test_url}")
    start_time_3 = time.time()
    r3 = requests.get(test_url, timeout=10)
    duration_3 = time.time() - start_time_3

    logger.info(f"Call 3 took: {duration_3:.3f}s")
    assert r3.status_code == 200
    # A standard response object has no 'from_cache' attribute
    assert getattr(r3, 'from_cache', None) is None
