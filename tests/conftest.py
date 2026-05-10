import os
import pytest

# Command line arguments for pytest
def pytest_addoption(parser):
    parser.addoption('--output_dir', action="store", default='tests/inference/samples', help='Output directory for generated images')
    parser.addoption("--listen", type=str, default="127.0.0.1", metavar="IP", nargs="?", const="0.0.0.0", help="Specify the IP address to listen on (default: 127.0.0.1). If --listen is provided without an argument, it defaults to 0.0.0.0. (listens on all)")
    parser.addoption("--port", type=int, default=8188, help="Set the listen port.")
    parser.addoption("--skip-timing-checks", action="store_true", default=False, help="Skip timing-related assertions in tests (useful for CI environments with variable performance)")

# This initializes args at the beginning of the test session
@pytest.fixture(scope="session", autouse=True)
def args_pytest(pytestconfig):
    args = {}
    args['output_dir'] = pytestconfig.getoption('output_dir')
    args['listen'] = pytestconfig.getoption('listen')
    args['port'] = pytestconfig.getoption('port')

    os.makedirs(args['output_dir'], exist_ok=True)

    return args

@pytest.fixture(scope="session")
def skip_timing_checks(pytestconfig):
    """Fixture that returns whether timing checks should be skipped."""
    return pytestconfig.getoption("--skip-timing-checks")

def pytest_collection_modifyitems(items):
    # Modifies items so tests run in the correct order

    LAST_TESTS = ['test_quality']

    # Move the last items to the end
    last_items = []
    for test_name in LAST_TESTS:
        for item in items.copy():
            print(item.module.__name__, item)  # noqa: T201
            if item.module.__name__  == test_name:
                last_items.append(item)
                items.remove(item)

    items.extend(last_items)
