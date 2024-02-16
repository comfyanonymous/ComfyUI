import os
import time
import urllib

import pytest


# Command line arguments for pytest
def pytest_addoption(parser):
    parser.addoption('--output_dir', action="store", default='tests/inference/samples',
                     help='Output directory for generated images')
    parser.addoption("--listen", type=str, default="127.0.0.1", metavar="IP", nargs="?", const="0.0.0.0",
                     help="Specify the IP address to listen on (default: 127.0.0.1). If --listen is provided without an argument, it defaults to 0.0.0.0. (listens on all)")
    parser.addoption("--port", type=int, default=8188, help="Set the listen port.")


def run_server(args_pytest):
    from comfy.cmd.main import main
    from comfy.cli_args import args
    import asyncio
    args.output_directory = args_pytest["output_dir"]
    args.listen = args_pytest["listen"]
    args.port = args_pytest["port"]
    print("running server anyway!")
    asyncio.run(main())


# This initializes args at the beginning of the test session
@pytest.fixture(scope="session", autouse=False)
def args_pytest(pytestconfig):
    args = {}
    args['output_dir'] = pytestconfig.getoption('output_dir')
    args['listen'] = pytestconfig.getoption('listen')
    args['port'] = pytestconfig.getoption('port')

    os.makedirs(args['output_dir'], exist_ok=True)

    return args


@pytest.fixture(scope="module", autouse=False)
def comfy_background_server(args_pytest):
    import multiprocessing
    import torch
    # Start server

    pickled_args = {
        "output_dir": args_pytest["output_dir"],
        "listen": args_pytest["listen"],
        "port": args_pytest["port"]
    }
    p = multiprocessing.Process(target=run_server, args=(pickled_args,))
    p.start()
    # wait for http url to be ready
    success = False
    for i in range(60):
        try:
            with urllib.request.urlopen(f"http://localhost:{pickled_args['port']}/object_info") as response:
                success = response.status == 200
                if success:
                    break
        except:
            pass
        time.sleep(1)
    if not success:
        raise Exception("Failed to start background server")
    yield
    p.kill()
    torch.cuda.empty_cache()


def pytest_collection_modifyitems(items):
    # Modifies items so tests run in the correct order

    LAST_TESTS = ['test_quality']

    # Move the last items to the end
    last_items = []
    for test_name in LAST_TESTS:
        for item in items.copy():
            print(item.module.__name__, item)
            if item.module.__name__ == test_name:
                last_items.append(item)
                items.remove(item)

    items.extend(last_items)
