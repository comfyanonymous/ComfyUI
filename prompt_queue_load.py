import execution
import folder_paths
import os
import time
import sys
import logging
import traceback
import importlib

from prompt_queue_interface import PromptQueueInterface

QUEUE_CLASS_MAPPINGS = {
}


def check_queue_class(cls) -> bool:
    if cls is None or not isinstance(cls, type) or not issubclass(cls, PromptQueueInterface):
        return False
    return True


def create_prompt_queue(server, queuq_name: str) -> PromptQueueInterface:
    PromptQueueClass = QUEUE_CLASS_MAPPINGS.get(queuq_name, execution.PromptQueue)
    return PromptQueueClass(server)


def load_queues():
    base_names = set(QUEUE_CLASS_MAPPINGS.keys())
    queue_import_times = []

    paths = folder_paths.get_folder_paths("queues")
    for queue_path in paths:
        possible_modules = os.listdir(os.path.realpath(queue_path))

        if "__pycache__" in possible_modules:
            possible_modules.remove("__pycache__")

        for possible_module in possible_modules:
            module_path = os.path.join(queue_path, possible_module)

            if os.path.isfile(module_path) and os.path.splitext(module_path)[1] != ".py":
                continue

            if module_path.endswith(".disabled"):
                continue

            time_before = time.perf_counter()
            success = load_queue(module_path, base_names)
            queue_import_times.append((time.perf_counter() - time_before, module_path, success))

    if len(queue_import_times) > 0:
        logging.info("\nImport times for queues:")
        for n in sorted(queue_import_times):
            if n[2]:
                queue_import_times = ""
            else:
                queue_import_times = " (IMPORT FAILED)"
            logging.info("{:6.1f} seconds{}: {}".format(n[0], queue_import_times, n[1]))
        logging.info("")


def load_queue(module_path, ignore: set):
    module_name = os.path.basename(module_path)

    if os.path.isfile(module_path):
        sp = os.path.splitext(module_path)
        module_name = sp[0]

    try:
        logging.debug("Trying to load queue {}".format(module_path))

        if os.path.isfile(module_path):
            module_spec = importlib.util.spec_from_file_location(module_name, module_path)
        else:
            module_spec = importlib.util.spec_from_file_location(module_name, os.path.join(module_path, "__init__.py"))

        module = importlib.util.module_from_spec(module_spec)
        sys.modules[module_name] = module
        module_spec.loader.exec_module(module)

        if hasattr(module, "QUEUE_CLASS_MAPPINGS") and getattr(module, "QUEUE_CLASS_MAPPINGS") is not None:
            for name in module.QUEUE_CLASS_MAPPINGS:
                if name not in ignore and check_queue_class(module.QUEUE_CLASS_MAPPINGS[name]):
                    QUEUE_CLASS_MAPPINGS[name] = module.QUEUE_CLASS_MAPPINGS[name]
            return True
        else:
            logging.warning(f"Skip {module_path} module for queue due to the lack of QUEUE_CLASS_MAPPINGS.")
            return False
    except Exception as e:
        logging.warning(traceback.format_exc())
        logging.warning(f"Cannot import {module_path} module for queue: {e}")
        return False
