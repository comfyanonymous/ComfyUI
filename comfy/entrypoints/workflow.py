from ..cmd.main_pre import args

import asyncio
import json
import logging
from typing import Optional, Literal

import typer
from ..cli_args_types import Configuration
from ..component_model.asyncio_files import stream_json_objects
from ..client.embedded_comfy_client import Comfy
from ..component_model.entrypoints_common import configure_application_paths, executor_from_args

logger = logging.getLogger(__name__)


async def main():
    workflows = args.workflows
    assert len(workflows) > 0, "specify at least one path to a workflow, a literal workflow json starting with `{` or `-` (for standard in) using --workflows cli arg"
    configure_application_paths(args)
    executor = await executor_from_args(args)

    await run_workflows(executor, workflows)


async def run_workflows(executor, workflows: list[str | Literal["-"]], configuration: Optional[Configuration] = None):
    if configuration is None:
        configuration = args
    async with Comfy(executor=executor, configuration=configuration) as comfy:
        for workflow in workflows:
            obj: dict
            async for obj in stream_json_objects(workflow):
                try:
                    res = await comfy.queue_prompt_api(obj)
                    typer.echo(json.dumps(res.outputs))
                except asyncio.CancelledError:
                    logger.info("Exiting gracefully.")
                    break


def entrypoint():
    asyncio.run(main())


if __name__ == "__main__":
    entrypoint()
