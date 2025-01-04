from __future__ import annotations

import os
import folder_paths
import glob
from aiohttp import web

class CustomNodeManager:
    """
    Placeholder to refactor the custom node management features from ComfyUI-Manager.
    Currently it only contains the custom workflow templates feature.
    """
    def add_routes(self, routes, webapp, loadedModules):

        @routes.get("/workflow_templates")
        async def get_workflow_templates(request):
            """Returns a web response that contains the map of custom_nodes names and their associated workflow templates. The ones without templates are omitted."""
            files = [
                file
                for folder in folder_paths.get_folder_paths("custom_nodes")
                for file in glob.glob(os.path.join(folder, '*/example_workflows/*.json'))
            ]
            workflow_templates_dict = {} # custom_nodes folder name -> example workflow names
            for file in files:
                custom_nodes_name = os.path.basename(os.path.dirname(os.path.dirname(file)))
                workflow_name = os.path.splitext(os.path.basename(file))[0]
                workflow_templates_dict.setdefault(custom_nodes_name, []).append(workflow_name)
            return web.json_response(workflow_templates_dict)

        # Serve workflow templates from custom nodes.
        for module_name, module_dir in loadedModules:
            workflows_dir = os.path.join(module_dir, 'example_workflows')
            if os.path.exists(workflows_dir):
                webapp.add_routes([web.static('/api/workflow_templates/' + module_name, workflows_dir)])
