from __future__ import annotations

from typing import TypedDict
import os
import folder_paths
import glob
from aiohttp import web
import hashlib


class Source:
    custom_node = "custom_node"

class SubgraphEntry(TypedDict):
    source: str
    """
    Source of subgraph - custom_nodes vs templates.
    """
    path: str
    """
    Relative path of the subgraph file.
    For custom nodes, will be the relative directory like <custom_node_dir>/subgraphs/<name>.json
    """
    name: str
    """
    Name of subgraph file.
    """
    info: CustomNodeSubgraphEntryInfo
    """
    Additional info about subgraph; in the case of custom_nodes, will contain nodepack name
    """
    data: str

class CustomNodeSubgraphEntryInfo(TypedDict):
    node_pack: str
    """Node pack name."""

class SubgraphManager:
    def __init__(self):
        self.cached_custom_node_subgraphs: dict[SubgraphEntry] | None = None

    async def load_entry_data(self, entry: SubgraphEntry):
        with open(entry['path'], 'r') as f:
            entry['data'] = f.read()
        return entry

    async def sanitize_entry(self, entry: SubgraphEntry | None, remove_data=False) -> SubgraphEntry | None:
        if entry is None:
            return None
        entry = entry.copy()
        entry.pop('path', None)
        if remove_data:
            entry.pop('data', None)
        return entry

    async def sanitize_entries(self, entries: dict[str, SubgraphEntry], remove_data=False) -> dict[str, SubgraphEntry]:
        entries = entries.copy()
        for key in list(entries.keys()):
            entries[key] = await self.sanitize_entry(entries[key], remove_data)
        return entries

    async def get_custom_node_subgraphs(self, loadedModules, force_reload=False):
        # if not forced to reload and cached, return cache
        if not force_reload and self.cached_custom_node_subgraphs is not None:
            return self.cached_custom_node_subgraphs
        # Load subgraphs from custom nodes
        subfolder = "subgraphs"
        subgraphs_dict: dict[SubgraphEntry] = {}

        for folder in folder_paths.get_folder_paths("custom_nodes"):
            pattern = os.path.join(folder, f"*/{subfolder}/*.json")
            matched_files = glob.glob(pattern)
            for file in matched_files:
                # replace backslashes with forward slashes
                file = file.replace('\\', '/')
                info: CustomNodeSubgraphEntryInfo = {
                    "node_pack": "custom_nodes." + file.split('/')[-3]
                }
                source = Source.custom_node
                # hash source + path to make sure id will be as unique as possible, but
                # reproducible across backend reloads
                id = hashlib.sha256(f"{source}{file}".encode()).hexdigest()
                entry: SubgraphEntry = {
                    "source": Source.custom_node,
                    "name": os.path.splitext(os.path.basename(file))[0],
                    "path": file,
                    "info": info,
                }
                subgraphs_dict[id] = entry
        self.cached_custom_node_subgraphs = subgraphs_dict
        return subgraphs_dict

    async def get_custom_node_subgraph(self, id: str, loadedModules):
        subgraphs = await self.get_custom_node_subgraphs(loadedModules)
        entry: SubgraphEntry = subgraphs.get(id, None)
        if entry is not None and entry.get('data', None) is None:
            await self.load_entry_data(entry)
        return entry

    def add_routes(self, routes, loadedModules):
        @routes.get("/global_subgraphs")
        async def get_global_subgraphs(request):
            subgraphs_dict = await self.get_custom_node_subgraphs(loadedModules)
            # NOTE: we may want to include other sources of global subgraphs such as templates in the future;
            # that's the reasoning for the current implementation
            return web.json_response(await self.sanitize_entries(subgraphs_dict, remove_data=True))

        @routes.get("/global_subgraphs/{id}")
        async def get_global_subgraph(request):
            id = request.match_info.get("id", None)
            subgraph = await self.get_custom_node_subgraph(id, loadedModules)
            return web.json_response(await self.sanitize_entry(subgraph))
