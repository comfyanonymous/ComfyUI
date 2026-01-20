from __future__ import annotations

from typing import TypedDict
import os
import folder_paths
import glob
from aiohttp import web
import hashlib


class Source:
    custom_node = "custom_node"
    templates = "templates"

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
        self.cached_blueprint_subgraphs: dict[SubgraphEntry] | None = None

    def _create_entry(self, file: str, source: str, node_pack: str) -> tuple[str, SubgraphEntry]:
        """Create a subgraph entry from a file path. Expects normalized path (forward slashes)."""
        entry_id = hashlib.sha256(f"{source}{file}".encode()).hexdigest()
        entry: SubgraphEntry = {
            "source": source,
            "name": os.path.splitext(os.path.basename(file))[0],
            "path": file,
            "info": {"node_pack": node_pack},
        }
        return entry_id, entry

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
        """Load subgraphs from custom nodes."""
        if not force_reload and self.cached_custom_node_subgraphs is not None:
            return self.cached_custom_node_subgraphs

        subgraphs_dict: dict[SubgraphEntry] = {}
        for folder in folder_paths.get_folder_paths("custom_nodes"):
            pattern = os.path.join(folder, "*/subgraphs/*.json")
            for file in glob.glob(pattern):
                file = file.replace('\\', '/')
                node_pack = "custom_nodes." + file.split('/')[-3]
                entry_id, entry = self._create_entry(file, Source.custom_node, node_pack)
                subgraphs_dict[entry_id] = entry

        self.cached_custom_node_subgraphs = subgraphs_dict
        return subgraphs_dict

    async def get_blueprint_subgraphs(self, force_reload=False):
        """Load subgraphs from the blueprints directory."""
        if not force_reload and self.cached_blueprint_subgraphs is not None:
            return self.cached_blueprint_subgraphs

        subgraphs_dict: dict[SubgraphEntry] = {}
        blueprints_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'blueprints')

        if os.path.exists(blueprints_dir):
            for file in glob.glob(os.path.join(blueprints_dir, "*.json")):
                file = file.replace('\\', '/')
                entry_id, entry = self._create_entry(file, Source.templates, "comfyui")
                subgraphs_dict[entry_id] = entry

        self.cached_blueprint_subgraphs = subgraphs_dict
        return subgraphs_dict

    async def get_all_subgraphs(self, loadedModules, force_reload=False):
        """Get all subgraphs from all sources (custom nodes and blueprints)."""
        custom_node_subgraphs = await self.get_custom_node_subgraphs(loadedModules, force_reload)
        blueprint_subgraphs = await self.get_blueprint_subgraphs(force_reload)
        return {**custom_node_subgraphs, **blueprint_subgraphs}

    async def get_subgraph(self, id: str, loadedModules):
        """Get a specific subgraph by ID from any source."""
        entry = (await self.get_all_subgraphs(loadedModules)).get(id)
        if entry is not None and entry.get('data') is None:
            await self.load_entry_data(entry)
        return entry

    def add_routes(self, routes, loadedModules):
        @routes.get("/global_subgraphs")
        async def get_global_subgraphs(request):
            subgraphs_dict = await self.get_all_subgraphs(loadedModules)
            return web.json_response(await self.sanitize_entries(subgraphs_dict, remove_data=True))

        @routes.get("/global_subgraphs/{id}")
        async def get_global_subgraph(request):
            id = request.match_info.get("id", None)
            subgraph = await self.get_subgraph(id, loadedModules)
            return web.json_response(await self.sanitize_entry(subgraph))
