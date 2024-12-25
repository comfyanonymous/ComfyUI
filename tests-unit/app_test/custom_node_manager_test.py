import pytest
from aiohttp import web
from unittest.mock import patch
from app.custom_node_manager import CustomNodeManager

pytestmark = (
    pytest.mark.asyncio
)  # This applies the asyncio mark to all test functions in the module

@pytest.fixture
def custom_node_manager():
    return CustomNodeManager()

@pytest.fixture
def app(custom_node_manager):
    app = web.Application()
    routes = web.RouteTableDef()
    custom_node_manager.add_routes(routes, app, [("ComfyUI-TestExtension1", "ComfyUI-TestExtension1")])
    app.add_routes(routes)
    return app

async def test_get_workflow_templates(aiohttp_client, app, tmp_path):
    client = await aiohttp_client(app)
    # Setup temporary custom nodes file structure with 1 workflow file
    custom_nodes_dir = tmp_path / "custom_nodes"
    example_workflows_dir = custom_nodes_dir / "ComfyUI-TestExtension1" / "example_workflows"
    example_workflows_dir.mkdir(parents=True)
    template_file = example_workflows_dir / "workflow1.json"
    template_file.write_text('')

    with patch('folder_paths.folder_names_and_paths', {
        'custom_nodes': ([str(custom_nodes_dir)], None)
    }):
        response = await client.get('/workflow_templates')
        assert response.status == 200
        workflows_dict = await response.json()
        assert isinstance(workflows_dict, dict)
        assert "ComfyUI-TestExtension1" in workflows_dict
        assert isinstance(workflows_dict["ComfyUI-TestExtension1"], list)
        assert workflows_dict["ComfyUI-TestExtension1"][0] == "workflow1"
