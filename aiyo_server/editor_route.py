
from aiohttp import web
    
from aiyo_server.aiyo_server import AIYoServer


@AIYoServer.instance.routes.post("/editor/{flow_id}/save_workflow")
async def save_workflow(request):
    post = await request.post()
    workflow = post["workflow"]
    prompt = post["prompt"]
