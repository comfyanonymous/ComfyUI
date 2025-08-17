from threading import Event
from server import PromptServer
from aiohttp import web
from comfy import model_management as mm
import time

class ChooserCancelled(Exception):
    pass

def get_chooser_cache():
    """获取选择器缓存"""
    if not hasattr(PromptServer.instance, '_easyuse_chooser_node'):
        PromptServer.instance._easyuse_chooser_node = {}
    return PromptServer.instance._easyuse_chooser_node

def cleanup_session_data(node_id):
    """清理会话数据"""
    node_data = get_chooser_cache()
    if node_id in node_data:
        session_keys = ["event", "selected", "images", "total_count", "cancelled"]
        for key in session_keys:
            if key in node_data[node_id]:
                del node_data[node_id][key]

def wait_for_chooser(id, images, mode, period=0.1):
    try:
        node_data = get_chooser_cache()

        if mode == "Keep Last Selection":
            if id in node_data and "last_selection" in node_data[id]:
                last_selection = node_data[id]["last_selection"]
                if last_selection and len(last_selection) > 0:
                    valid_indices = [idx for idx in last_selection if 0 <= idx < len(images)]
                    if valid_indices:
                        try:
                            PromptServer.instance.send_sync("easyuse-image-keep-selection", {
                                "id": id,
                                "selected": valid_indices
                            })
                        except Exception as e:
                            pass
                        cleanup_session_data(id)
                        indices_str = ','.join(str(i) for i in valid_indices)
                        return {"result": ([images[idx] for idx in valid_indices],)}

        if id in node_data:
            del node_data[id]

        event = Event()
        node_data[id] = {
            "event": event,
            "images": images,
            "selected": None,
            "total_count": len(images),
            "cancelled": False,
        }

        while id in node_data:
            node_info = node_data[id]
            if node_info.get("cancelled", False):
                cleanup_session_data(id)
                raise ChooserCancelled("Manual selection cancelled")

            if "selected" in node_info and node_info["selected"] is not None:
                break

            time.sleep(period)

        if id in node_data:
            node_info = node_data[id]
            selected_indices = node_info.get("selected")

            if selected_indices is not None and len(selected_indices) > 0:
                valid_indices = [idx for idx in selected_indices if 0 <= idx < len(images)]
                if valid_indices:
                    selected_images = [images[idx] for idx in valid_indices]

                    if id not in node_data:
                        node_data[id] = {}
                    node_data[id]["last_selection"] = valid_indices

                    cleanup_session_data(id)
                    indices_str = ','.join(str(i) for i in valid_indices)
                    return {"result": (selected_images, indices_str)}
                else:
                    cleanup_session_data(id)
                    return {"result": ([images[0]] if len(images) > 0 else [], "0" if len(images) > 0 else "")}
            else:
                cleanup_session_data(id)
                return {
                    "result": ([images[0]] if len(images) > 0 else [],)}
        else:
            return {"result": ([images[0]] if len(images) > 0 else [],)}

    except ChooserCancelled:
        raise mm.InterruptProcessingException()
    except Exception as e:
        node_data = get_chooser_cache()
        if id in node_data:
            cleanup_session_data(id)
        if 'image_list' in locals() and len(images) > 0:
            return {"result": ([images[0]])}
        else:
            return {"result": ([])}


@PromptServer.instance.routes.post('/easyuse/image_chooser_message')
async def handle_image_selection(request):
    try:
        data = await request.json()
        node_id = data.get("node_id")
        selected = data.get("selected", [])
        action = data.get("action")

        node_data = get_chooser_cache()

        if node_id not in node_data:
            return web.json_response({"code": -1, "error": "Node data does not exist"})

        try:
            node_info = node_data[node_id]

            if "total_count" not in node_info:
                return web.json_response({"code": -1, "error": "The node has been processed"})

            if action == "cancel":
                node_info["cancelled"] = True
                node_info["selected"] = []
            elif action == "select" and isinstance(selected, list):
                valid_indices = [idx for idx in selected if isinstance(idx, int) and 0 <= idx < node_info["total_count"]]
                if valid_indices:
                    node_info["selected"] = valid_indices
                    node_info["cancelled"] = False
                else:
                    return web.json_response({"code": -1, "error": "Invalid Selection Index"})
            else:
                return web.json_response({"code": -1, "error": "Invalid operation"})

            node_info["event"].set()
            return web.json_response({"code": 1})

        except Exception as e:
            if node_id in node_data and "event" in node_data[node_id]:
                node_data[node_id]["event"].set()
            return web.json_response({"code": -1, "message": "Processing Failed"})

    except Exception as e:
        return web.json_response({"code": -1, "message": "Request Failed"})
