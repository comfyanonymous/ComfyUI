from server import PromptServer
from aiohttp import web
import time
import json

class MessageCancelled(Exception):
    pass

class Message:
    stash = {}
    messages = {}
    cancelled = False

    @classmethod
    def addMessage(cls, id, message):
        if message == '__cancel__':
            cls.messages = {}
            cls.cancelled = True
        elif message == '__start__':
            cls.messages = {}
            cls.stash = {}
            cls.cancelled = False
        else:
            cls.messages[str(id)] = message

    @classmethod
    def waitForMessage(cls, id, period=0.1, asList=False):
        sid = str(id)
        while not (sid in cls.messages) and not ("-1" in cls.messages):
            if cls.cancelled:
                cls.cancelled = False
                raise MessageCancelled()
            time.sleep(period)
        if cls.cancelled:
            cls.cancelled = False
            raise MessageCancelled()
        message = cls.messages.pop(str(id), None) or cls.messages.pop("-1")
        try:
            if asList:
                return [str(x.strip()) for x in message.split(",")]
            else:
                try:
                    return json.loads(message)
                except ValueError:
                    return message
        except ValueError:
            print( f"ERROR IN MESSAGE - failed to parse '${message}' as ${'comma separated list of strings' if asList else 'string'}")
            return [message] if asList else message


@PromptServer.instance.routes.post('/easyuse/message_callback')
async def message_callback(request):
    post = await request.post()
    Message.addMessage(post.get("id"), post.get("message"))
    return web.json_response({})