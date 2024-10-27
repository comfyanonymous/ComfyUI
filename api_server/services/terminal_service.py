import asyncio
from app.logger import on_flush
import os


class TerminalService:
    def __init__(self, server):
        self.server = server
        self.cols = None
        self.rows = None
        self.subscriptions = set()
        on_flush(self.send_messages_sync)

    def update_size(self):
        sz = os.get_terminal_size()
        changed = False
        if sz.columns != self.cols:
            self.cols = sz.columns
            changed = True 

        if sz.lines != self.rows:
            self.rows = sz.lines
            changed = True

        if changed:
            return {"cols": self.cols, "rows": self.rows}

        return None

    def subscribe(self, client_id):
        self.subscriptions.add(client_id)

    def unsubscribe(self, client_id):
        self.subscriptions.discard(client_id)

    def send_messages_sync(self, entries):
        if not len(entries) or not len(self.subscriptions):
            return
        
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError: 
            loop = None

        if loop and loop.is_running():
            loop.create_task(self.send_messages(entries))
        else:
            asyncio.run(self.send_messages(entries))

    async def send_messages(self, entries):
        if not len(entries):
            return

        new_size = self.update_size()

        for client_id in self.subscriptions:
            if client_id not in self.server.sockets:
                # Automatically unsub if the socket has disconnected
                self.unsubscribe(client_id)
                continue

            await self.server.send_json(
                "logs", {"entries": entries, "size": new_size}, client_id)
