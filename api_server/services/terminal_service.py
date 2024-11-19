from app.logger import on_flush
import os
import shutil


class TerminalService:
    def __init__(self, server):
        self.server = server
        self.cols = None
        self.rows = None
        self.subscriptions = set()
        on_flush(self.send_messages)

    def get_terminal_size(self):
        try:
            size = os.get_terminal_size()
            return (size.columns, size.lines)
        except OSError:
            try:
                size = shutil.get_terminal_size()
                return (size.columns, size.lines)
            except OSError:
                return (80, 24)  # fallback to 80x24

    def update_size(self):
        columns, lines = self.get_terminal_size()
        changed = False
        
        if columns != self.cols:
            self.cols = columns
            changed = True 

        if lines != self.rows:
            self.rows = lines
            changed = True

        if changed:
            return {"cols": self.cols, "rows": self.rows}

        return None

    def subscribe(self, client_id):
        self.subscriptions.add(client_id)

    def unsubscribe(self, client_id):
        self.subscriptions.discard(client_id)

    def send_messages(self, entries):
        if not len(entries) or not len(self.subscriptions):
            return
        
        new_size = self.update_size()
        
        for client_id in self.subscriptions.copy(): # prevent: Set changed size during iteration
            if client_id not in self.server.sockets:
                # Automatically unsub if the socket has disconnected
                self.unsubscribe(client_id)
                continue

            self.server.send_sync("logs", {"entries": entries, "size": new_size}, client_id)
