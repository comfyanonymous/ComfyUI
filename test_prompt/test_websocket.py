import asyncio
import websockets

async def websocket_client():
    group_id = "group1"
    user_id = "user1"
    
    async with websockets.connect("ws://eac76d8a629d:8000/ws/group_id/user_id") as websocket:
        while True:
            try:
                # 等待來自伺服器的消息
                temperature_data = await websocket.recv()
                print(f"Received: {temperature_data}")
            except websockets.exceptions.ConnectionClosedOK:
                print("Connection closed by server")
                break

# 使用 asyncio.run() 啟動 WebSocket 客戶端
asyncio.run(websocket_client())
