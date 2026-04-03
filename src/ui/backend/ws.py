import json
from typing import Dict, List
from fastapi import WebSocket, WebSocketDisconnect

class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, List[WebSocket]] = {}

    async def connect(self, websocket: WebSocket, run_id: str):
        await websocket.accept()
        if run_id not in self.active_connections:
            self.active_connections[run_id] = []
        self.active_connections[run_id].append(websocket)

    def disconnect(self, websocket: WebSocket, run_id: str):
        if run_id in self.active_connections:
            try:
                self.active_connections[run_id].remove(websocket)
                if not self.active_connections[run_id]:
                    del self.active_connections[run_id]
            except ValueError:
                pass

    async def send_message(self, message: str, run_id: str):
        if run_id in self.active_connections:
            for connection in self.active_connections[run_id]:
                await connection.send_text(message)

    async def send_json(self, data: dict, run_id: str):
        if run_id in self.active_connections:
            for connection in self.active_connections[run_id]:
                await connection.send_json(data)

manager = ConnectionManager()

async def websocket_endpoint(websocket: WebSocket, run_id: str):
    await manager.connect(websocket, run_id)
    try:
        while True:
            data = await websocket.receive_text()
            # We don't really expect clients to send much, mostly just listen
    except WebSocketDisconnect:
        manager.disconnect(websocket, run_id)
