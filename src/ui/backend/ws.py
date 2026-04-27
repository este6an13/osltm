import json
from collections import defaultdict
from typing import Dict, List
from fastapi import WebSocket, WebSocketDisconnect

class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, List[WebSocket]] = {}
        # Buffer messages so clients that connect slightly late still get full logs
        self.message_buffer: Dict[str, List[dict]] = defaultdict(list)

    async def connect(self, websocket: WebSocket, run_id: str):
        await websocket.accept()
        if run_id not in self.active_connections:
            self.active_connections[run_id] = []
        self.active_connections[run_id].append(websocket)

        # Replay any buffered messages to the newly connected client
        for buffered in self.message_buffer.get(run_id, []):
            await websocket.send_json(buffered)

    def disconnect(self, websocket: WebSocket, run_id: str):
        if run_id in self.active_connections:
            try:
                self.active_connections[run_id].remove(websocket)
                if not self.active_connections[run_id]:
                    del self.active_connections[run_id]
            except ValueError:
                pass

    async def send_json(self, data: dict, run_id: str):
        # Always buffer so late-connecting clients can replay
        self.message_buffer[run_id].append(data)

        if run_id in self.active_connections:
            dead = []
            for connection in self.active_connections[run_id]:
                try:
                    await connection.send_json(data)
                except Exception:
                    dead.append(connection)
            for d in dead:
                self.disconnect(d, run_id)

    def clear_buffer(self, run_id: str):
        """Call after a run completes to free memory."""
        self.message_buffer.pop(run_id, None)


manager = ConnectionManager()


async def websocket_endpoint(websocket: WebSocket, run_id: str):
    await manager.connect(websocket, run_id)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(websocket, run_id)
