import { useState, useEffect, useRef } from 'react';

export interface LogEntry {
  type: 'stdout' | 'stderr' | 'status';
  line?: string;
  status?: string;
  exit_code?: number;
  message?: string;
}

export function useWebSocket(runId: string | null) {
  const [logs, setLogs] = useState<LogEntry[]>([]);
  const [status, setStatus] = useState<'idle' | 'running' | 'completed' | 'failed' | 'error'>('idle');
  const wsRef = useRef<WebSocket | null>(null);

  useEffect(() => {
    if (!runId) return;

    setLogs([]);
    setStatus('running');

    const wsUrl = `ws://127.0.0.1:8000/ws/runs/${runId}`;
    const ws = new WebSocket(wsUrl);
    wsRef.current = ws;

    ws.onmessage = (event) => {
      try {
        const data: LogEntry = JSON.parse(event.data);
        setLogs(prev => [...prev, data]);
        
        if (data.type === 'status') {
          if (data.status === 'completed' || data.status === 'failed' || data.status === 'error') {
            setStatus(data.status as any);
          }
        }
      } catch (err) {
        console.error("Failed to parse WS message", err);
      }
    };

    ws.onclose = () => {
      wsRef.current = null;
    };

    return () => {
      if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
        wsRef.current.close();
      }
    };
  }, [runId]);

  return { logs, status, setLogs, setStatus };
}
