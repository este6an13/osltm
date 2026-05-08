'use client';

import { useCallback, useEffect, useRef, useState } from 'react';

const API = 'http://127.0.0.1:8000';
const WS = 'ws://127.0.0.1:8000';

// seconds from window start that represent the full observation day
const WINDOW_START_SEC = (4 * 3600);   // 04:00
const WINDOW_END_SEC = (23 * 3600);  // 23:00
const WINDOW_TOTAL = WINDOW_END_SEC - WINDOW_START_SEC;

export type ModelType = 'hawkes' | 'lgcp_prior' | 'lgcp_posterior' | 'avg_profile';
export type AdaptationMethod = 'bayesian' | 'multiplicative' | 'hawkes_kappa' | 'trend';

export interface SessionConfig {
  date_str: string;
  day_type: string;
  station_codes: string[];
  model: ModelType | '';
  adaptation_method: AdaptationMethod;
  clock_start_hhmm: number;
  speed: number;
  lookahead_min: number;
  seed: number;
  count_type: 'checkins' | 'checkouts';
}

export interface ForecastBin {
  time_hours: number[];
  model_raw: number[];
  corrected: number[];
}

export interface AdaptationState {
  method: string;
  ratios?: number[];
  ratio?: number;
  prior_means?: number[];
  post_means?: number[];
}

export interface TickPayload {
  t: number;
  model_events: Record<string, { model: number[]; real: number[] }>;
  forecast: Record<string, ForecastBin>;
  adaptation: Record<string, AdaptationState>;
  has_real_data: boolean;
}

export interface SessionMeta {
  session_id: string;
  day_type: string;
  has_real_data: boolean;
  station_codes: string[];
  model: string;
  model_event_counts: Record<string, number>;
  real_event_counts: Record<string, number>;
  clock_start_sec: number;
}

export interface RealtimeState {
  // Session
  sessionId: string | null;
  meta: SessionMeta | null;
  loading: boolean;
  error: string | null;

  // Clock
  clockSec: number;        // current position in window (seconds from 04:00)
  isRunning: boolean;

  // Accumulated events (per station)
  modelEvents: Record<string, number[]>;
  realEvents: Record<string, number[]>;

  // Latest tick's forecast & adaptation
  forecast: Record<string, ForecastBin>;
  adaptation: Record<string, AdaptationState>;

  // Helpers
  hasRealData: boolean;
}

export interface RealtimeActions {
  createSession: (cfg: SessionConfig) => Promise<void>;
  destroySession: () => Promise<void>;
  startClock: () => void;
  pauseClock: () => void;
  resetClock: () => void;
  setSpeed: (s: number) => void;
}

const TICK_MS = 250; // UI tick interval

export function useRealtimeSession(): [RealtimeState, RealtimeActions] {
  const [state, setState] = useState<RealtimeState>({
    sessionId: null,
    meta: null,
    loading: false,
    error: null,
    clockSec: 0,
    isRunning: false,
    modelEvents: {},
    realEvents: {},
    forecast: {},
    adaptation: {},
    hasRealData: false,
  });

  const wsRef = useRef<WebSocket | null>(null);
  const timerRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const speedRef = useRef<number>(1.0);
  const clockSecRef = useRef<number>(0);
  const sessionIdRef = useRef<string | null>(null);

  // -------------------------------------------------------------------------
  // WebSocket management
  // -------------------------------------------------------------------------
  const connectWs = useCallback((sessionId: string, startSec: number) => {
    if (wsRef.current) {
      wsRef.current.close();
    }
    const ws = new WebSocket(`${WS}/ws/realtime/${sessionId}`);
    wsRef.current = ws;

    ws.onmessage = (ev) => {
      try {
        const data: TickPayload = JSON.parse(ev.data);
        if ((data as any).error) {
          setState(s => ({ ...s, error: (data as any).error }));
          return;
        }

        setState(s => {
          // Accumulate events
          const newModel: Record<string, number[]> = { ...s.modelEvents };
          const newReal: Record<string, number[]> = { ...s.realEvents };

          for (const [sc, ev] of Object.entries(data.model_events)) {
            newModel[sc] = [...(newModel[sc] || []), ...ev.model];
            newReal[sc] = [...(newReal[sc] || []), ...ev.real];
          }

          return {
            ...s,
            modelEvents: newModel,
            realEvents: newReal,
            forecast: data.forecast,
            adaptation: data.adaptation,
            hasRealData: data.has_real_data,
            clockSec: data.t,
          };
        });
      } catch {
        /* ignore parse errors */
      }
    };

    ws.onerror = () => {
      setState(s => ({ ...s, error: 'WebSocket connection error' }));
    };

    clockSecRef.current = startSec;
  }, []);

  // -------------------------------------------------------------------------
  // Clock tick loop
  // -------------------------------------------------------------------------
  const sendTick = useCallback(() => {
    if (!wsRef.current || wsRef.current.readyState !== WebSocket.OPEN) return;
    const newSec = clockSecRef.current + (TICK_MS / 1000) * speedRef.current;
    if (newSec >= WINDOW_TOTAL) {
      // End of day — pause
      setState(s => ({ ...s, isRunning: false }));
      if (timerRef.current) clearInterval(timerRef.current);
      timerRef.current = null;
      return;
    }
    clockSecRef.current = newSec;
    setState(s => ({ ...s, clockSec: newSec }));
    wsRef.current.send(JSON.stringify({ t: newSec }));
  }, []);

  // -------------------------------------------------------------------------
  // Actions
  // -------------------------------------------------------------------------
  const createSession = useCallback(async (cfg: SessionConfig) => {
    setState(s => ({ ...s, loading: true, error: null }));
    try {
      const res = await fetch(`${API}/api/realtime/session`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(cfg),
      });
      if (!res.ok) {
        const err = await res.json();
        throw new Error(err.detail || 'Failed to create session');
      }
      const meta: SessionMeta = await res.json();
      sessionIdRef.current = meta.session_id;
      const startSec = meta.clock_start_sec ?? 0;
      clockSecRef.current = startSec;

      setState(s => ({
        ...s,
        sessionId: meta.session_id,
        meta,
        loading: false,
        error: null,
        clockSec: startSec,
        isRunning: false,
        modelEvents: {},
        realEvents: {},
        forecast: {},
        adaptation: {},
        hasRealData: meta.has_real_data,
      }));

      connectWs(meta.session_id, startSec);
    } catch (e: any) {
      setState(s => ({ ...s, loading: false, error: e.message }));
    }
  }, [connectWs]);

  const destroySession = useCallback(async () => {
    if (timerRef.current) { clearInterval(timerRef.current); timerRef.current = null; }
    if (wsRef.current) { wsRef.current.close(); wsRef.current = null; }
    const sid = sessionIdRef.current;
    if (sid) {
      try { await fetch(`${API}/api/realtime/session/${sid}`, { method: 'DELETE' }); } catch { /* ignore */ }
    }
    sessionIdRef.current = null;
    setState(s => ({
      ...s,
      sessionId: null, meta: null, isRunning: false,
      modelEvents: {}, realEvents: {}, forecast: {}, adaptation: {},
      clockSec: 0, error: null,
    }));
  }, []);

  const startClock = useCallback(() => {
    if (timerRef.current) return;
    setState(s => ({ ...s, isRunning: true }));
    timerRef.current = setInterval(sendTick, TICK_MS);
  }, [sendTick]);

  const pauseClock = useCallback(() => {
    if (timerRef.current) { clearInterval(timerRef.current); timerRef.current = null; }
    setState(s => ({ ...s, isRunning: false }));
  }, []);

  const resetClock = useCallback(() => {
    if (timerRef.current) { clearInterval(timerRef.current); timerRef.current = null; }
    const startSec = state.meta?.clock_start_sec ?? 0;
    clockSecRef.current = startSec;
    setState(s => ({
      ...s,
      isRunning: false,
      clockSec: startSec,
      modelEvents: {},
      realEvents: {},
      forecast: {},
      adaptation: {},
    }));
  }, [state.meta]);

  const setSpeed = useCallback((s: number) => {
    speedRef.current = s;
  }, []);

  // cleanup on unmount
  useEffect(() => {
    return () => {
      if (timerRef.current) clearInterval(timerRef.current);
      if (wsRef.current) wsRef.current.close();
    };
  }, []);

  return [
    state,
    { createSession, destroySession, startClock, pauseClock, resetClock, setSpeed },
  ];
}

// -------------------------------------------------------------------------
// Utility: convert seconds-from-04:00 to wall-clock HH:MM:SS string
// -------------------------------------------------------------------------
export function secToWallClock(sec: number): string {
  const totalSec = Math.floor(sec) + WINDOW_START_SEC;
  const h = Math.floor(totalSec / 3600) % 24;
  const m = Math.floor((totalSec % 3600) / 60);
  const s = totalSec % 60;
  return `${String(h).padStart(2, '0')}:${String(m).padStart(2, '0')}:${String(s).padStart(2, '0')}`;
}

export function secToHourFloat(sec: number): number {
  return (sec + WINDOW_START_SEC) / 3600;
}
