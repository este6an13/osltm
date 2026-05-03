'use client';

import React, { useMemo } from 'react';
import {
  ResponsiveContainer,
  ComposedChart,
  Area,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ReferenceLine,
} from 'recharts';
import { ForecastBin, secToHourFloat } from './useRealtimeSession';

interface RealtimeChartProps {
  modelEvents: number[];   // seconds from 04:00 (pre-accumulated)
  realEvents:  number[];   // seconds from 04:00
  forecast:    ForecastBin | null;
  clockSec:    number;
  hasRealData: boolean;
  stationName?: string;
  currentRatio?: number;
}

const BIN_SEC = 900; // 15 min
const START_H = 4;   // 04:00
const END_H   = 23;  // 23:00
const N_BINS  = Math.round(((END_H - START_H) * 3600) / BIN_SEC);

function binEvents(events: number[], nBins: number, binSec: number): number[] {
  const bins = new Array(nBins).fill(0);
  for (const t of events) {
    const idx = Math.floor(t / binSec);
    if (idx >= 0 && idx < nBins) bins[idx]++;
  }
  return bins;
}

function hourLabel(h: number): string {
  const hh = Math.floor(h);
  const mm = Math.round((h - hh) * 60);
  return `${String(hh).padStart(2,'0')}:${String(mm).padStart(2,'0')}`;
}

const CustomTooltip = ({ active, payload, label }: any) => {
  if (!active || !payload?.length) return null;
  return (
    <div style={{
      background: '#fff',
      border: '1px solid #ced4da',
      borderRadius: 2,
      padding: '0.5rem 0.75rem',
      fontSize: '0.78rem',
      boxShadow: '0 2px 4px rgba(0,0,0,0.1)',
    }}>
      <div style={{ fontWeight: 700, marginBottom: '0.25rem' }}>{label}</div>
      {payload.map((p: any) => (
        <div key={p.name} style={{ color: p.color, marginBottom: '0.1rem' }}>
          {p.name}: <strong>{typeof p.value === 'number' ? p.value.toFixed(1) : p.value}</strong>
        </div>
      ))}
    </div>
  );
};

export default function RealtimeChart({
  modelEvents, realEvents, forecast, clockSec, hasRealData, stationName, currentRatio,
}: RealtimeChartProps) {
  const data = useMemo(() => {
    const modelBins = binEvents(modelEvents, N_BINS, BIN_SEC);
    const realBins  = binEvents(realEvents,  N_BINS, BIN_SEC);

    // Build time axis
    const rows: Record<string, number | string | null>[] = [];

    // Forecast map: time_hours → {raw, corrected}
    const fcastMap: Record<string, { raw: number; corr: number }> = {};
    if (forecast && forecast.time_hours?.length) {
      forecast.time_hours.forEach((h, i) => {
        const key = hourLabel(h);
        fcastMap[key] = {
          raw:  forecast.model_raw[i]  ?? 0,
          corr: forecast.corrected[i]  ?? 0,
        };
      });
    }

    for (let i = 0; i < N_BINS; i++) {
      const h = START_H + (i * BIN_SEC) / 3600;
      const key = hourLabel(h);
      const fc = fcastMap[key];

      rows.push({
        time: key,
        'Model':     modelBins[i] || null,
        'Model (Corrected)': (currentRatio !== undefined && modelBins[i]) ? modelBins[i] * currentRatio : null,
        'Real':      hasRealData ? (realBins[i] || null) : undefined,
        'Forecast (Raw)':        fc?.raw  ?? null,
        'Forecast (Corrected)':  fc?.corr ?? null,
      });
    }
    return rows;
  }, [modelEvents, realEvents, forecast, hasRealData, currentRatio]);

  // Current clock hour for reference line
  const clockHour = secToHourFloat(clockSec);
  const nowLabel  = hourLabel(clockHour);

  return (
    <div style={{ width: '100%' }}>
      {stationName && (
        <div style={{ fontSize: '0.8rem', fontWeight: 600, color: 'var(--text-secondary)', marginBottom: '0.5rem' }}>
          {stationName}
        </div>
      )}
      <ResponsiveContainer width="100%" height={260}>
        <ComposedChart data={data} margin={{ top: 4, right: 16, left: 0, bottom: 0 }}>
          <CartesianGrid stroke="#dee2e6" strokeDasharray="3 3" vertical={false} />
          <XAxis
            dataKey="time"
            tick={{ fontSize: 10, fill: '#6c757d' }}
            interval={7}
            tickLine={false}
          />
          <YAxis
            tick={{ fontSize: 10, fill: '#6c757d' }}
            tickLine={false}
            axisLine={false}
            width={32}
          />
          <Tooltip content={<CustomTooltip />} />
          <Legend
            iconSize={10}
            wrapperStyle={{ fontSize: '0.75rem', paddingTop: '0.25rem' }}
          />

          {/* "Now" reference line */}
          <ReferenceLine
            x={nowLabel}
            stroke="#343a40"
            strokeWidth={1.5}
            strokeDasharray="4 2"
            label={{ value: 'Now', position: 'top', fontSize: 9, fill: '#343a40' }}
          />

          {/* Look-ahead shaded region — area behind forecast */}
          <Area
            dataKey="Forecast (Raw)"
            stroke="#0d6efd"
            strokeDasharray="5 3"
            strokeWidth={1.5}
            fill="#cfe2ff"
            fillOpacity={0.35}
            dot={false}
            activeDot={false}
            connectNulls={false}
          />
          <Area
            dataKey="Forecast (Corrected)"
            stroke="#198754"
            strokeDasharray="5 3"
            strokeWidth={1.5}
            fill="#d1e7dd"
            fillOpacity={0.35}
            dot={false}
            activeDot={false}
            connectNulls={false}
          />

          {/* Observed model events */}
          <Line
            dataKey="Model"
            stroke="#0d6efd"
            strokeWidth={2}
            dot={false}
            activeDot={{ r: 3 }}
            connectNulls={false}
          />

          <Line
            dataKey="Model (Corrected)"
            stroke="#198754"
            strokeDasharray="4 4"
            strokeWidth={2}
            dot={false}
            activeDot={false}
            connectNulls={false}
          />

          {/* Real observed events */}
          {hasRealData && (
            <Line
              dataKey="Real"
              stroke="#dc3545"
              strokeWidth={2}
              dot={false}
              activeDot={{ r: 3 }}
              connectNulls={false}
            />
          )}
        </ComposedChart>
      </ResponsiveContainer>
    </div>
  );
}
