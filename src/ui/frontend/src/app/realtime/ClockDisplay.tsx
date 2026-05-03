'use client';

import React from 'react';
import { secToWallClock } from './useRealtimeSession';

interface ClockDisplayProps {
  clockSec: number;
  isRunning: boolean;
  dayType: string;
  dateStr: string;
}

const DAY_TYPE_LABELS: Record<string, { label: string; color: string }> = {
  WD: { label: 'Weekday',  color: '#004085' },
  SA: { label: 'Saturday', color: '#6f42c1' },
  SU: { label: 'Sunday',   color: '#e67e22' },
  HO: { label: 'Holiday',  color: '#c0392b' },
};

function formatDate(dateStr: string): string {
  if (dateStr.length !== 8) return dateStr;
  return `${dateStr.slice(0,4)}-${dateStr.slice(4,6)}-${dateStr.slice(6,8)}`;
}

export default function ClockDisplay({ clockSec, isRunning, dayType, dateStr }: ClockDisplayProps) {
  const timeStr = secToWallClock(clockSec);
  const [hh, mm, ss] = timeStr.split(':').map(Number);

  // Analog clock geometry
  const cx = 60, cy = 60, r = 54;

  // Hand angles (0 = 12 o'clock, clockwise)
  const hourAngle   = ((hh % 12) / 12 + mm / 720) * 2 * Math.PI - Math.PI / 2;
  const minuteAngle = (mm / 60 + ss / 3600) * 2 * Math.PI - Math.PI / 2;
  const secondAngle = (ss / 60) * 2 * Math.PI - Math.PI / 2;

  const hand = (angle: number, len: number) => ({
    x2: cx + Math.cos(angle) * len,
    y2: cy + Math.sin(angle) * len,
  });

  const { label: dtLabel, color: dtColor } = DAY_TYPE_LABELS[dayType] || { label: dayType, color: '#495057' };

  const pulseStyle = isRunning ? {
    animation: 'pulse 1s ease-in-out infinite',
  } : {};

  return (
    <div style={{
      display: 'flex',
      flexDirection: 'column',
      alignItems: 'center',
      gap: '0.5rem',
      padding: '1rem',
    }}>
      <style>{`
        @keyframes pulse {
          0%,100% { opacity: 1; }
          50%      { opacity: 0.6; }
        }
        @keyframes tickHand {
          from { transform-origin: 60px 60px; }
        }
      `}</style>

      {/* Analog clock face */}
      <svg width={120} height={120} viewBox="0 0 120 120">
        {/* Face */}
        <circle cx={cx} cy={cy} r={r} fill="#fff" stroke="#ced4da" strokeWidth={2} />

        {/* Hour ticks */}
        {Array.from({ length: 12 }, (_, i) => {
          const a = (i / 12) * 2 * Math.PI - Math.PI / 2;
          const x1 = cx + Math.cos(a) * 48;
          const y1 = cy + Math.sin(a) * 48;
          const x2 = cx + Math.cos(a) * 54;
          const y2 = cy + Math.sin(a) * 54;
          return <line key={i} x1={x1} y1={y1} x2={x2} y2={y2} stroke="#adb5bd" strokeWidth={1.5} />;
        })}

        {/* Hour hand */}
        <line x1={cx} y1={cy} {...hand(hourAngle, 30)}
          stroke="#212529" strokeWidth={4} strokeLinecap="round" />

        {/* Minute hand */}
        <line x1={cx} y1={cy} {...hand(minuteAngle, 44)}
          stroke="#495057" strokeWidth={2.5} strokeLinecap="round" />

        {/* Second hand */}
        <line x1={cx} y1={cy} {...hand(secondAngle, 48)}
          stroke={isRunning ? '#dc3545' : '#adb5bd'}
          strokeWidth={1.5} strokeLinecap="round"
          style={pulseStyle}
        />

        {/* Center dot */}
        <circle cx={cx} cy={cy} r={3} fill="#212529" />
      </svg>

      {/* Digital time */}
      <div style={{
        fontFamily: "'Courier New', monospace",
        fontSize: '1.5rem',
        fontWeight: 700,
        letterSpacing: '0.05em',
        color: 'var(--text-primary)',
        lineHeight: 1,
      }}>
        {timeStr}
      </div>

      {/* Date + day type badge */}
      <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: '0.25rem' }}>
        <span style={{ fontSize: '0.8rem', color: 'var(--text-secondary)' }}>
          {formatDate(dateStr)}
        </span>
        <span style={{
          fontSize: '0.7rem',
          fontWeight: 700,
          padding: '0.15rem 0.5rem',
          borderRadius: '2px',
          backgroundColor: dtColor,
          color: '#fff',
          textTransform: 'uppercase',
          letterSpacing: '0.06em',
        }}>
          {dtLabel}
        </span>
      </div>

      {/* Running indicator */}
      <div style={{
        width: 8, height: 8,
        borderRadius: '50%',
        backgroundColor: isRunning ? '#28a745' : '#6c757d',
        transition: 'background-color 0.2s',
        ...(isRunning ? pulseStyle : {}),
      }} />
    </div>
  );
}
