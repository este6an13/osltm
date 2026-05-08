'use client';
import React, { useEffect, useState, useRef } from 'react';
import dynamic from 'next/dynamic';
import ClockDisplay from './ClockDisplay';
import { useRealtimeSession, SessionConfig, ModelType, AdaptationMethod } from './useRealtimeSession';

const RealtimeChart = dynamic(() => import('./RealtimeChart'), { ssr: false });

const API = 'http://127.0.0.1:8000';
const SPEEDS = [1, 10, 100, 1000, 10000];
const MODEL_OPTIONS_MAP: Record<string, string> = {
  hawkes: 'Hawkes Process',
  lgcp_prior: 'LGCP Prior',
  lgcp_posterior: 'LGCP Posterior',
  avg_profile: 'Average Profile',
};
const ADAPT_OPTIONS = [
  { value: 'bayesian', label: 'Bayesian Gamma (B)' },
  { value: 'multiplicative', label: 'Multiplicative (A)' },
  { value: 'hawkes_kappa', label: 'Hawkes κ Re-weight (C)' },
  { value: 'trend', label: 'Trend Derivative (D)' },
];
const DAY_LABELS: Record<string, string> = { WD: 'Weekday', SA: 'Saturday', SU: 'Sunday', HO: 'Holiday' };

function Badge({ type }: { type: string }) {
  const colors: Record<string, string> = { WD: '#004085', SA: '#6f42c1', SU: '#e67e22', HO: '#c0392b' };
  return (
    <span style={{
      fontSize: '0.7rem', fontWeight: 700, padding: '0.15rem 0.5rem', borderRadius: 2,
      background: colors[type] || '#6c757d', color: '#fff', textTransform: 'uppercase', letterSpacing: '0.06em'
    }}>
      {DAY_LABELS[type] || type}
    </span>
  );
}

function Gauge({ ratio }: { ratio: number }) {
  const pct = Math.min(Math.max((ratio - 0.5) / 1.5, 0), 1);
  const color = ratio > 1.2 ? '#dc3545' : ratio < 0.8 ? '#0d6efd' : '#198754';
  const label = ratio > 1.2 ? '▲ Over-demand' : ratio < 0.8 ? '▼ Under-demand' : '✓ Aligned';
  return (
    <div style={{ marginBottom: '0.5rem' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.72rem', marginBottom: '0.2rem' }}>
        <span style={{ color }}>r = {ratio.toFixed(2)}</span>
        <span style={{ color, fontWeight: 600 }}>{label}</span>
      </div>
      <div style={{ height: 5, background: '#dee2e6', borderRadius: 2, overflow: 'hidden' }}>
        <div style={{ height: '100%', width: `${pct * 100}%`, background: color, transition: 'width 0.4s ease' }} />
      </div>
    </div>
  );
}

export default function RealtimePage() {
  const [state, actions] = useRealtimeSession();
  const [speed, setSpeedLocal] = useState(1);
  const [selectedStation, setSelectedStation] = useState<string>('');
  const [availDates, setAvailDates] = useState<Record<string, string>>({});
  const [cutoffDate, setCutoffDate] = useState<string | null>(null);
  const [inventory, setInventory] = useState<{ station_code: string; station_name: string; count_types: Record<string, Record<string, string[]>> }[]>([]);
  const [cfg, setCfg] = useState<SessionConfig>({
    date_str: '', day_type: 'WD', station_codes: [], model: '',
    adaptation_method: 'bayesian', clock_start_hhmm: 400,
    speed: 1, lookahead_min: 60, seed: 42, count_type: 'checkins',
  });
  const feedRef = useRef<HTMLDivElement>(null);
  const [feed, setFeed] = useState<{ time: string; sc: string; src: string; t: number }[]>([]);

  // Load dates + inventory on mount
  useEffect(() => {
    fetch(`${API}/api/realtime/dates`).then(r => r.json()).then(d => {
      setAvailDates(d.dates || {});
      setCutoffDate(d.cutoff_date || null);
    }).catch(() => { });
    fetch(`${API}/api/realtime/inventory`).then(r => r.json()).then(d => {
      const inv = d.inventory || [];
      setInventory(inv);
      if (inv.length > 0 && !cfg.station_codes.length) {
        const sc = inv[0].station_code;
        const ct = 'checkins';
        const dt = inv[0].count_types?.[ct] ? Object.keys(inv[0].count_types[ct])[0] || 'WD' : 'WD';
        const mod = inv[0].count_types?.[ct]?.[dt] ? inv[0].count_types[ct][dt][0] : '';
        setCfg(c => ({ ...c, station_codes: [sc], count_type: ct as any, day_type: dt, model: mod as any }));
      }
    }).catch(() => { });
  }, []);

  // Update selectedStation default
  useEffect(() => {
    if (state.meta?.station_codes?.length && !selectedStation) {
      setSelectedStation(state.meta.station_codes[0]);
    }
  }, [state.meta]);

  // Accumulate feed events
  useEffect(() => {
    if (!state.meta) return;
    const sc = selectedStation || state.meta.station_codes[0];
    if (!sc) return;
    const evs = state.modelEvents[sc] || [];
    const rev = state.realEvents[sc] || [];
    if (evs.length === 0 && rev.length === 0) return;
    const newEntries: typeof feed = [];
    const totalEvs = evs.slice(-3).map(t => ({ time: new Date().toLocaleTimeString(), sc, src: 'model', t }));
    const totalRev = rev.slice(-3).map(t => ({ time: new Date().toLocaleTimeString(), sc, src: 'real', t }));
    newEntries.push(...totalEvs, ...totalRev);
    if (newEntries.length === 0) return;
    setFeed(f => [...f, ...newEntries].slice(-120));
  }, [state.clockSec]);

  useEffect(() => {
    if (feedRef.current) feedRef.current.scrollTop = feedRef.current.scrollHeight;
  }, [feed]);

  const handleCreate = async () => {
    await actions.createSession({ ...cfg, speed });
  };

  const selectedInv = inventory.find(i => i.station_code === cfg.station_codes[0]);
  const currentCtData = selectedInv?.count_types?.[cfg.count_type] || {};
  const availableDayTypes = Object.keys(currentCtData);
  const availableModels = cfg.day_type ? (currentCtData[cfg.day_type] || []) : [];
  const validDates = Object.keys(availDates).filter(d => availDates[d] === cfg.day_type).sort().reverse();

  const sc = selectedStation || state.meta?.station_codes?.[0] || '';
  const forecastForSc = state.forecast[sc] || null;
  const adaptForSc = state.adaptation[sc] || null;
  const avgRatio = adaptForSc?.ratios
    ? adaptForSc.ratios.reduce((a, b) => a + b, 0) / adaptForSc.ratios.length
    : adaptForSc?.ratio ?? 1;

  return (
    <div className="animate-fade-in" style={{ display: 'flex', gap: '1.5rem', alignItems: 'flex-start' }}>
      <style>{`
        .rt-section-title { font-size:0.7rem; text-transform:uppercase; font-weight:700; color:var(--text-tertiary); letter-spacing:0.07em; margin-bottom:0.5rem; padding-bottom:0.25rem; border-bottom:1px solid var(--border-light); }
        .rt-feed-row { display:flex; gap:0.5rem; align-items:center; padding:0.15rem 0; border-bottom:1px solid #f8f9fa; font-size:0.72rem; }
        .rt-tag { font-size:0.65rem; font-weight:700; padding:0.1rem 0.35rem; border-radius:2px; text-transform:uppercase; }
      `}</style>

      {/* ── LEFT: Config panel ── */}
      <div style={{ width: 240, flexShrink: 0, display: 'flex', flexDirection: 'column', gap: '0.75rem' }}>
        <div className="card" style={{ padding: '1rem' }}>
          <div className="rt-section-title">Session Setup</div>

          <div className="form-group" style={{ marginBottom: '0.65rem' }}>
            <label className="form-label">Direction</label>
            <select className="form-select" value={cfg.count_type}
              onChange={e => {
                const ct = e.target.value as 'checkins' | 'checkouts';
                const sc = cfg.station_codes[0] || '';
                const inv = inventory.find(i => i.station_code === sc);
                const dt = inv?.count_types?.[ct] ? Object.keys(inv.count_types[ct])[0] || 'WD' : 'WD';
                const mod = inv?.count_types?.[ct]?.[dt] ? inv.count_types[ct][dt][0] : '';
                setCfg(c => ({ ...c, count_type: ct, day_type: dt, model: mod as any, date_str: '' }));
              }}>
              <option value="checkins">Check-ins</option>
              <option value="checkouts">Check-outs</option>
            </select>
          </div>

          <div className="form-group" style={{ marginBottom: '0.65rem' }}>
            <label className="form-label">Station</label>
            <select className="form-select"
              value={cfg.station_codes[0] || ''}
              onChange={e => {
                const sc = e.target.value;
                const inv = inventory.find(i => i.station_code === sc);
                const ct = cfg.count_type;
                const dt = inv?.count_types?.[ct] ? Object.keys(inv.count_types[ct])[0] || 'WD' : 'WD';
                const mod = inv?.count_types?.[ct]?.[dt] ? inv.count_types[ct][dt][0] : '';
                setCfg(c => ({ ...c, station_codes: [sc], day_type: dt, model: mod as any, date_str: '' }));
              }}>
              <option value="" disabled>Select a station...</option>
              {inventory.map(s => (
                <option key={s.station_code} value={s.station_code}>
                  ({s.station_code}) {s.station_name}
                </option>
              ))}
            </select>
          </div>

          <div className="form-group" style={{ marginBottom: '0.65rem' }}>
            <label className="form-label">Day Type</label>
            <select className="form-select" value={cfg.day_type}
              disabled={!cfg.station_codes.length}
              onChange={e => {
                const dt = e.target.value;
                const inv = inventory.find(i => i.station_code === cfg.station_codes[0]);
                const mod = inv?.count_types?.[cfg.count_type]?.[dt] ? inv.count_types[cfg.count_type][dt][0] : '';
                setCfg(c => ({ ...c, day_type: dt, model: mod as any, date_str: '' }));
              }}>
              {availableDayTypes.map(dt => (
                <option key={dt} value={dt}>{DAY_LABELS[dt] || dt}</option>
              ))}
            </select>
          </div>

          <div className="form-group" style={{ marginBottom: '0.65rem' }}>
            <label className="form-label">Model</label>
            <select className="form-select" value={cfg.model}
              disabled={!availableModels.length}
              onChange={e => setCfg(c => ({ ...c, model: e.target.value as ModelType }))}>
              {availableModels.map(m => (
                <option key={m} value={m}>{MODEL_OPTIONS_MAP[m] || m}</option>
              ))}
            </select>
            {cfg.model === 'hawkes_kappa' && (
              <span style={{ fontSize: '0.7rem', color: '#6c757d' }}>Option C available for Hawkes</span>
            )}
          </div>

          <div className="form-group" style={{ marginBottom: '0.65rem' }}>
            <label className="form-label">Date (Real Data)</label>
            <select className="form-select" value={cfg.date_str}
              onChange={e => setCfg(c => ({ ...c, date_str: e.target.value }))}>
              <option value="">-- No Real Data (Simulation Only) --</option>
              {validDates.map(d => {
                const label = cutoffDate ? (d <= cutoffDate ? `${d} (Training)` : `${d}  ✨ (Testing)`) : d;
                return <option key={d} value={d}>{label}</option>;
              })}
            </select>
          </div>

          <div className="form-group" style={{ marginBottom: '0.65rem' }}>
            <label className="form-label">Adaptation</label>
            <select className="form-select" value={cfg.adaptation_method}
              onChange={e => setCfg(c => ({ ...c, adaptation_method: e.target.value as AdaptationMethod }))}>
              {ADAPT_OPTIONS.filter(o => o.value !== 'hawkes_kappa' || cfg.model === 'hawkes').map(o => (
                <option key={o.value} value={o.value}>{o.label}</option>
              ))}
            </select>
          </div>

          <div className="form-group" style={{ marginBottom: '0.65rem' }}>
            <label className="form-label">Clock Start (HHMM)</label>
            <input className="form-input" type="number" min={400} max={2300} step={100}
              value={cfg.clock_start_hhmm}
              onChange={e => setCfg(c => ({ ...c, clock_start_hhmm: +e.target.value }))} />
          </div>

          <button className="btn btn-primary" style={{ width: '100%', marginBottom: '0.5rem' }}
            disabled={state.loading || !cfg.station_codes.length || !cfg.model}
            onClick={handleCreate}>
            {state.loading ? 'Loading…' : state.sessionId ? '↺ Reload Session' : '▶ Create Session'}
          </button>
          {state.error && (
            <div style={{
              fontSize: '0.72rem', color: 'var(--accent-error)', background: '#fff3f3',
              border: '1px solid #f5c6cb', borderRadius: 2, padding: '0.4rem 0.5rem', marginTop: '0.25rem'
            }}>
              {state.error}
            </div>
          )}
          {state.sessionId && (
            <button className="btn btn-secondary" style={{ width: '100%', marginTop: '0.25rem', fontSize: '0.75rem' }}
              onClick={actions.destroySession}>
              ✕ Destroy Session
            </button>
          )}
        </div>


      </div>

      {/* ── CENTER: Main view ── */}
      <div style={{ flex: 1, minWidth: 0, display: 'flex', flexDirection: 'column', gap: '1rem' }}>
        <h1 className="page-title" style={{ marginBottom: 0 }}>Real-Time Simulation</h1>
        <p className="page-subtitle" style={{ marginBottom: 0, marginTop: '0.25rem' }}>
          Live model vs. observed passenger arrivals with adaptive look-ahead correction.
        </p>

        {!state.sessionId ? (
          <div className="card" style={{ textAlign: 'center', padding: '3rem', color: 'var(--text-tertiary)' }}>
            <div style={{ fontSize: '2rem', marginBottom: '0.5rem' }}>⏱</div>
            <div style={{ fontWeight: 600 }}>Configure and create a session to begin.</div>
            <div style={{ fontSize: '0.85rem', marginTop: '0.5rem' }}>
              Select a date, station(s), and model from the panel on the left.
            </div>
          </div>
        ) : (
          <>
            {/* ── Controls bar ── */}
            <div className="card" style={{ padding: '1rem', display: 'flex', alignItems: 'center', gap: '1.5rem', flexWrap: 'wrap' }}>

              <div style={{ flex: 1, display: 'flex', flexDirection: 'column', gap: '0.75rem' }}>
                {/* Play/Pause/Reset */}
                <div style={{ display: 'flex', gap: '0.5rem', alignItems: 'center' }}>
                  {!state.isRunning ? (
                    <button className="btn btn-primary" onClick={actions.startClock} style={{ minWidth: 80 }}>▶ Play</button>
                  ) : (
                    <button className="btn btn-secondary" onClick={actions.pauseClock} style={{ minWidth: 80 }}>⏸ Pause</button>
                  )}
                  <button className="btn btn-secondary" onClick={actions.resetClock}>↺ Reset</button>

                  {/* Speed selector */}
                  <div style={{ display: 'flex', gap: '0.25rem', marginLeft: '0.5rem', alignItems: 'center' }}>
                    <span style={{ fontSize: '0.75rem', color: 'var(--text-tertiary)', marginRight: '0.25rem' }}>Speed:</span>
                    {SPEEDS.map(s => (
                      <button key={s} onClick={() => { setSpeedLocal(s); actions.setSpeed(s); }}
                        style={{
                          padding: '0.2rem 0.4rem', fontSize: '0.72rem', fontWeight: speed === s ? 700 : 400,
                          background: speed === s ? 'var(--accent-primary)' : 'var(--bg-secondary)',
                          color: speed === s ? '#fff' : 'var(--text-secondary)',
                          border: '1px solid var(--border-strong)', borderRadius: 2, cursor: 'pointer'
                        }}>
                        {s}×
                      </button>
                    ))}
                  </div>
                </div>

                {/* Session info chips */}
                <div style={{ display: 'flex', gap: '0.5rem', flexWrap: 'wrap', fontSize: '0.72rem' }}>
                  <span style={{ background: '#e9ecef', border: '1px solid #dee2e6', borderRadius: 2, padding: '0.15rem 0.4rem' }}>
                    {MODEL_OPTIONS_MAP[state.meta?.model || ''] || state.meta?.model}
                  </span>
                  <span style={{ background: '#e9ecef', border: '1px solid #dee2e6', borderRadius: 2, padding: '0.15rem 0.4rem' }}>
                    {ADAPT_OPTIONS.find(o => o.value === cfg.adaptation_method)?.label}
                  </span>
                  {state.hasRealData ? (
                    <span style={{ background: '#d1e7dd', border: '1px solid #a3cfbb', borderRadius: 2, padding: '0.15rem 0.4rem', color: '#0f5132' }}>
                      ✓ Real data loaded
                    </span>
                  ) : (
                    <span style={{ background: '#fff3cd', border: '1px solid #ffc107', borderRadius: 2, padding: '0.15rem 0.4rem', color: '#664d03' }}>
                      ⚠ No real data for this date
                    </span>
                  )}
                </div>
              </div>

              {/* Station switcher */}
              {(state.meta?.station_codes?.length || 0) > 1 && (
                <div>
                  <div style={{ fontSize: '0.72rem', color: 'var(--text-tertiary)', marginBottom: '0.25rem' }}>Station</div>
                  <select className="form-select" value={selectedStation}
                    onChange={e => setSelectedStation(e.target.value)}>
                    {state.meta?.station_codes.map(sc => (
                      <option key={sc} value={sc}>{sc}</option>
                    ))}
                  </select>
                </div>
              )}
            </div>

            {/* ── Event summary chips ── */}
            <div style={{ display: 'flex', gap: '0.75rem', flexWrap: 'wrap' }}>
              {(state.meta?.station_codes || [sc]).map(s => {
                const adapt = state.adaptation[s] || null;
                const r = adapt?.ratios
                  ? adapt.ratios.reduce((a, b) => a + b, 0) / adapt.ratios.length
                  : adapt?.ratio ?? 1;
                const modelCount = (state.modelEvents[s] || []).length;
                const correctedCount = Math.round(modelCount * r);
                return (
                  <div key={s} className="card" style={{ padding: '0.5rem 0.75rem', flex: '0 0 auto' }}>
                    <div style={{ fontSize: '0.7rem', color: 'var(--text-tertiary)', marginBottom: '0.2rem' }}>{s}</div>
                    <div style={{ display: 'flex', gap: '0.75rem', fontSize: '0.82rem' }}>
                      <span><span style={{ color: '#0d6efd', fontWeight: 700 }}>{modelCount}</span> model</span>
                      <span><span style={{ color: '#198754', fontWeight: 700 }}>{correctedCount}</span> corrected</span>
                      {state.hasRealData && (
                        <span><span style={{ color: '#dc3545', fontWeight: 700 }}>{(state.realEvents[s] || []).length}</span> real</span>
                      )}
                    </div>
                  </div>
                );
              })}
            </div>

            {/* ── Chart ── */}
            <div className="card" style={{ padding: '1rem' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.75rem' }}>
                <div style={{ fontWeight: 600, fontSize: '0.85rem' }}>Arrivals Over Time — {sc}</div>
                <div style={{ display: 'flex', gap: '1rem', fontSize: '0.72rem', color: 'var(--text-tertiary)' }}>
                  <span style={{ color: '#0d6efd' }}>── Model</span>
                  {state.hasRealData && <span style={{ color: '#dc3545' }}>── Real</span>}
                  <span style={{ color: '#198754' }}>- - Model (Corrected)</span>
                  <span style={{ color: '#0d6efd', opacity: 0.6 }}>- - Forecast (Raw)</span>
                  <span style={{ color: '#198754', opacity: 0.8 }}>- - Forecast (Corrected)</span>
                </div>
              </div>
              <RealtimeChart
                modelEvents={state.modelEvents[sc] || []}
                realEvents={state.realEvents[sc] || []}
                forecast={forecastForSc}
                clockSec={state.clockSec}
                hasRealData={state.hasRealData}
                currentRatio={avgRatio}
              />
            </div>

            {/* ── Live feed ── */}
            <div className="card" style={{ padding: '1rem' }}>
              <div className="rt-section-title">Live Event Feed</div>
              <div ref={feedRef} style={{ height: 140, overflowY: 'auto', fontFamily: "'Courier New', monospace" }}>
                {feed.length === 0 ? (
                  <div style={{ color: 'var(--text-tertiary)', fontSize: '0.78rem', padding: '0.5rem 0' }}>
                    Waiting for events…
                  </div>
                ) : feed.map((f, i) => (
                  <div key={i} className="rt-feed-row">
                    <span style={{ color: 'var(--text-tertiary)', minWidth: 72 }}>{f.time}</span>
                    <span style={{ color: 'var(--text-secondary)', minWidth: 60 }}>{f.sc}</span>
                    <span className="rt-tag" style={{
                      background: f.src === 'model' ? '#cfe2ff' : '#f8d7da',
                      color: f.src === 'model' ? '#084298' : '#842029',
                    }}>{f.src}</span>
                    <span style={{ color: 'var(--text-tertiary)', fontSize: '0.68rem' }}>
                      t={f.t.toFixed(0)}s
                    </span>
                  </div>
                ))}
              </div>
            </div>
          </>
        )}
      </div>

      {/* ── RIGHT: Adaptation panel ── */}
      {state.sessionId && (
        <div style={{ width: 200, flexShrink: 0, display: 'flex', flexDirection: 'column', gap: '0.75rem' }}>

          <div className="card" style={{ padding: '1rem', display: 'flex', justifyContent: 'center' }}>
            <ClockDisplay
              clockSec={state.clockSec}
              isRunning={state.isRunning}
              dayType={state.meta?.day_type || 'WD'}
              dateStr={cfg.date_str || undefined}
            />
          </div>

          <div className="card" style={{ padding: '1rem' }}>
            <div className="rt-section-title">Adaptation — {sc}</div>
            {adaptForSc ? (
              <>
                <div style={{ fontSize: '0.72rem', color: 'var(--text-tertiary)', marginBottom: '0.5rem' }}>
                  Method: <strong>{adaptForSc.method}</strong>
                </div>
                <Gauge ratio={avgRatio} />
                {adaptForSc.ratios && (
                  <div style={{ marginTop: '0.75rem' }}>
                    <div style={{ fontSize: '0.7rem', color: 'var(--text-tertiary)', marginBottom: '0.35rem' }}>
                      Correction ratios (per bin)
                    </div>
                    <div style={{ display: 'flex', flexWrap: 'wrap', gap: 2 }}>
                      {adaptForSc.ratios.slice(0, 32).map((r, i) => {
                        const c = r > 1.2 ? '#dc3545' : r < 0.8 ? '#0d6efd' : '#198754';
                        const h = Math.min(Math.max(Math.abs(r - 1) * 40, 2), 20);
                        return (
                          <div key={i} title={`Bin ${i}: r=${r.toFixed(2)}`}
                            style={{ width: 4, height: h, background: c, borderRadius: 1, alignSelf: 'flex-end' }} />
                        );
                      })}
                    </div>
                    <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.65rem', color: 'var(--text-tertiary)', marginTop: '0.2rem' }}>
                      <span>04:00</span><span>12:00</span>
                    </div>
                  </div>
                )}
                {adaptForSc.post_means && adaptForSc.prior_means && (
                  <div style={{ marginTop: '0.75rem', fontSize: '0.72rem' }}>
                    <div style={{ color: 'var(--text-tertiary)', marginBottom: '0.25rem' }}>
                      Last observed window
                    </div>
                    <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '0.15rem' }}>
                      <span style={{ color: '#0d6efd' }}>Model avg</span>
                      <strong>{(adaptForSc.prior_means.slice(-4).reduce((a, b) => a + b, 0) / 4).toFixed(1)}</strong>
                    </div>
                    <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                      <span style={{ color: '#198754' }}>Posterior avg</span>
                      <strong>{(adaptForSc.post_means.slice(-4).reduce((a, b) => a + b, 0) / 4).toFixed(1)}</strong>
                    </div>
                  </div>
                )}
              </>
            ) : (
              <div style={{ color: 'var(--text-tertiary)', fontSize: '0.78rem' }}>
                Adaptation state will appear once the clock advances.
              </div>
            )}
          </div>

          {/* Look-ahead summary */}
          {forecastForSc && (
            <div className="card" style={{ padding: '1rem' }}>
              <div className="rt-section-title">Look-ahead (+60 min)</div>
              <div style={{ fontSize: '0.75rem' }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '0.3rem' }}>
                  <span style={{ color: '#0d6efd' }}>Raw total</span>
                  <strong>{forecastForSc.model_raw.reduce((a, b) => a + b, 0).toFixed(0)}</strong>
                </div>
                <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                  <span style={{ color: '#198754' }}>Corrected total</span>
                  <strong>{forecastForSc.corrected.reduce((a, b) => a + b, 0).toFixed(0)}</strong>
                </div>
                <div style={{ marginTop: '0.5rem', height: 2, background: '#dee2e6', borderRadius: 1 }} />
                <div style={{ display: 'flex', justifyContent: 'space-between', marginTop: '0.3rem', color: 'var(--text-tertiary)', fontSize: '0.7rem' }}>
                  <span>Δ correction</span>
                  <span style={{ color: avgRatio > 1 ? '#dc3545' : '#198754', fontWeight: 700 }}>
                    {((avgRatio - 1) * 100).toFixed(0)}%
                  </span>
                </div>
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
