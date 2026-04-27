'use client';

import { useState, useEffect } from 'react';
import { useWebSocket } from '@/hooks/useWebSocket';

interface PipelineMeta {
  pipeline_id: string;
  created_at?: string;
  step1?: any;
  step3?: any;
  [key: string]: any;
}

export default function PipelinePage() {
  const [params, setParams] = useState<any>({});
  const [selectedSteps, setSelectedSteps] = useState<number[]>([1, 2, 3, 4]);
  const [runId, setRunId] = useState<string | null>(null);
  const [lastPipelineId, setLastPipelineId] = useState<string | null>(null);
  const [experiments, setExperiments] = useState<PipelineMeta[]>([]);
  const { logs, status } = useWebSocket(runId);

  useEffect(() => {
    fetch('http://127.0.0.1:8000/api/pipeline/params')
      .then(res => res.json())
      .then(setParams)
      .catch(console.error);

    fetchExperiments();
  }, []);

  const fetchExperiments = () => {
    fetch('http://127.0.0.1:8000/api/pipeline/experiments')
      .then(res => res.json())
      .then(setExperiments)
      .catch(console.error);
  };

  const handleRun = async () => {
    try {
      const res = await fetch('http://127.0.0.1:8000/api/pipeline/run', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ steps: selectedSteps, params }),
      });
      const data = await res.json();
      if (data.run_id) setRunId(data.run_id);
      if (data.pipeline_id) {
        setLastPipelineId(data.pipeline_id);
        // Refresh history after a small delay to let the file be written
        setTimeout(fetchExperiments, 2000);
      }
    } catch (err) {
      console.error(err);
    }
  };

  const toggleStep = (step: number) => {
    setSelectedSteps(prev =>
      prev.includes(step) ? prev.filter(s => s !== step) : [...prev, step].sort()
    );
  };

  const handleParamChange = (section: string, key: string, value: any) => {
    setParams((prev: any) => ({ ...prev, [section]: { ...prev[section], [key]: value } }));
  };

  const handleRootParamChange = (key: string, value: any) => {
    setParams((prev: any) => ({ ...prev, [key]: value }));
  };

  const formatDate = (iso?: string) => iso ? new Date(iso).toLocaleString() : '';

  const getParamSummary = (p: PipelineMeta) => {
    const s1 = p.step1;
    const s3 = p.step3;
    const parts = [];
    if (s1?.start_date && s1?.end_date) parts.push(`${s1.start_date} → ${s1.end_date}`);
    if (s1?.n_per_stratum) parts.push(`${s1.n_per_stratum}/stratum`);
    if (s3?.n_stations) parts.push(`${s3.n_stations} stations`);
    return parts.join(' · ');
  };

  return (
    <div className="animate-fade-in" style={{ display: 'flex', flexDirection: 'column', gap: '2rem' }}>
      <div>
        <h1 className="page-title">Data Pipeline</h1>
        <p className="page-subtitle">Configure parameters and run the 4-step data processing pipeline.</p>
      </div>

      {/* Banner after a successful run */}
      {lastPipelineId && (
        <div style={{ padding: '1rem 1.5rem', borderRadius: 'var(--radius-md)', background: 'rgba(16,185,129,0.1)', border: '1px solid var(--accent-success)', display: 'flex', alignItems: 'center', gap: '1rem' }}>
          <span style={{ fontSize: '1.25rem' }}>✅</span>
          <div>
            <div style={{ fontWeight: 600, color: 'var(--accent-success)' }}>Pipeline Run Created</div>
            <div style={{ fontFamily: 'monospace', fontSize: '0.85rem', color: 'var(--text-secondary)', marginTop: '0.25rem' }}>{lastPipelineId}</div>
          </div>
        </div>
      )}

      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '2rem' }}>
        {/* Config Card */}
        <div className="card" style={{ overflow: 'visible' }}>
          <h2 style={{ marginBottom: '1.5rem', fontSize: '1.25rem' }}>Configuration</h2>

          {/* Global */}
          <div style={{ marginBottom: '2rem' }}>
            <h3 style={{ fontSize: '0.9rem', marginBottom: '1rem', color: 'var(--accent-secondary)', textTransform: 'uppercase', letterSpacing: '0.05em' }}>Global</h3>
            <div className="form-group" style={{ maxWidth: '50%' }}>
              <label className="form-label">Seed</label>
              <input type="number" className="form-input" value={params.seed ?? ''} onChange={e => handleRootParamChange('seed', parseInt(e.target.value))} />
            </div>
          </div>

          {/* Step 1 */}
          {params.step1 && (
            <div style={{ marginBottom: '2rem' }}>
              <h3 style={{ fontSize: '0.9rem', marginBottom: '1rem', color: 'var(--accent-primary)', textTransform: 'uppercase', letterSpacing: '0.05em' }}>Step 1 — Date Sampling</h3>
              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1rem' }}>
                <div className="form-group">
                  <label className="form-label">Start Date</label>
                  <input type="date" className="form-input" value={params.step1.start_date || ''} onChange={e => handleParamChange('step1', 'start_date', e.target.value)} />
                </div>
                <div className="form-group">
                  <label className="form-label">End Date</label>
                  <input type="date" className="form-input" value={params.step1.end_date || ''} onChange={e => handleParamChange('step1', 'end_date', e.target.value)} />
                </div>
                <div className="form-group">
                  <label className="form-label">Dates per Stratum</label>
                  <input type="number" className="form-input" value={params.step1.n_per_stratum ?? ''} onChange={e => handleParamChange('step1', 'n_per_stratum', parseInt(e.target.value))} />
                </div>
                <div className="form-group">
                  <label className="form-label">Days Offset</label>
                  <input type="number" className="form-input" value={params.step1.days_offset ?? ''} onChange={e => handleParamChange('step1', 'days_offset', parseInt(e.target.value))} />
                </div>
              </div>
            </div>
          )}

          {/* Step 2 */}
          {params.step2 && (
            <div style={{ marginBottom: '2rem' }}>
              <h3 style={{ fontSize: '0.9rem', marginBottom: '1rem', color: 'var(--accent-primary)', textTransform: 'uppercase', letterSpacing: '0.05em' }}>Step 2 — Download Data</h3>
              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1rem' }}>
                <div className="form-group">
                  <label className="form-label">Data Type</label>
                  <select className="form-input" value={params.step2.type || 'both'} onChange={e => handleParamChange('step2', 'type', e.target.value)}>
                    <option value="both">Both</option>
                    <option value="checkins">Check-ins only</option>
                    <option value="checkouts">Check-outs only</option>
                  </select>
                </div>
                <div className="form-group" style={{ justifyContent: 'flex-end' }}>
                  <label style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', cursor: 'pointer', fontSize: '0.875rem', color: 'var(--text-secondary)' }}>
                    <input type="checkbox" checked={params.step2.force_redownload || false} onChange={e => handleParamChange('step2', 'force_redownload', e.target.checked)} style={{ accentColor: 'var(--accent-primary)', width: '1rem', height: '1rem' }} />
                    Force Redownload
                  </label>
                </div>
              </div>
            </div>
          )}

          {/* Step 3 */}
          {params.step3 && (
            <div style={{ marginBottom: '2rem' }}>
              <h3 style={{ fontSize: '0.9rem', marginBottom: '1rem', color: 'var(--accent-primary)', textTransform: 'uppercase', letterSpacing: '0.05em' }}>Step 3 — Station Sampling</h3>
              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1rem' }}>
                <div className="form-group">
                  <label className="form-label">N Stations</label>
                  <input type="number" className="form-input" value={params.step3.n_stations ?? ''} onChange={e => handleParamChange('step3', 'n_stations', parseInt(e.target.value))} />
                </div>
                <div className="form-group">
                  <label className="form-label">N Files</label>
                  <input type="number" className="form-input" value={params.step3.n_files ?? ''} onChange={e => handleParamChange('step3', 'n_files', parseInt(e.target.value))} />
                </div>
              </div>
            </div>
          )}

          {/* Step 4 */}
          {params.step4 && (
            <div style={{ marginBottom: '2rem' }}>
              <h3 style={{ fontSize: '0.9rem', marginBottom: '1rem', color: 'var(--accent-primary)', textTransform: 'uppercase', letterSpacing: '0.05em' }}>Step 4 — Populate Counts</h3>
              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1rem' }}>
                <div className="form-group">
                  <label className="form-label">Time Min (HHMM)</label>
                  <input type="number" className="form-input" value={params.step4.time_min ?? ''} onChange={e => handleParamChange('step4', 'time_min', parseInt(e.target.value))} />
                </div>
                <div className="form-group">
                  <label className="form-label">Time Max (HHMM)</label>
                  <input type="number" className="form-input" value={params.step4.time_max ?? ''} onChange={e => handleParamChange('step4', 'time_max', parseInt(e.target.value))} />
                </div>
                <div className="form-group">
                  <label className="form-label">Time Step (min)</label>
                  <input type="number" className="form-input" value={params.step4.time_step ?? ''} onChange={e => handleParamChange('step4', 'time_step', parseInt(e.target.value))} />
                </div>
                <div className="form-group">
                  <label className="form-label">Delta (min)</label>
                  <input type="number" className="form-input" value={params.step4.delta_minutes ?? ''} onChange={e => handleParamChange('step4', 'delta_minutes', parseInt(e.target.value))} />
                </div>
              </div>
            </div>
          )}

          <div style={{ paddingTop: '1.5rem', borderTop: '1px solid var(--border-light)' }}>
            <h3 style={{ fontSize: '0.9rem', marginBottom: '1rem' }}>Steps to Run</h3>
            <div style={{ display: 'flex', gap: '1rem', marginBottom: '1.5rem', flexWrap: 'wrap' }}>
              {[1, 2, 3, 4].map(step => (
                <label key={step} style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', cursor: 'pointer' }}>
                  <input type="checkbox" checked={selectedSteps.includes(step)} onChange={() => toggleStep(step)} style={{ accentColor: 'var(--accent-primary)', width: '1rem', height: '1rem' }} />
                  <span>Step {step}</span>
                </label>
              ))}
            </div>
            <button
              className="btn btn-primary"
              style={{ width: '100%', padding: '0.75rem' }}
              onClick={handleRun}
              disabled={status === 'running' || selectedSteps.length === 0}
            >
              {status === 'running' ? 'Running Pipeline...' : 'Run Selected Steps'}
            </button>
          </div>
        </div>

        {/* Terminal */}
        <div className="card" style={{ display: 'flex', flexDirection: 'column' }}>
          <h2 style={{ marginBottom: '1.5rem', fontSize: '1.25rem' }}>Execution Log</h2>
          <div style={{ flex: 1, backgroundColor: '#000', borderRadius: 'var(--radius-md)', padding: '1rem', fontFamily: 'monospace', fontSize: '0.875rem', overflowY: 'auto', maxHeight: '500px', color: '#a1a1aa' }}>
            {logs.length === 0 && <div style={{ color: '#52525b', fontStyle: 'italic' }}>Awaiting execution...</div>}
            {logs.map((log, i) => (
              <div key={i} style={{ color: log.type === 'stderr' ? 'var(--accent-error)' : log.type === 'status' ? 'var(--accent-success)' : 'inherit', whiteSpace: 'pre-wrap', marginBottom: '0.25rem' }}>
                {log.line || (log.type === 'status' ? `[Process exited: ${log.exit_code}]` : '')}
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Pipeline History */}
      {experiments.length > 0 && (
        <div className="card">
          <h2 style={{ marginBottom: '1.25rem', fontSize: '1.25rem' }}>Pipeline Run History</h2>
          <div style={{ display: 'flex', flexDirection: 'column', gap: '0.75rem' }}>
            {experiments.map((exp, i) => (
              <div key={exp.pipeline_id} style={{ display: 'flex', alignItems: 'center', gap: '1rem', padding: '0.75rem 1rem', borderRadius: 'var(--radius-md)', background: i === 0 ? 'rgba(59,130,246,0.08)' : 'var(--bg-primary)', border: `1px solid ${i === 0 ? 'var(--accent-primary)' : 'var(--border-light)'}` }}>
                <div style={{ flex: 1 }}>
                  <div style={{ fontFamily: 'monospace', fontSize: '0.85rem', color: i === 0 ? 'var(--accent-primary)' : 'var(--text-secondary)' }}>
                    {exp.pipeline_id} {i === 0 && <span style={{ marginLeft: '0.5rem', fontSize: '0.7rem', background: 'var(--accent-primary)', color: 'white', padding: '0.1rem 0.4rem', borderRadius: 'var(--radius-full)' }}>LATEST</span>}
                  </div>
                  <div style={{ fontSize: '0.8rem', color: 'var(--text-tertiary)', marginTop: '0.25rem' }}>
                    {getParamSummary(exp)} · {formatDate(exp.created_at)}
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
