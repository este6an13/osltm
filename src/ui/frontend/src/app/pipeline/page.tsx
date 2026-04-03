'use client';

import { useState, useEffect } from 'react';
import { useWebSocket } from '@/hooks/useWebSocket';

export default function PipelinePage() {
  const [params, setParams] = useState<any>({});
  const [selectedSteps, setSelectedSteps] = useState<number[]>([1, 2, 3, 4]);
  const [runId, setRunId] = useState<string | null>(null);
  const { logs, status } = useWebSocket(runId);

  useEffect(() => {
    fetch('http://127.0.0.1:8000/api/pipeline/params')
      .then(res => res.json())
      .then(data => setParams(data))
      .catch(console.error);
  }, []);

  const handleRun = async () => {
    try {
      const res = await fetch('http://127.0.0.1:8000/api/pipeline/run', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ steps: selectedSteps, params })
      });
      const data = await res.json();
      if (data.run_id) {
        setRunId(data.run_id);
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
    setParams((prev: any) => ({
      ...prev,
      [section]: {
        ...prev[section],
        [key]: value
      }
    }));
  };

  return (
    <div className="animate-fade-in" style={{ display: 'flex', flexDirection: 'column', gap: '2rem' }}>
      <div>
        <h1 className="page-title">Data Pipeline</h1>
        <p className="page-subtitle">Configure parameters and run the 4-step data processing pipeline.</p>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '2rem' }}>
        <div className="card">
          <h2 style={{ marginBottom: '1.5rem', fontSize: '1.25rem' }}>Configuration</h2>
          
          {params.step1 && (
            <div className="form-group" style={{ marginBottom: '2rem' }}>
              <h3 style={{ fontSize: '1rem', marginBottom: '1rem', color: 'var(--accent-primary)' }}>Step 1: Sampling parameters</h3>
              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1rem' }}>
                <div className="form-group">
                  <label className="form-label">Start Date</label>
                  <input type="date" className="form-input" value={params.step1.start_date} onChange={(e) => handleParamChange('step1', 'start_date', e.target.value)} />
                </div>
                <div className="form-group">
                  <label className="form-label">End Date</label>
                  <input type="date" className="form-input" value={params.step1.end_date} onChange={(e) => handleParamChange('step1', 'end_date', e.target.value)} />
                </div>
                <div className="form-group">
                  <label className="form-label">Dates per Stratum</label>
                  <input type="number" className="form-input" value={params.step1.n_per_stratum} onChange={(e) => handleParamChange('step1', 'n_per_stratum', parseInt(e.target.value))} />
                </div>
              </div>
            </div>
          )}

          {params.step3 && (
            <div className="form-group">
              <h3 style={{ fontSize: '1rem', marginBottom: '1rem', color: 'var(--accent-primary)' }}>Step 3: Station sampling</h3>
              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1rem' }}>
                <div className="form-group">
                  <label className="form-label">N Stations</label>
                  <input type="number" className="form-input" value={params.step3.n_stations} onChange={(e) => handleParamChange('step3', 'n_stations', parseInt(e.target.value))} />
                </div>
              </div>
            </div>
          )}

          <div style={{ marginTop: '2rem', paddingTop: '1.5rem', borderTop: '1px solid var(--border-light)' }}>
            <h3 style={{ fontSize: '1rem', marginBottom: '1rem' }}>Steps to Run</h3>
            <div style={{ display: 'flex', gap: '1rem', marginBottom: '1.5rem', flexWrap: 'wrap' }}>
              {[1, 2, 3, 4].map(step => (
                <label key={step} style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', cursor: 'pointer' }}>
                  <input 
                    type="checkbox" 
                    checked={selectedSteps.includes(step)}
                    onChange={() => toggleStep(step)}
                    style={{ accentColor: 'var(--accent-primary)', width: '1rem', height: '1rem' }}
                  />
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

        <div className="card" style={{ display: 'flex', flexDirection: 'column' }}>
          <h2 style={{ marginBottom: '1.5rem', fontSize: '1.25rem' }}>Execution Log</h2>
          <div style={{ 
            flex: 1, 
            backgroundColor: '#000', 
            borderRadius: 'var(--radius-md)', 
            padding: '1rem',
            fontFamily: 'monospace',
            fontSize: '0.875rem',
            overflowY: 'auto',
            maxHeight: '500px',
            color: '#a1a1aa'
          }}>
            {logs.length === 0 && <div style={{ color: '#52525b', fontStyle: 'italic' }}>Awaiting execution...</div>}
            {logs.map((log, i) => (
              <div key={i} style={{ 
                color: log.type === 'stderr' ? 'var(--accent-error)' : 
                       log.type === 'status' ? 'var(--accent-success)' : 'inherit',
                whiteSpace: 'pre-wrap',
                marginBottom: '0.25rem'
              }}>
                {log.line || (log.type === 'status' ? `[Process Exited with Code: ${log.exit_code}]` : '')}
              </div>
            ))}
            {status === 'running' && <div style={{ marginTop: '1rem' }} className="animate-pulse">Loading...</div>}
          </div>
        </div>
      </div>
    </div>
  );
}
