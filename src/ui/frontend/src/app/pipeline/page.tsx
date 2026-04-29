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

  // New UI state
  const [isRunModalOpen, setIsRunModalOpen] = useState(false);
  const [selectedPipelineId, setSelectedPipelineId] = useState<string | null>(null);

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
      .then((data: PipelineMeta[]) => {
        setExperiments(data);
        if (data.length > 0 && !selectedPipelineId) {
          setSelectedPipelineId(data[0].pipeline_id);
        }
      })
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
        setTimeout(() => {
          fetchExperiments();
          setSelectedPipelineId(data.pipeline_id);
        }, 2000);
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

  const activePipeline = experiments.find(e => e.pipeline_id === selectedPipelineId);

  return (
    <div className="animate-fade-in" style={{ paddingBottom: '4rem' }}>
      
      {/* Header */}
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: '2rem' }}>
        <div>
          <h1 className="page-title">Data Pipeline</h1>
          <p className="page-subtitle">View past pipeline executions or run a new processing pipeline.</p>
        </div>
        <button 
          className="btn btn-primary"
          onClick={() => setIsRunModalOpen(true)}
          style={{ padding: '0.75rem 1.5rem', fontSize: '1rem' }}
        >
          ▶ Run Data Pipeline
        </button>
      </div>

      {/* Main Split View */}
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 2fr', gap: '2rem' }}>
        
        {/* Left Column: Pipeline History List */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
          <h2 style={{ fontSize: '1.25rem', margin: 0 }}>Pipeline Runs</h2>
          {experiments.length === 0 ? (
            <div style={{ color: 'var(--text-tertiary)', fontStyle: 'italic', fontSize: '0.9rem' }}>No pipeline runs found.</div>
          ) : (
            <div style={{ display: 'flex', flexDirection: 'column', gap: '0.75rem', maxHeight: 'calc(100vh - 200px)', overflowY: 'auto', paddingRight: '0.5rem' }}>
              {experiments.map((exp, i) => (
                <div 
                  key={exp.pipeline_id} 
                  onClick={() => setSelectedPipelineId(exp.pipeline_id)}
                  style={{ 
                    cursor: 'pointer',
                    display: 'flex', 
                    alignItems: 'center', 
                    gap: '1rem', 
                    padding: '0.75rem 1rem', 
                    borderRadius: 'var(--radius-md)', 
                    background: selectedPipelineId === exp.pipeline_id ? 'rgba(59,130,246,0.1)' : 'var(--bg-primary)', 
                    border: `1px solid ${selectedPipelineId === exp.pipeline_id ? 'var(--accent-primary)' : 'var(--border-light)'}`,
                    transition: 'all 0.2s ease'
                  }}
                >
                  <div style={{ flex: 1 }}>
                    <div style={{ fontFamily: 'monospace', fontSize: '0.85rem', color: selectedPipelineId === exp.pipeline_id ? 'var(--accent-primary)' : 'var(--text-secondary)' }}>
                      {exp.pipeline_id} {i === 0 && <span style={{ marginLeft: '0.5rem', fontSize: '0.7rem', background: 'var(--accent-primary)', color: 'white', padding: '0.1rem 0.4rem', borderRadius: 'var(--radius-full)' }}>LATEST</span>}
                    </div>
                    <div style={{ fontSize: '0.8rem', color: 'var(--text-tertiary)', marginTop: '0.25rem' }}>
                      {getParamSummary(exp)}
                    </div>
                    <div style={{ fontSize: '0.7rem', color: 'var(--text-tertiary)', marginTop: '0.2rem' }}>
                      {formatDate(exp.created_at)}
                    </div>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>

        {/* Right Column: Pipeline Details */}
        <div>
          <div className="card glass" style={{ minHeight: '400px' }}>
            <h2 style={{ fontSize: '1.25rem', marginBottom: '1.5rem' }}>Pipeline Details</h2>
            {!activePipeline ? (
              <div style={{ color: 'var(--text-tertiary)', fontStyle: 'italic' }}>Select a pipeline to view details.</div>
            ) : (
              <div style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>
                
                {/* ID & Date */}
                <div style={{ display: 'flex', gap: '2rem', paddingBottom: '1rem', borderBottom: '1px solid var(--border-light)' }}>
                  <div>
                    <div style={{ fontSize: '0.75rem', color: 'var(--text-tertiary)', textTransform: 'uppercase', letterSpacing: '0.05em' }}>Pipeline ID</div>
                    <div style={{ fontFamily: 'monospace', fontSize: '1rem', color: 'var(--accent-primary)' }}>{activePipeline.pipeline_id}</div>
                  </div>
                  <div>
                    <div style={{ fontSize: '0.75rem', color: 'var(--text-tertiary)', textTransform: 'uppercase', letterSpacing: '0.05em' }}>Created At</div>
                    <div style={{ fontSize: '0.9rem', color: 'var(--text-secondary)' }}>{formatDate(activePipeline.created_at)}</div>
                  </div>
                </div>

                {/* Parameters Dump */}
                <div>
                  <h3 style={{ fontSize: '1rem', marginBottom: '1rem', color: 'var(--text-primary)' }}>Configuration Parameters</h3>
                  <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1rem' }}>
                    {Object.entries(activePipeline)
                      .filter(([k]) => !['pipeline_id', 'created_at'].includes(k))
                      .map(([stepKey, stepParams]) => {
                        // Special rendering for sampled_dates
                        if (stepKey === 'sampled_dates' && Array.isArray(stepParams)) {
                          const breakdown = { Weekdays: 0, Weekends: 0 };
                          stepParams.forEach((dStr: string) => {
                            if (dStr.length === 8) {
                              const year = parseInt(dStr.substring(0, 4));
                              const month = parseInt(dStr.substring(4, 6)) - 1;
                              const day = parseInt(dStr.substring(6, 8));
                              const date = new Date(year, month, day);
                              const dayOfWeek = date.getDay();
                              if (dayOfWeek === 0 || dayOfWeek === 6) breakdown.Weekends++;
                              else breakdown.Weekdays++;
                            }
                          });
                          return (
                            <div key={stepKey} style={{ background: 'var(--bg-secondary)', padding: '1rem', borderRadius: 'var(--radius-md)', border: '1px solid var(--border-light)' }}>
                              <div style={{ fontSize: '0.85rem', fontWeight: 600, color: 'var(--accent-secondary)', textTransform: 'uppercase', marginBottom: '0.75rem' }}>Sampled Dates</div>
                              <div style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem', fontSize: '0.85rem' }}>
                                <div style={{ display: 'flex', justifyContent: 'space-between' }}><span style={{ color: 'var(--text-tertiary)' }}>Total Dates</span><span style={{ fontWeight: 600 }}>{stepParams.length}</span></div>
                                <div style={{ display: 'flex', justifyContent: 'space-between' }}><span style={{ color: 'var(--text-tertiary)' }}>Weekdays</span><span style={{ fontFamily: 'monospace' }}>{breakdown.Weekdays}</span></div>
                                <div style={{ display: 'flex', justifyContent: 'space-between' }}><span style={{ color: 'var(--text-tertiary)' }}>Weekends</span><span style={{ fontFamily: 'monospace' }}>{breakdown.Weekends}</span></div>
                              </div>
                            </div>
                          );
                        }
                        
                        // Special rendering for sampled_stations
                        if (stepKey === 'sampled_stations' && Array.isArray(stepParams)) {
                          return (
                            <div key={stepKey} style={{ background: 'var(--bg-secondary)', padding: '1rem', borderRadius: 'var(--radius-md)', border: '1px solid var(--border-light)', gridColumn: '1 / -1' }}>
                              <div style={{ fontSize: '0.85rem', fontWeight: 600, color: 'var(--accent-secondary)', textTransform: 'uppercase', marginBottom: '0.75rem' }}>Sampled Stations ({stepParams.length})</div>
                              <div style={{ maxHeight: '200px', overflowY: 'auto', border: '1px solid var(--border-strong)', borderRadius: 'var(--radius-sm)' }}>
                                <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '0.8rem', textAlign: 'left' }}>
                                  <thead style={{ background: 'var(--bg-tertiary)', position: 'sticky', top: 0 }}>
                                    <tr>
                                      <th style={{ padding: '0.5rem', borderBottom: '1px solid var(--border-strong)' }}>Code</th>
                                      <th style={{ padding: '0.5rem', borderBottom: '1px solid var(--border-strong)' }}>Station Name</th>
                                    </tr>
                                  </thead>
                                  <tbody>
                                    {stepParams.map((s: any, i: number) => (
                                      <tr key={i} style={{ borderBottom: i < stepParams.length - 1 ? '1px solid var(--border-light)' : 'none' }}>
                                        <td style={{ padding: '0.4rem 0.5rem', fontFamily: 'monospace', color: 'var(--accent-primary)' }}>{s.code}</td>
                                        <td style={{ padding: '0.4rem 0.5rem', color: 'var(--text-secondary)' }}>{s.name}</td>
                                      </tr>
                                    ))}
                                  </tbody>
                                </table>
                              </div>
                            </div>
                          );
                        }

                        // Default rendering for everything else
                        return (
                          <div key={stepKey} style={{ background: 'var(--bg-secondary)', padding: '1rem', borderRadius: 'var(--radius-md)', border: '1px solid var(--border-light)' }}>
                            <div style={{ fontSize: '0.85rem', fontWeight: 600, color: 'var(--accent-secondary)', textTransform: 'uppercase', marginBottom: '0.5rem' }}>
                              {stepKey}
                            </div>
                            {typeof stepParams === 'object' && stepParams !== null ? (
                              <div style={{ display: 'flex', flexDirection: 'column', gap: '0.4rem' }}>
                                {Object.entries(stepParams).map(([k, v]) => (
                                  <div key={k} style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.8rem' }}>
                                    <span style={{ color: 'var(--text-tertiary)' }}>{k}</span>
                                    <span style={{ fontFamily: 'monospace', color: 'var(--text-primary)' }}>{JSON.stringify(v)}</span>
                                  </div>
                                ))}
                              </div>
                            ) : (
                              <div style={{ fontFamily: 'monospace', fontSize: '0.8rem', color: 'var(--text-primary)' }}>{JSON.stringify(stepParams)}</div>
                            )}
                          </div>
                        );
                      })}
                  </div>
                </div>

              </div>
            )}
          </div>
        </div>
      </div>

      {/* Run Modal */}
      {isRunModalOpen && (
        <div style={{ position: 'fixed', inset: 0, zIndex: 100, display: 'flex', alignItems: 'center', justifyContent: 'center', background: 'rgba(0,0,0,0.6)', backdropFilter: 'blur(4px)' }}>
          <div style={{ background: 'var(--bg-primary)', border: '1px solid var(--border-strong)', borderRadius: 'var(--radius-lg)', width: '90vw', height: '85vh', display: 'flex', flexDirection: 'column', overflow: 'hidden', boxShadow: '0 25px 50px -12px rgba(0,0,0,0.5)' }}>
            
            {/* Modal Header */}
            <div style={{ padding: '1rem 1.5rem', borderBottom: '1px solid var(--border-light)', display: 'flex', justifyContent: 'space-between', alignItems: 'center', background: 'var(--bg-secondary)' }}>
              <h2 style={{ margin: 0, fontSize: '1.25rem' }}>Configure & Run: Data Pipeline</h2>
              <button onClick={() => setIsRunModalOpen(false)} style={{ background: 'transparent', border: 'none', color: 'var(--text-tertiary)', fontSize: '1.5rem', cursor: 'pointer', lineHeight: 1 }}>&times;</button>
            </div>

            {/* Modal Body */}
            <div style={{ display: 'flex', flex: 1, overflow: 'hidden' }}>
              
              {/* Left Column: Config */}
              <div style={{ flex: 1, padding: '1.5rem', overflowY: 'auto', borderRight: '1px solid var(--border-light)' }}>
                
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

              </div>

              {/* Right Column: Terminal & Run Actions */}
              <div style={{ flex: 1, display: 'flex', flexDirection: 'column', background: '#0a0a0a', borderLeft: '1px solid #27272a' }}>
                <div style={{ padding: '0.75rem 1rem', borderBottom: '1px solid #27272a', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                  <span style={{ fontSize: '0.8rem', color: '#a1a1aa', textTransform: 'uppercase', letterSpacing: '0.05em' }}>Execution Log</span>
                  <span style={{ fontSize: '0.8rem', fontWeight: 600, color: status === 'running' ? 'var(--accent-warning)' : status === 'completed' ? 'var(--accent-success)' : '#71717a' }}>
                    {status.toUpperCase()}
                  </span>
                </div>
                
                <div style={{ flex: 1, padding: '1rem', overflowY: 'auto', fontFamily: 'monospace', fontSize: '0.85rem', color: '#e4e4e7', background: '#000' }}>
                  {logs.length === 0 && <div style={{ color: '#52525b', fontStyle: 'italic' }}>Output captured here...</div>}
                  {logs.map((log, i) => (
                    <div key={i} style={{ color: log.type === 'stderr' ? 'var(--accent-error)' : log.type === 'status' ? 'var(--accent-success)' : 'inherit', whiteSpace: 'pre-wrap', wordBreak: 'break-all', marginBottom: '0.2rem' }}>
                      {log.line || (log.type === 'status' ? `[Exited: ${log.exit_code}]` : '')}
                    </div>
                  ))}
                </div>
                
                {/* Actions */}
                <div style={{ padding: '1rem', borderTop: '1px solid #27272a', background: '#0a0a0a' }}>
                  <h3 style={{ fontSize: '0.8rem', marginBottom: '0.75rem', color: '#a1a1aa' }}>Steps to Run</h3>
                  <div style={{ display: 'flex', gap: '1rem', marginBottom: '1rem', flexWrap: 'wrap' }}>
                    {[1, 2, 3, 4].map(step => (
                      <label key={step} style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', cursor: 'pointer', color: '#e4e4e7', fontSize: '0.85rem' }}>
                        <input type="checkbox" checked={selectedSteps.includes(step)} onChange={() => toggleStep(step)} style={{ accentColor: 'var(--accent-primary)' }} />
                        <span>Step {step}</span>
                      </label>
                    ))}
                  </div>
                  <button
                    className="btn btn-primary"
                    style={{ width: '100%', padding: '0.875rem', fontSize: '1.05rem' }}
                    onClick={handleRun}
                    disabled={status === 'running' || selectedSteps.length === 0}
                  >
                    {status === 'running' ? 'Running Pipeline...' : 'Run Selected Steps'}
                  </button>
                </div>
              </div>

            </div>
          </div>
        </div>
      )}

    </div>
  );
}
