'use client';

import { useState, useEffect } from 'react';
import { useWebSocket } from '@/hooks/useWebSocket';
import ResultsViewer from '@/components/ResultsViewer';

interface PipelineMeta {
  pipeline_id: string;
  created_at?: string;
  params?: any;
}

interface UpstreamExp {
  experiment_id: string;
  pipeline_id?: string;
  script?: string;
  created_at?: string;
  exit_code?: number;
}

export default function AnalysisPage({ category }: { category: 'profiles' | 'intensity' | 'models/hawkes' | 'models/lgcp' }) {
  const [scripts, setScripts] = useState<any>({});
  const [selectedScript, setSelectedScript] = useState<string | null>(null);
  const [scriptParams, setScriptParams] = useState<any>({});

  // Pipelines
  const [pipelines, setPipelines] = useState<PipelineMeta[]>([]);
  
  // Run Modal State
  const [isRunModalOpen, setIsRunModalOpen] = useState(false);
  const [runPipeline, setRunPipeline] = useState<PipelineMeta | null>(null);
  const [stations, setStations] = useState<any[]>([]);
  const [upstreamExps, setUpstreamExps] = useState<UpstreamExp[]>([]);
  const [selectedUpstreamExp, setSelectedUpstreamExp] = useState<string | null>(null);

  // Run state
  const [runId, setRunId] = useState<string | null>(null);
  const [lastExperimentId, setLastExperimentId] = useState<string | undefined>(undefined);
  const { logs, status } = useWebSocket(runId);

  useEffect(() => {
    fetch('http://127.0.0.1:8000/api/analysis')
      .then(res => res.json())
      .then(data => {
        const filtered = Object.entries(data).reduce((acc: any, [key, val]: [string, any]) => {
          if (val.category === category) acc[key] = val;
          return acc;
        }, {});
        setScripts(filtered);
        const keys = Object.keys(filtered);
        if (keys.length > 0 && !selectedScript) {
          handleSelectScript(keys[0], filtered);
        }
      })
      .catch(console.error);

    fetch('http://127.0.0.1:8000/api/pipeline/experiments')
      .then(res => res.json())
      .then((data: PipelineMeta[]) => {
        setPipelines(data);
        if (data.length > 0) setRunPipeline(data[0]);
      })
      .catch(console.error);
  }, [category]);

  // Reload stations when run pipeline changes
  useEffect(() => {
    const url = runPipeline
      ? `http://127.0.0.1:8000/api/analysis/stations?pipeline_id=${runPipeline.pipeline_id}`
      : 'http://127.0.0.1:8000/api/analysis/stations';
    fetch(url).then(res => res.json()).then(setStations).catch(console.error);
  }, [runPipeline]);

  // Reload upstream experiments when script or run pipeline changes
  useEffect(() => {
    if (!selectedScript || !scripts[selectedScript]?.depends_on) {
      setUpstreamExps([]);
      setSelectedUpstreamExp(null);
      return;
    }
    const dependsOn = scripts[selectedScript].depends_on;
    const pipelineParam = runPipeline ? `?pipeline_id=${runPipeline.pipeline_id}` : '';
    fetch(`http://127.0.0.1:8000/api/analysis/upstream/${dependsOn}${pipelineParam}`)
      .then(res => res.json())
      .then((data: UpstreamExp[]) => {
        setUpstreamExps(data);
        setSelectedUpstreamExp(data.length > 0 ? data[0].experiment_id : null);
      })
      .catch(console.error);
  }, [selectedScript, runPipeline, scripts]);

  const handleSelectScript = (key: string, loadedScripts = scripts) => {
    setSelectedScript(key);
    const defs = loadedScripts[key]?.params || [];
    const initialParams: any = {};
    defs.forEach((p: any) => {
      if (p.type === 'station_list') initialParams[p.name] = p.default ?? [];
      else if (p.type === 'multi_choice') initialParams[p.name] = Array.isArray(p.default) ? [...p.default] : (p.default ? [p.default] : []);
      else initialParams[p.name] = p.default ?? (p.type === 'flag' ? false : '');
    });
    setScriptParams(initialParams);
    setRunId(null);
  };

  const handleRun = async () => {
    if (!selectedScript) return;
    const script = scripts[selectedScript];
    const needsUpstream = !!script.depends_on;
    if (needsUpstream && !selectedUpstreamExp) {
      alert(`This script requires a prior "${script.depends_on}" result.`);
      return;
    }
    try {
      const res = await fetch(`http://127.0.0.1:8000/api/analysis/${selectedScript}/run`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          params: scriptParams,
          pipeline_id: runPipeline?.pipeline_id ?? null,
          exp_id: needsUpstream ? selectedUpstreamExp : null,
        }),
      });
      const data = await res.json();
      if (data.run_id) setRunId(data.run_id);
      if (data.experiment_id) setLastExperimentId(data.experiment_id);
    } catch (err) {
      console.error(err);
    }
  };

  const updateParam = (name: string, value: any) => {
    setScriptParams((prev: any) => ({ ...prev, [name]: value }));
  };

  const toggleStation = (code: string) => {
    setScriptParams((prev: any) => {
      const list: string[] = prev.stations || [];
      return {
        ...prev,
        stations: list.includes(code) ? list.filter((c: string) => c !== code) : [...list, code],
      };
    });
  };

  const formatPipelineLabel = (p: PipelineMeta) => {
    const step1 = (p as any).step1;
    const step3 = (p as any).step3;
    const range = step1 ? `${step1.start_date} → ${step1.end_date}` : '';
    const stations = step3?.n_stations ? `${step3.n_stations} stations` : '';
    const strata = step1?.n_per_stratum ? `${step1.n_per_stratum}/stratum` : '';
    return `${p.pipeline_id} · ${[range, stations, strata].filter(Boolean).join(' · ')}`;
  };

  const pageTitle = {
    profiles: 'Profile Analysis',
    intensity: 'Intensity Analysis',
    'models/hawkes': 'Hawkes Process',
    'models/lgcp': 'LGCP Pipeline',
  }[category];

  return (
    <div className="animate-fade-in" style={{ paddingBottom: '4rem' }}>
      <h1 className="page-title">{pageTitle}</h1>
      <p className="page-subtitle" style={{ marginBottom: '1.5rem' }}>View results or configure parameters and execute analysis scripts dynamically.</p>

      {/* Tabs */}
      <div style={{ display: 'flex', gap: '0.5rem', borderBottom: '1px solid var(--border-light)', overflowX: 'auto', paddingBottom: '0.5rem', marginBottom: '1.5rem' }}>
        {Object.entries(scripts).map(([key, script]: [string, any]) => (
          <button
            key={key}
            onClick={() => handleSelectScript(key)}
            style={{
              padding: '0.5rem 1rem',
              background: selectedScript === key ? 'var(--bg-secondary)' : 'transparent',
              color: selectedScript === key ? 'var(--accent-primary)' : 'var(--text-secondary)',
              border: `1px solid ${selectedScript === key ? 'var(--border-strong)' : 'transparent'}`,
              borderRadius: 'var(--radius-md)',
              fontWeight: selectedScript === key ? 600 : 400,
              whiteSpace: 'nowrap',
              transition: 'all 0.2s ease'
            }}
          >
            {script.name}
          </button>
        ))}
      </div>

      {/* Script Header (Description & Run Button) */}
      {selectedScript && scripts[selectedScript] && (
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', gap: '2rem', marginBottom: '2rem' }}>
          <div>
            <h2 style={{ fontSize: '1.5rem', color: 'var(--text-primary)', marginBottom: '0.5rem' }}>
              {scripts[selectedScript].name}
            </h2>
            <p style={{ color: 'var(--text-secondary)', fontSize: '0.9rem', maxWidth: '800px' }}>
              {scripts[selectedScript].description}
            </p>
            {scripts[selectedScript].depends_on && (
              <span style={{ display: 'inline-block', marginTop: '0.5rem', fontSize: '0.75rem', padding: '0.1rem 0.5rem', borderRadius: '999px', background: 'rgba(245,158,11,0.15)', color: 'var(--accent-warning)', border: '1px solid rgba(245,158,11,0.3)' }}>
                ↳ requires {scripts[selectedScript].depends_on}
              </span>
            )}
          </div>
          <button 
            className="btn btn-primary"
            onClick={() => setIsRunModalOpen(true)}
            style={{ padding: '0.75rem 1.5rem', fontSize: '1rem', flexShrink: 0 }}
          >
            ▶ Run Experiment
          </button>
        </div>
      )}

      {/* Results View */}
      {selectedScript && scripts[selectedScript]?.output_dir && (
        <ResultsViewer
          outputDir={scripts[selectedScript].output_dir}
          experimentId={lastExperimentId}
        />
      )}

      {/* Run Modal */}
      {isRunModalOpen && selectedScript && scripts[selectedScript] && (
        <div style={{ position: 'fixed', inset: 0, zIndex: 100, display: 'flex', alignItems: 'center', justifyContent: 'center', background: 'rgba(0,0,0,0.6)', backdropFilter: 'blur(4px)' }}>
          <div style={{ background: 'var(--bg-primary)', border: '1px solid var(--border-strong)', borderRadius: 'var(--radius-lg)', width: '90vw', height: '85vh', display: 'flex', flexDirection: 'column', overflow: 'hidden', boxShadow: '0 25px 50px -12px rgba(0,0,0,0.5)' }}>
            
            {/* Modal Header */}
            <div style={{ padding: '1rem 1.5rem', borderBottom: '1px solid var(--border-light)', display: 'flex', justifyContent: 'space-between', alignItems: 'center', background: 'var(--bg-secondary)' }}>
              <h2 style={{ margin: 0, fontSize: '1.25rem' }}>Configure & Run: {scripts[selectedScript].name}</h2>
              <button onClick={() => setIsRunModalOpen(false)} style={{ background: 'transparent', border: 'none', color: 'var(--text-tertiary)', fontSize: '1.5rem', cursor: 'pointer', lineHeight: 1 }}>&times;</button>
            </div>

            {/* Modal Body */}
            <div style={{ display: 'flex', flex: 1, overflow: 'hidden' }}>
              
              {/* Left Column: Config */}
              <div style={{ flex: 1, padding: '1.5rem', overflowY: 'auto', borderRight: '1px solid var(--border-light)' }}>
                
                {/* Pipeline Selector */}
                <div className="form-group" style={{ marginBottom: '2rem' }}>
                  <label className="form-label">Target Pipeline</label>
                  <p style={{ fontSize: '0.8rem', color: 'var(--text-tertiary)', marginBottom: '0.5rem' }}>Select the pipeline data context to run this script on.</p>
                  <select 
                    className="form-select" 
                    value={runPipeline?.pipeline_id || ''} 
                    onChange={e => {
                      const p = pipelines.find(x => x.pipeline_id === e.target.value);
                      if (p) setRunPipeline(p);
                    }}
                    style={{ fontFamily: 'monospace' }}
                  >
                    {pipelines.map(p => (
                      <option key={p.pipeline_id} value={p.pipeline_id}>{formatPipelineLabel(p)}</option>
                    ))}
                  </select>
                </div>

                {/* Upstream Experiment Picker */}
                {scripts[selectedScript].depends_on && (
                  <div style={{ marginBottom: '2rem', padding: '1rem', background: 'rgba(245,158,11,0.06)', border: '1px solid rgba(245,158,11,0.25)', borderRadius: 'var(--radius-md)' }}>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '0.75rem' }}>
                      <span style={{ fontSize: '0.9rem', fontWeight: 600, color: 'var(--accent-warning)' }}>⬆ Upstream Experiment</span>
                      <span style={{ fontSize: '0.75rem', color: 'var(--text-tertiary)' }}>from <code>{scripts[selectedScript].depends_on}</code></span>
                    </div>

                    {upstreamExps.length === 0 ? (
                      <div style={{ fontSize: '0.85rem', color: 'var(--accent-error)' }}>
                        ⚠️ No results found in <code>{scripts[selectedScript].depends_on}</code> for this pipeline.
                        Run the root step first.
                      </div>
                    ) : (
                      <div style={{ display: 'flex', flexDirection: 'column', gap: '0.4rem' }}>
                        {upstreamExps.map(exp => (
                          <label
                            key={exp.experiment_id}
                            style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', padding: '0.5rem 0.75rem', borderRadius: 'var(--radius-sm)', background: selectedUpstreamExp === exp.experiment_id ? 'rgba(245,158,11,0.12)' : 'var(--bg-primary)', border: `1px solid ${selectedUpstreamExp === exp.experiment_id ? 'var(--accent-warning)' : 'var(--border-light)'}`, cursor: 'pointer' }}
                          >
                            <input
                              type="radio"
                              name="upstream_exp"
                              value={exp.experiment_id}
                              checked={selectedUpstreamExp === exp.experiment_id}
                              onChange={() => setSelectedUpstreamExp(exp.experiment_id)}
                              style={{ accentColor: 'var(--accent-warning)' }}
                            />
                            <div>
                              <div style={{ fontFamily: 'monospace', fontSize: '0.82rem', color: 'var(--text-primary)' }}>
                                {exp.experiment_id}
                                {exp.exit_code === 0 && <span style={{ marginLeft: '0.5rem', color: 'var(--accent-success)' }}>✓</span>}
                                {exp.exit_code !== undefined && exp.exit_code !== 0 && <span style={{ marginLeft: '0.5rem', color: 'var(--accent-error)' }}>✗</span>}
                              </div>
                              {exp.created_at && (
                                <div style={{ fontSize: '0.72rem', color: 'var(--text-tertiary)' }}>
                                  {new Date(exp.created_at).toLocaleString()}
                                </div>
                              )}
                            </div>
                          </label>
                        ))}
                      </div>
                    )}
                  </div>
                )}

                {/* Script Parameters */}
                <h3 style={{ fontSize: '1.1rem', marginBottom: '1rem', paddingBottom: '0.5rem', borderBottom: '1px solid var(--border-light)' }}>Parameters</h3>
                <div style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>
                  {scripts[selectedScript].params.map((param: any) => (
                    <div key={param.name} className="form-group" style={{ marginBottom: 0 }}>
                      <label className="form-label">{param.name.replace(/_/g, ' ').replace(/\b\w/g, (l: string) => l.toUpperCase())}</label>
                      {param.description && <p style={{ fontSize: '0.75rem', color: 'var(--text-tertiary)', marginBottom: '0.5rem' }}>{param.description}</p>}

                      {param.type === 'station_list' && (
                        <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.5rem', maxHeight: '150px', overflowY: 'auto', padding: '0.5rem', background: 'var(--bg-secondary)', borderRadius: 'var(--radius-sm)', border: '1px solid var(--border-strong)' }}>
                          <label style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', fontSize: '0.85rem' }}>
                            <input type="checkbox" checked={(scriptParams.stations?.length ?? 0) === 0} onChange={() => updateParam('stations', [])} />
                            <span style={{ color: 'var(--accent-warning)' }}>All Stations</span>
                          </label>
                          {stations.map((s: any) => (
                            <label key={s.station_code} style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', fontSize: '0.85rem', background: 'var(--bg-primary)', padding: '0.2rem 0.5rem', borderRadius: '4px' }}>
                              <input type="checkbox" checked={scriptParams.stations?.includes(s.station_code)} onChange={() => toggleStation(s.station_code)} />
                              {s.station_code} – {s.station_name?.split(' ')[0]}
                            </label>
                          ))}
                        </div>
                      )}
                      {param.type === 'choice' && (
                        <select className="form-select" value={scriptParams[param.name] || ''} onChange={e => updateParam(param.name, e.target.value)}>
                          {param.choices.map((c: string) => <option key={c} value={c}>{c}</option>)}
                        </select>
                      )}
                      {param.type === 'multi_choice' && (
                        <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.5rem' }}>
                          {param.choices.map((c: string) => {
                            const selected = (scriptParams[param.name] || []).includes(c);
                            return (
                              <button
                                key={c}
                                type="button"
                                onClick={() => {
                                  const current: string[] = scriptParams[param.name] || [];
                                  updateParam(param.name, selected ? current.filter((x: string) => x !== c) : [...current, c]);
                                }}
                                style={{
                                  padding: '0.35rem 0.75rem',
                                  borderRadius: '999px',
                                  border: selected ? '1px solid var(--accent-primary)' : '1px solid var(--border-light)',
                                  background: selected ? 'rgba(59,130,246,0.15)' : 'var(--bg-primary)',
                                  color: selected ? 'var(--accent-primary)' : 'var(--text-tertiary)',
                                  fontSize: '0.82rem',
                                  fontWeight: selected ? 600 : 400,
                                  cursor: 'pointer',
                                  transition: 'all 0.15s ease',
                                }}
                              >
                                {selected ? '✓ ' : ''}{c}
                              </button>
                            );
                          })}
                        </div>
                      )}
                      {param.type === 'flag' && (
                        <label style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                          <input type="checkbox" checked={scriptParams[param.name] || false} onChange={e => updateParam(param.name, e.target.checked)} />
                          <span style={{ fontSize: '0.875rem' }}>Enable</span>
                        </label>
                      )}
                      {(param.type === 'int' || param.type === 'float' || param.type === 'date_percentage') && (
                        <input type="number" step={param.type === 'int' ? '1' : '0.1'} className="form-input" value={scriptParams[param.name] ?? ''} onChange={e => updateParam(param.name, param.type === 'int' ? parseInt(e.target.value) : parseFloat(e.target.value))} />
                      )}
                      {param.type === 'str' && (
                        <input type="text" className="form-input" value={scriptParams[param.name] || ''} onChange={e => updateParam(param.name, e.target.value)} />
                      )}
                    </div>
                  ))}
                </div>
              </div>

              {/* Right Column: Terminal */}
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
                  <button
                    className="btn btn-primary"
                    style={{ width: '100%', padding: '0.875rem', fontSize: '1.05rem' }}
                    onClick={handleRun}
                    disabled={status === 'running' || (!!scripts[selectedScript].depends_on && !selectedUpstreamExp)}
                  >
                    {status === 'running' ? 'Running Script...' : 'Run Script'}
                  </button>
                  {scripts[selectedScript].depends_on && !selectedUpstreamExp && (
                    <p style={{ marginTop: '0.5rem', fontSize: '0.75rem', color: 'var(--accent-error)', textAlign: 'center' }}>
                      Select an upstream experiment to enable this button.
                    </p>
                  )}
                </div>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
