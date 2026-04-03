'use client';

import { useState, useEffect } from 'react';
import { useWebSocket } from '@/hooks/useWebSocket';
import ResultsViewer from '@/components/ResultsViewer';

export default function AnalysisPage({ category }: { category: 'profiles' | 'intensity' | 'models/hawkes' | 'models/lgcp' }) {
  const [scripts, setScripts] = useState<any>({});
  const [stations, setStations] = useState<any[]>([]);
  const [selectedScript, setSelectedScript] = useState<string | null>(null);
  const [scriptParams, setScriptParams] = useState<any>({});
  
  const [runId, setRunId] = useState<string | null>(null);
  const { logs, status } = useWebSocket(runId);

  useEffect(() => {
    // Fetch script definitions
    fetch('http://127.0.0.1:8000/api/analysis')
      .then(res => res.json())
      .then(data => {
        // Filter by category
        const filtered = Object.entries(data).reduce((acc: any, [key, val]: [string, any]) => {
          if (val.category === category) acc[key] = val;
          return acc;
        }, {});
        setScripts(filtered);
      })
      .catch(console.error);

    // Fetch stations
    fetch('http://127.0.0.1:8000/api/analysis/stations')
      .then(res => res.json())
      .then(data => setStations(data))
      .catch(console.error);
  }, [category]);

  const handleSelectScript = (key: string) => {
    setSelectedScript(key);
    // Initialize default params
    const defs = scripts[key].params;
    const initialParams: any = {};
    defs.forEach((p: any) => {
      initialParams[p.name] = p.default ?? (p.type === 'station_list' ? [] : '');
    });
    setScriptParams(initialParams);
    setRunId(null);
  };

  const handleRun = async () => {
    if (!selectedScript) return;
    try {
      const res = await fetch(`http://127.0.0.1:8000/api/analysis/${selectedScript}/run`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ params: scriptParams })
      });
      const data = await res.json();
      if (data.run_id) {
        setRunId(data.run_id);
      }
    } catch (err) {
      console.error(err);
    }
  };

  const updateParam = (name: string, value: any) => {
    setScriptParams((prev: any) => ({ ...prev, [name]: value }));
  };

  const toggleStation = (code: string) => {
    setScriptParams((prev: any) => {
      const list = prev.stations || [];
      return {
        ...prev,
        stations: list.includes(code) ? list.filter((c: string) => c !== code) : [...list, code]
      };
    });
  };

  return (
    <div className="animate-fade-in">
      <h1 className="page-title">{category === 'profiles' ? 'Profile Analysis' : 'Intensity Analysis'}</h1>
      <p className="page-subtitle">Configure parameters and execute analysis scripts dynamically.</p>

      <div style={{ display: 'grid', gridTemplateColumns: 'minmax(300px, 1fr) minmax(400px, 2fr)', gap: '2rem' }}>
        
        {/* Script Selection List */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
          {Object.entries(scripts).map(([key, script]: [string, any]) => (
            <div 
              key={key} 
              className="card" 
              style={{ cursor: 'pointer', borderColor: selectedScript === key ? 'var(--accent-primary)' : 'var(--border-light)' }}
              onClick={() => handleSelectScript(key)}
            >
              <h3 style={{ fontSize: '1rem', color: selectedScript === key ? 'var(--accent-primary)' : 'inherit' }}>{script.name}</h3>
              <p style={{ fontSize: '0.8rem', color: 'var(--text-secondary)', marginTop: '0.5rem' }}>{script.description}</p>
            </div>
          ))}
        </div>

        {/* Configuration and Execution */}
        {selectedScript && scripts[selectedScript] && (
          <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
            <div className="card glass">
              <h2 style={{ marginBottom: '1.5rem', fontSize: '1.25rem' }}>Configure {scripts[selectedScript].name}</h2>
              
              <div style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>
                {scripts[selectedScript].params.map((param: any) => (
                  <div key={param.name} className="form-group" style={{ marginBottom: 0 }}>
                    <label className="form-label">{param.name.replace(/_/g, ' ').replace(/\b\w/g, (l:string) => l.toUpperCase())}</label>
                    <p style={{ fontSize: '0.75rem', color: 'var(--text-tertiary)', marginBottom: '0.5rem' }}>{param.description}</p>
                    
                    {param.type === 'station_list' && (
                      <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.5rem', maxHeight: '150px', overflowY: 'auto', padding: '0.5rem', background: 'var(--bg-primary)', borderRadius: 'var(--radius-sm)', border: '1px solid var(--border-strong)' }}>
                        <label style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', fontSize: '0.85rem' }}>
                          <input type="checkbox" checked={scriptParams.stations?.length === 0} onChange={() => updateParam('stations', [])} />
                          <span style={{ color: 'var(--accent-warning)' }}>All Stations</span>
                        </label>
                        {stations.map(s => (
                          <label key={s.station_code} style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', fontSize: '0.85rem', background: 'var(--bg-secondary)', padding: '0.2rem 0.5rem', borderRadius: '4px' }}>
                            <input 
                              type="checkbox" 
                              checked={scriptParams.stations?.includes(s.station_code)} 
                              onChange={() => toggleStation(s.station_code)} 
                            />
                            {s.station_code} - {s.station_name?.split(' ')[0]}
                          </label>
                        ))}
                      </div>
                    )}
                    
                    {param.type === 'choice' && (
                      <select className="form-select" value={scriptParams[param.name] || ''} onChange={(e) => updateParam(param.name, e.target.value)}>
                        {param.choices.map((c: string) => <option key={c} value={c}>{c}</option>)}
                      </select>
                    )}
                    
                    {param.type === 'flag' && (
                      <label style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                        <input type="checkbox" checked={scriptParams[param.name] || false} onChange={(e) => updateParam(param.name, e.target.checked)} />
                        <span style={{ fontSize: '0.875rem' }}>Enable</span>
                      </label>
                    )}
                    
                    {(param.type === 'int' || param.type === 'float' || param.type === 'date_percentage') && (
                      <input type="number" 
                        step={param.type === 'float' || param.type === 'date_percentage' ? '0.1' : '1'} 
                        className="form-input" 
                        value={scriptParams[param.name] ?? ''} 
                        onChange={(e) => updateParam(param.name, param.type === 'int' ? parseInt(e.target.value) : parseFloat(e.target.value))} 
                      />
                    )}

                    {param.type === 'str' && (
                      <input type="text" className="form-input" value={scriptParams[param.name] || ''} onChange={(e) => updateParam(param.name, e.target.value)} />
                    )}
                  </div>
                ))}
              </div>

              <div style={{ marginTop: '2rem', borderTop: '1px solid var(--border-light)', paddingTop: '1.5rem' }}>
                <button 
                  className="btn btn-primary" 
                  style={{ width: '100%', padding: '0.75rem' }} 
                  onClick={handleRun}
                  disabled={status === 'running'}
                >
                  {status === 'running' ? 'Running Script...' : 'Run Script'}
                </button>
              </div>
            </div>

            {/* Terminal Log */}
            <div className="card" style={{ backgroundColor: '#000', padding: '1rem', height: '300px', display: 'flex', flexDirection: 'column' }}>
              <div style={{ marginBottom: '0.5rem', fontSize: '0.75rem', color: '#71717a', display: 'flex', justifyContent: 'space-between' }}>
                <span>Execution Log</span>
                <span>{status.toUpperCase()}</span>
              </div>
              <div style={{ 
                flex: 1, 
                fontFamily: 'monospace',
                fontSize: '0.8rem',
                overflowY: 'auto',
                color: '#a1a1aa'
              }}>
                {logs.length === 0 && <div style={{ color: '#52525b', fontStyle: 'italic' }}>Output captures here...</div>}
                {logs.map((log, i) => (
                  <div key={i} style={{ 
                    color: log.type === 'stderr' ? 'var(--accent-error)' : 
                           log.type === 'status' ? 'var(--accent-success)' : 'inherit',
                    whiteSpace: 'pre-wrap',
                    wordBreak: 'break-all'
                  }}>
                    {log.line || (log.type === 'status' ? `[Exited: ${log.exit_code}]` : '')}
                  </div>
                ))}
              </div>
            </div>

            {/* Results Viewer injected below terminal */}
            {scripts[selectedScript].output_dir && (
              <ResultsViewer outputDir={scripts[selectedScript].output_dir} />
            )}

          </div>
        )}

      </div>
    </div>
  );
}
