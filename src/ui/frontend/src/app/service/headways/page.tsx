'use client';

import { useState, useEffect } from 'react';
import { useWebSocket } from '@/hooks/useWebSocket';

interface PipelineMeta {
  pipeline_id: string;
  created_at?: string;
  params?: any;
}

interface ExperimentMeta {
  experiment_id: string;
  pipeline_id: string;
  script?: string;
  created_at?: string;
  exit_code?: number;
  params?: any;
}

interface FitReport {
  metadata: {
    route_name: string;
    period: string;
    cv: number;
    scheduled_mean: number;
    simulated_mean: number;
    simulated_std: number;
  };
  fits: Record<
    string,
    {
      params: Record<string, number>;
      log_likelihood: number;
      aic: number;
      bic: number;
    }
  >;
  available_routes?: string[];
  selected_route?: string;
}

export default function HeadwaysPage() {
  // Pipelines Snapshot
  const [pipelines, setPipelines] = useState<PipelineMeta[]>([]);
  const [runPipeline, setRunPipeline] = useState<PipelineMeta | null>(null);

  // Historical Results State
  const [historyExperiments, setHistoryExperiments] = useState<ExperimentMeta[]>([]);
  const [selectedPipelineId, setSelectedPipelineId] = useState<string>('default');
  const [selectedExpId, setSelectedExpId] = useState<string>('default');
  const [refreshCounter, setRefreshCounter] = useState(0);

  // Modal Configuration State
  const [isRunModalOpen, setIsRunModalOpen] = useState(false);
  const [scriptParams, setScriptParams] = useState({
    route_name: ['B12'],
    period: 'peak',
    cv: 0.25,
    n_samples: 1000,
  });

  const [runId, setRunId] = useState<string | null>(null);
  const [lastExperimentId, setLastExperimentId] = useState<string | undefined>(undefined);
  const { logs, status } = useWebSocket(runId);

  // Active Headway Data State
  const [fitData, setFitData] = useState<FitReport | null>(null);
  const [availableRoutesInResult, setAvailableRoutesInResult] = useState<string[]>([]);
  const [selectedRouteInResult, setSelectedRouteInResult] = useState<string>('');
  const [isLoading, setIsLoading] = useState(false);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);

  // Fetch pipelines on mount
  useEffect(() => {
    fetch('http://127.0.0.1:8000/api/pipeline/experiments')
      .then((res) => res.json())
      .then((data: PipelineMeta[]) => {
        setPipelines(data);
        if (data.length > 0) setRunPipeline(data[0]);
      })
      .catch(console.error);
  }, []);

  // Fetch all experiments on load or when refresh counter changes
  useEffect(() => {
    fetch('http://127.0.0.1:8000/api/results/headway_fitting')
      .then((res) => {
        if (!res.ok) return [];
        return res.json();
      })
      .then((data: ExperimentMeta[]) => {
        setHistoryExperiments(data);
      })
      .catch(console.error);
  }, [refreshCounter]);

  // Fetch Fit Data function
  const fetchFitData = (pId: string = 'default', eId: string = 'default', route?: string) => {
    setIsLoading(true);
    setErrorMsg(null);
    const routeParam = route ? `?route=${route}` : '';
    fetch(`http://127.0.0.1:8000/api/results/service/headway_fitting/${pId}/${eId}/fit${routeParam}`)
      .then((res) => {
        if (!res.ok) {
          throw new Error('MLE headway fitting results not found for this execution.');
        }
        return res.json();
      })
      .then((data: FitReport) => {
        setFitData(data);
        setAvailableRoutesInResult(data.available_routes || []);
        setSelectedRouteInResult(data.selected_route || data.metadata.route_name);
        setIsLoading(false);
      })
      .catch((err) => {
        console.error(err);
        setErrorMsg(err.message || 'Failed to load fitting details.');
        setIsLoading(false);
      });
  };

  // Load default data on mount
  useEffect(() => {
    fetchFitData('default', 'default');
  }, []);

  // When a run completes, automatically reload results and refresh history
  useEffect(() => {
    if (status === 'completed' && lastExperimentId && runPipeline) {
      setSelectedPipelineId(runPipeline.pipeline_id);
      setSelectedExpId(lastExperimentId);
      fetchFitData(runPipeline.pipeline_id, lastExperimentId);
      setRefreshCounter((c) => c + 1);
    }
  }, [status, lastExperimentId]);

  const handleRun = async () => {
    try {
      const res = await fetch('http://127.0.0.1:8000/api/analysis/service/headway_fitting/run', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          params: scriptParams,
          pipeline_id: runPipeline?.pipeline_id ?? null,
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
    setScriptParams((prev) => ({ ...prev, [name]: value }));
  };

  const getCvLabel = (cv: number) => {
    if (cv <= 0.12) return '🟢 Low Noise (Dedicated Lanes / Reliable)';
    if (cv <= 0.35) return '🟡 Moderate Traffic (Typical delays)';
    return '🔴 High Noise (Congestion / Bus Bunching)';
  };

  // Determine winner distribution based on minimum AIC/BIC
  const getWinner = () => {
    if (!fitData?.fits) return '';
    const keys = Object.keys(fitData.fits);
    return keys.reduce((winner, current) => 
      fitData.fits[current].aic < fitData.fits[winner].aic ? current : winner
    , keys[0]);
  };

  const winner = getWinner();

  const formatParamVal = (val: number) => {
    return val.toFixed(3);
  };

  const formatPipelineLabel = (p: PipelineMeta) => {
    const step1 = (p as any).step1;
    const step3 = (p as any).step3;
    const range = step1 ? `${step1.start_date} → ${step1.end_date}` : '';
    const nStations = step3?.n_stations ? `${step3.n_stations} stations` : '';
    return `${p.pipeline_id} · ${[range, nStations].filter(Boolean).join(' · ')}`;
  };

  // Popular troncal route names from TransMilenio frequencies sheet
  const availableRoutes = [
    "B12", "G12", "F23", "J23", "H75", "B75", "D20", "H20",
    "B13", "H13", "F28", "B28", "K10", "L10", "D21", "H21",
    "M47", "G47", "C30", "G30"
  ];

  // Derived high-res image link
  const staticPlotUrl = selectedPipelineId === 'default'
    ? 'http://127.0.0.1:8000/api/results/headway_fitting/default/default/fitted_headways_comparison.png/view'
    : `http://127.0.0.1:8000/api/results/headway_fitting/${selectedPipelineId}/${selectedExpId}/fitted_headways_comparison_${selectedRouteInResult || 'B12'}.png/view`;

  return (
    <div className="animate-fade-in" style={{ paddingBottom: '4rem' }}>
      {/* Page Header */}
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', gap: '2rem', marginBottom: '1.5rem', borderBottom: '2px solid var(--accent-primary)', paddingBottom: '0.5rem' }}>
        <div>
          <h1 style={{ fontSize: '1.75rem', fontWeight: 600, color: 'var(--accent-primary)', margin: 0 }}>Bus Arrival Headway Fitting</h1>
          <p style={{ color: 'var(--text-secondary)', fontSize: '0.9rem', marginTop: '0.25rem' }}>
            Model transit service rates stochastically by fitting continuous distributions (Gamma, Erlang, Log-Normal) via Maximum Likelihood Estimation.
          </p>
        </div>
        <button 
          className="btn btn-primary"
          onClick={() => setIsRunModalOpen(true)}
          style={{ padding: '0.65rem 1.25rem', fontSize: '0.9rem', flexShrink: 0, fontWeight: 600 }}
        >
          ▶ Fit Bus Service Rates
        </button>
      </div>

      {/* Historical Experiments Selectors */}
      <div className="card glass" style={{ display: 'flex', flexWrap: 'wrap', gap: '1rem', background: 'var(--bg-secondary)', padding: '1rem', borderRadius: 'var(--radius-md)', border: '1px solid var(--border-light)', marginBottom: '1.5rem', alignItems: 'center' }}>
        <div style={{ fontSize: '0.85rem', fontWeight: 700, color: 'var(--accent-primary)', textTransform: 'uppercase', marginRight: '0.5rem' }}>
          📂 Load Fit Results
        </div>
        
        {/* Pipeline Selector */}
        <div className="form-group" style={{ margin: 0, flex: '1 1 200px' }}>
          <select 
            className="form-select" 
            value={selectedPipelineId} 
            onChange={(e) => {
              const pId = e.target.value;
              setSelectedPipelineId(pId);
              if (pId === 'default') {
                setSelectedExpId('default');
                fetchFitData('default', 'default');
              } else {
                const related = historyExperiments.filter((ex) => ex.pipeline_id === pId);
                if (related.length > 0) {
                  setSelectedExpId(related[0].experiment_id);
                  fetchFitData(pId, related[0].experiment_id);
                }
              }
            }}
            style={{ fontFamily: 'monospace', fontSize: '0.82rem', padding: '0.4rem 0.75rem' }}
          >
            <option value="default">Default baseline context</option>
            {Array.from(new Set(historyExperiments.map((ex) => ex.pipeline_id).filter(Boolean))).map((pId) => (
              <option key={pId} value={pId}>{pId}</option>
            ))}
          </select>
        </div>

        {/* Experiment Selector */}
        <div className="form-group" style={{ margin: 0, flex: '1 1 200px' }}>
          <select 
            className="form-select" 
            value={selectedExpId} 
            onChange={(e) => {
              const eId = e.target.value;
              setSelectedExpId(eId);
              fetchFitData(selectedPipelineId, eId);
            }}
            style={{ fontFamily: 'monospace', fontSize: '0.82rem', padding: '0.4rem 0.75rem' }}
          >
            {selectedPipelineId === 'default' ? (
              <option value="default">Baseline B12 Model Fit</option>
            ) : (
              historyExperiments
                .filter((ex) => ex.pipeline_id === selectedPipelineId)
                .map((ex) => (
                  <option key={ex.experiment_id} value={ex.experiment_id}>
                    {ex.experiment_id} {ex.exit_code === 0 ? '✓' : '✗'}
                  </option>
                ))
            )}
          </select>
        </div>

        {/* Dynamic Route Sub-Selector */}
        {availableRoutesInResult.length > 1 && (
          <div className="form-group" style={{ margin: 0, flex: '1 1 150px' }}>
            <select
              className="form-select"
              value={selectedRouteInResult}
              onChange={(e) => {
                const r = e.target.value;
                setSelectedRouteInResult(r);
                fetchFitData(selectedPipelineId, selectedExpId, r);
              }}
              style={{ fontFamily: 'monospace', fontSize: '0.82rem', padding: '0.4rem 0.75rem', borderColor: 'var(--accent-success)', borderWidth: '1.5px' }}
            >
              {availableRoutesInResult.map((r) => (
                <option key={r} value={r}>Route {r} fit curves</option>
              ))}
            </select>
          </div>
        )}
      </div>

      {/* Main Results Board */}
      <div className="card glass" style={{ display: 'flex', flexDirection: 'column', gap: '2rem', border: '1px solid var(--border-strong)' }}>
        
        {/* Loading Overlay */}
        {isLoading && (
          <div style={{ padding: '5rem', textAlign: 'center', color: 'var(--text-tertiary)' }}>
            <span style={{ fontSize: '1.5rem' }}>🔄 Solving MLE distribution fits...</span>
          </div>
        )}

        {/* Error Overlay */}
        {errorMsg && !isLoading && (
          <div style={{ padding: '3rem', textAlign: 'center', background: 'rgba(239, 68, 68, 0.05)', border: '1px dashed var(--accent-error)', borderRadius: 'var(--radius-md)' }}>
            <h3 style={{ color: 'var(--accent-error)', fontSize: '1.1rem', fontWeight: 600, marginBottom: '0.5rem' }}>Statistical Results Not Found</h3>
            <p style={{ color: 'var(--text-secondary)', fontSize: '0.85rem', marginBottom: '1rem' }}>{errorMsg}</p>
            <button className="btn btn-secondary" onClick={() => { setSelectedPipelineId('default'); setSelectedExpId('default'); fetchFitData('default', 'default'); }}>
              Load Baseline scheduled Route B12 Parameters
            </button>
          </div>
        )}

        {/* Successful Fit Dashboard */}
        {fitData && !isLoading && !errorMsg && (
          <div style={{ display: 'flex', flexDirection: 'column', gap: '2rem' }}>
            
            {/* Metadata Summary Banner */}
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(180px, 1fr))', gap: '1rem', background: 'var(--bg-secondary)', padding: '1.25rem', borderRadius: 'var(--radius-sm)', border: '1px solid var(--border-light)' }}>
              <div>
                <span style={{ fontSize: '0.72rem', color: 'var(--text-tertiary)', fontWeight: 600, display: 'block', textTransform: 'uppercase' }}>Selected Route</span>
                <span style={{ fontSize: '1.35rem', fontWeight: 700, color: 'var(--accent-primary)' }}>Route {fitData.metadata.route_name}</span>
              </div>
              <div>
                <span style={{ fontSize: '0.72rem', color: 'var(--text-tertiary)', fontWeight: 600, display: 'block', textTransform: 'uppercase' }}>Period</span>
                <span style={{ fontSize: '1.15rem', fontWeight: 600, color: 'var(--text-primary)', textTransform: 'capitalize' }}>{fitData.metadata.period} Hours</span>
              </div>
              <div>
                <span style={{ fontSize: '0.72rem', color: 'var(--text-tertiary)', fontWeight: 600, display: 'block', textTransform: 'uppercase' }}>scheduled Mean</span>
                <span style={{ fontSize: '1.15rem', fontWeight: 600, color: 'var(--text-primary)' }}>{fitData.metadata.scheduled_mean.toFixed(1)} minutes</span>
              </div>
              <div>
                <span style={{ fontSize: '0.72rem', color: 'var(--text-tertiary)', fontWeight: 600, display: 'block', textTransform: 'uppercase' }}>Simulated Mean / Std</span>
                <span style={{ fontSize: '1.15rem', fontWeight: 600, color: 'var(--text-primary)' }}>
                  {fitData.metadata.simulated_mean.toFixed(2)}m (±{fitData.metadata.simulated_std.toFixed(2)}m)
                </span>
              </div>
              <div>
                <span style={{ fontSize: '0.72rem', color: 'var(--text-tertiary)', fontWeight: 600, display: 'block', textTransform: 'uppercase' }}>Optimal Queuing Fit</span>
                <span style={{ fontSize: '1.15rem', fontWeight: 700, color: 'var(--accent-success)' }}>
                  {winner === 'lognormal' ? 'Log-Normal 🏆' : winner.toUpperCase() + ' 🏆'}
                </span>
              </div>
            </div>

            {/* Layout: Chart + Stats Side Panel */}
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 340px', gap: '2rem', alignItems: 'start' }}>
              
              {/* Left Column: High-Resolution Matplotlib Comparison Overlay */}
              <div style={{ border: '1px solid var(--border-strong)', borderRadius: 'var(--radius-sm)', padding: '1.5rem', background: '#ffffff', minHeight: '420px', display: 'flex', flexDirection: 'column', justifyContent: 'center', alignItems: 'center' }}>
                <h3 style={{ fontSize: '1rem', fontWeight: 600, color: 'var(--accent-primary)', marginBottom: '1.5rem', borderBottom: '1px solid var(--border-light)', paddingBottom: '0.5rem', width: '100%' }}>
                  📊 Continuous PDF fits vs Binned Headway Telemetry
                </h3>
                
                <div style={{ flex: 1, display: 'flex', justifyContent: 'center', alignItems: 'center', width: '100%', background: '#fff' }}>
                  <img
                    src={staticPlotUrl}
                    alt="Headways Static Plot Comparison"
                    style={{ maxWidth: '100%', height: 'auto', display: 'block', borderRadius: '3px', border: '1px solid var(--border-light)', padding: '0.25rem' }}
                    onError={(e) => {
                      (e.target as HTMLImageElement).src = 'http://127.0.0.1:8000/api/results/headway_fitting/default/default/fitted_headways_comparison.png/view';
                    }}
                  />
                </div>
              </div>

              {/* Right Column: Comparative Stats and formulas */}
              <div style={{ display: 'flex', flexDirection: 'column', gap: '1.25rem' }}>
                
                {/* Goodness of Fit Table Card */}
                <div className="card" style={{ padding: '1.25rem', background: 'var(--bg-secondary)', border: '1px solid var(--border-strong)', margin: 0 }}>
                  <h3 style={{ fontSize: '0.85rem', fontWeight: 700, color: 'var(--accent-primary)', borderBottom: '1px solid var(--border-light)', paddingBottom: '0.5rem', marginBottom: '0.75rem', textTransform: 'uppercase' }}>
                    📈 Information Criteria Table
                  </h3>
                  <div style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem' }}>
                    {Object.entries(fitData.fits).map(([dist, metrics]) => {
                      const isWinner = winner === dist;
                      return (
                        <div
                          key={dist}
                          style={{
                            background: 'var(--bg-primary)',
                            border: `1px solid ${isWinner ? 'var(--accent-success)' : 'var(--border-light)'}`,
                            borderRadius: '3px',
                            padding: '0.65rem 0.85rem',
                            fontSize: '0.78rem',
                            boxShadow: isWinner ? '0 1px 4px rgba(21, 87, 36, 0.08)' : 'none',
                          }}
                        >
                          <div style={{ display: 'flex', justifyContent: 'space-between', fontWeight: 700, color: 'var(--text-primary)', marginBottom: '0.3rem' }}>
                            <span style={{ textTransform: 'capitalize' }}>
                              {dist === 'lognormal' ? 'Log-Normal' : dist}
                              {isWinner && <span style={{ marginLeft: '0.4rem', color: 'var(--accent-success)', fontSize: '0.7rem', verticalAlign: 'middle' }}>🏆 WINNER</span>}
                            </span>
                          </div>
                          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '0.2rem', borderTop: '1px dashed var(--border-light)', paddingTop: '0.4rem', color: 'var(--text-secondary)', fontSize: '0.72rem' }}>
                            <div>AIC: <strong style={{ color: isWinner ? 'var(--accent-success)' : 'inherit' }}>{metrics.aic.toFixed(1)}</strong></div>
                            <div>BIC: <strong>{metrics.bic.toFixed(1)}</strong></div>
                            <div style={{ gridColumn: 'span 2' }}>Log-Likelihood: <strong style={{ fontFamily: 'monospace' }}>{metrics.log_likelihood.toFixed(2)}</strong></div>
                          </div>
                        </div>
                      );
                    })}
                  </div>
                </div>

                {/* Equation Parameters Card */}
                <div className="card" style={{ padding: '1.25rem', background: 'var(--bg-secondary)', border: '1px solid var(--border-strong)', margin: 0 }}>
                  <h3 style={{ fontSize: '0.85rem', fontWeight: 700, color: 'var(--accent-primary)', borderBottom: '1px solid var(--border-light)', paddingBottom: '0.5rem', marginBottom: '0.75rem', textTransform: 'uppercase' }}>
                    🧮 Fitted Mathematical parameters
                  </h3>
                  <div style={{ display: 'flex', flexDirection: 'column', gap: '0.6rem', fontSize: '0.75rem', fontFamily: 'monospace', color: 'var(--text-secondary)' }}>
                    
                    <div>
                      <span style={{ fontWeight: 600, color: 'var(--text-primary)', display: 'block', fontSize: '0.72rem', fontFamily: 'sans-serif', marginBottom: '0.15rem' }}>Gamma (α, β):</span>
                      shape (α) = {formatParamVal(fitData.fits.gamma.params.shape)}, scale (β) = {formatParamVal(fitData.fits.gamma.params.scale)}
                    </div>
                    
                    <div style={{ borderTop: '1px dashed var(--border-light)', paddingTop: '0.4rem' }}>
                      <span style={{ fontWeight: 600, color: 'var(--text-primary)', display: 'block', fontSize: '0.72rem', fontFamily: 'sans-serif', marginBottom: '0.15rem' }}>Erlang-k (k, θ):</span>
                      shape (k) = {fitData.fits.erlang.params.shape_k}, scale (θ) = {formatParamVal(fitData.fits.erlang.params.scale)}
                    </div>
                    
                    <div style={{ borderTop: '1px dashed var(--border-light)', paddingTop: '0.4rem' }}>
                      <span style={{ fontWeight: 600, color: 'var(--text-primary)', display: 'block', fontSize: '0.72rem', fontFamily: 'sans-serif', marginBottom: '0.15rem' }}>Log-Normal (σ, μ):</span>
                      shape (σ) = {formatParamVal(fitData.fits.lognormal.params.sigma)}, log-mean (μ) = {formatParamVal(fitData.fits.lognormal.params.mu)}
                    </div>

                  </div>
                </div>

              </div>

            </div>

          </div>
        )}
      </div>

      {/* Parameter Configuration & WebSocket Logs Modal */}
      {isRunModalOpen && (
        <div style={{ position: 'fixed', inset: 0, zIndex: 100, display: 'flex', alignItems: 'center', justifyContent: 'center', background: 'rgba(0,0,0,0.5)', backdropFilter: 'none' }}>
          <div style={{ background: 'var(--bg-primary)', border: '1px solid var(--border-strong)', borderRadius: 'var(--radius-md)', width: '90vw', height: '85vh', display: 'flex', flexDirection: 'column', overflow: 'hidden', boxShadow: 'var(--shadow-lg)' }}>
            
            {/* Modal Header */}
            <div style={{ padding: '1rem 1.5rem', borderBottom: '1px solid var(--border-strong)', display: 'flex', justifyContent: 'space-between', alignItems: 'center', background: 'var(--bg-secondary)' }}>
              <h2 style={{ margin: 0, fontSize: '1.25rem', color: 'var(--accent-primary)', fontWeight: 600 }}>Configure & Run: MLE Headway Fitting</h2>
              <button onClick={() => setIsRunModalOpen(false)} style={{ background: 'transparent', border: 'none', color: 'var(--text-tertiary)', fontSize: '1.5rem', cursor: 'pointer', lineHeight: 1 }}>&times;</button>
            </div>

            {/* Modal Body */}
            <div style={{ display: 'flex', flex: 1, overflow: 'hidden' }}>
              
              {/* Left Column: Parameter Form */}
              <div style={{ flex: 1, padding: '1.5rem', overflowY: 'auto', borderRight: '1px solid var(--border-strong)', background: 'var(--bg-primary)' }}>
                
                {/* Pipeline Selector */}
                <div className="form-group" style={{ marginBottom: '2rem' }}>
                  <label className="form-label">Data Pipeline Context</label>
                  <p style={{ fontSize: '0.8rem', color: 'var(--text-tertiary)', marginBottom: '0.5rem' }}>Select the pipeline data snapshot to bind parameters to.</p>
                  <select 
                    className="form-select" 
                    value={runPipeline?.pipeline_id || ''} 
                    onChange={(e) => {
                      const p = pipelines.find((x) => x.pipeline_id === e.target.value);
                      if (p) setRunPipeline(p);
                    }}
                    style={{ fontFamily: 'monospace' }}
                  >
                    {pipelines.map((p) => (
                      <option key={p.pipeline_id} value={p.pipeline_id}>{formatPipelineLabel(p)}</option>
                    ))}
                  </select>
                </div>

                {/* Parameters */}
                <h3 style={{ fontSize: '1.1rem', marginBottom: '1rem', paddingBottom: '0.5rem', borderBottom: '1px solid var(--border-strong)', color: 'var(--accent-primary)', fontWeight: 600 }}>Parameters</h3>
                <div style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>
                  
                  {/* Route Selection */}
                  <div className="form-group" style={{ marginBottom: 0 }}>
                    <label className="form-label">Transit Route Services</label>
                    <p style={{ fontSize: '0.75rem', color: 'var(--text-tertiary)', marginBottom: '0.5rem' }}>Choose the route services to pull frequencies and fit models for.</p>
                    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(80px, 1fr))', gap: '0.5rem', maxHeight: '180px', overflowY: 'auto', padding: '0.5rem', background: 'var(--bg-secondary)', border: '1px solid var(--border-strong)', borderRadius: 'var(--radius-sm)' }}>
                      {availableRoutes.map((r) => {
                        const isSelected = scriptParams.route_name.includes(r);
                        return (
                          <button
                            key={r}
                            type="button"
                            onClick={() => {
                              const current = scriptParams.route_name;
                              const updated = isSelected
                                ? current.filter((x: string) => x !== r)
                                : [...current, r];
                              // Keep at least one route selected
                              if (updated.length > 0) {
                                updateParam('route_name', updated);
                              }
                            }}
                            style={{
                              padding: '0.35rem 0.5rem',
                              border: isSelected ? '1px solid var(--accent-primary)' : '1px solid var(--border-strong)',
                              background: isSelected ? 'var(--bg-highlight)' : 'var(--bg-primary)',
                              color: isSelected ? 'var(--accent-primary)' : 'var(--text-secondary)',
                              fontSize: '0.75rem',
                              fontWeight: isSelected ? 600 : 400,
                              cursor: 'pointer',
                              borderRadius: '3px',
                              transition: 'all 0.1s ease',
                              textAlign: 'center',
                            }}
                          >
                            {isSelected ? '✓ ' : ''}{r}
                          </button>
                        );
                      })}
                    </div>
                  </div>

                  {/* Day Period */}
                  <div className="form-group" style={{ marginBottom: 0 }}>
                    <label className="form-label">Operating Period</label>
                    <p style={{ fontSize: '0.75rem', color: 'var(--text-tertiary)', marginBottom: '0.5rem' }}>Toggle between high-demand peak frequencies and off-peak planned intervals.</p>
                    <select
                      className="form-select"
                      value={scriptParams.period}
                      onChange={(e) => updateParam('period', e.target.value)}
                    >
                      <option value="peak">Peak Hours (Frecuencia Peak)</option>
                      <option value="offpeak">Off-Peak Hours (Frecuencia Off-Peak)</option>
                    </select>
                  </div>

                  {/* Coefficient of Variation standard deviation delay index */}
                  <div className="form-group" style={{ marginBottom: 0 }}>
                    <label className="form-label">Traffic Noise Index ($C_v$)</label>
                    <p style={{ fontSize: '0.75rem', color: 'var(--text-tertiary)', marginBottom: '0.4rem' }}>
                      Coefficient of variation ($C_v = \sigma / \mu$) representing delay variance (driver wobble + exponential traffic spikes).
                    </p>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '1rem' }}>
                      <input
                        type="range"
                        min="0.05"
                        max="0.80"
                        step="0.05"
                        value={scriptParams.cv}
                        onChange={(e) => updateParam('cv', parseFloat(e.target.value))}
                        style={{ flex: 1, accentColor: 'var(--accent-primary)', cursor: 'pointer' }}
                      />
                      <span style={{ fontSize: '0.85rem', fontFamily: 'monospace', fontWeight: 700, minWidth: '40px', color: 'var(--accent-primary)' }}>
                        {scriptParams.cv.toFixed(2)}
                      </span>
                    </div>
                    <span style={{ fontSize: '0.75rem', color: 'var(--text-secondary)', fontWeight: 600, marginTop: '0.2rem', display: 'block' }}>
                      {getCvLabel(scriptParams.cv)}
                    </span>
                  </div>

                  {/* Sample size */}
                  <div className="form-group" style={{ marginBottom: 0 }}>
                    <label className="form-label">Sample Size (Bus intervals)</label>
                    <p style={{ fontSize: '0.75rem', color: 'var(--text-tertiary)', marginBottom: '0.5rem' }}>Number of headway observations to simulate and fit.</p>
                    <input
                      type="number"
                      className="form-input"
                      value={scriptParams.n_samples}
                      onChange={(e) => updateParam('n_samples', parseInt(e.target.value))}
                    />
                  </div>

                </div>
              </div>

              {/* Right Column: Console log */}
              <div style={{ flex: 1, display: 'flex', flexDirection: 'column', background: '#0a0a0a', borderLeft: '1px solid var(--border-strong)' }}>
                <div style={{ padding: '0.75rem 1rem', borderBottom: '1px solid #27272a', display: 'flex', justifyContent: 'space-between', alignItems: 'center', background: '#121212' }}>
                  <span style={{ fontSize: '0.8rem', color: '#a1a1aa', textTransform: 'uppercase', fontWeight: 600 }}>Subprocess Output Console</span>
                  <span style={{ fontSize: '0.8rem', fontWeight: 600, color: status === 'running' ? '#f59e0b' : status === 'completed' ? '#10b981' : '#71717a' }}>
                    {status.toUpperCase()}
                  </span>
                </div>
                <div style={{ flex: 1, padding: '1rem', overflowY: 'auto', fontFamily: '"Courier New", Courier, monospace', fontSize: '0.82rem', color: '#e4e4e7', background: '#000' }}>
                  {logs.length === 0 && <div style={{ color: '#52525b', fontStyle: 'italic' }}>CLI stdout stream will print here on run...</div>}
                  {logs.map((log, i) => (
                    <div key={i} style={{ color: log.type === 'stderr' ? '#ef4444' : log.type === 'status' ? '#10b981' : 'inherit', whiteSpace: 'pre-wrap', wordBreak: 'break-all', marginBottom: '0.2rem' }}>
                      {log.line || (log.type === 'status' ? `[Process Exited with status code: ${log.exit_code}]` : '')}
                    </div>
                  ))}
                </div>
                
                {/* Modal actions */}
                <div style={{ padding: '1rem', borderTop: '1px solid #27272a', background: '#121212' }}>
                  <button
                    className="btn btn-primary"
                    style={{ width: '100%', padding: '0.75rem', fontSize: '1rem', fontWeight: 600 }}
                    onClick={handleRun}
                    disabled={status === 'running'}
                  >
                    {status === 'running' ? 'Solving MLE Parameters...' : 'Solve Headway Fitting'}
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
