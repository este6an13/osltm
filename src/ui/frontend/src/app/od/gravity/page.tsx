'use client';

import { useState, useEffect, useRef } from 'react';
import { useWebSocket } from '@/hooks/useWebSocket';

interface PipelineMeta {
  pipeline_id: string;
  created_at?: string;
  params?: any;
}

interface StationInfo {
  station_code: string;
  station_name: string;
}

interface MatrixDataResponse {
  station_codes: string[];
  station_names: string[];
  time_bins: string[];
  flows: Record<string, number[][]>;
  probabilities: Record<string, number[][]>;
}

export default function GravityODPage() {
  // Pipelines & Stations loading
  const [pipelines, setPipelines] = useState<PipelineMeta[]>([]);
  const [runPipeline, setRunPipeline] = useState<PipelineMeta | null>(null);
  const [stations, setStations] = useState<StationInfo[]>([]);

  // Runs & Logs Modal State
  const [isRunModalOpen, setIsRunModalOpen] = useState(false);
  const [scriptParams, setScriptParams] = useState({
    day_type: 'WD',
    cutoff_date: '2025-11-30',
    gamma: 0.0001,
    min_days: 5,
    stations: [] as string[],
  });
  const [runId, setRunId] = useState<string | null>(null);
  const [lastExperimentId, setLastExperimentId] = useState<string | undefined>(undefined);
  const { logs, status } = useWebSocket(runId);

  // Matrix Animation Data State
  const [matrixData, setMatrixData] = useState<MatrixDataResponse | null>(null);
  const [selectedTimeIdx, setSelectedTimeIdx] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);
  const [animationSpeed, setAnimationSpeed] = useState(250); // ms per step
  const [metric, setMetric] = useState<'flows' | 'probabilities'>('flows');
  const [scaleMode, setScaleMode] = useState<'global' | 'local'>('global');
  const [hoveredCell, setHoveredCell] = useState<{ rowIdx: number; colIdx: number } | null>(null);
  const [isLoadingMatrix, setIsLoadingMatrix] = useState(false);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);

  // Stats computed from active matrix
  const [globalMaxFlow, setGlobalMaxFlow] = useState(1.0);
  const [globalMaxProb, setGlobalMaxProb] = useState(1.0);

  const animationIntervalRef = useRef<NodeJS.Timeout | null>(null);

  // Fetch Pipelines and Stations on mount
  useEffect(() => {
    fetch('http://127.0.0.1:8000/api/pipeline/experiments')
      .then((res) => res.json())
      .then((data: PipelineMeta[]) => {
        setPipelines(data);
        if (data.length > 0) setRunPipeline(data[0]);
      })
      .catch(console.error);

    fetch('http://127.0.0.1:8000/api/analysis/stations')
      .then((res) => res.json())
      .then((data: any[]) => {
        const formatted = data.map((s) => ({
          station_code: s.station_code,
          station_name: s.station_name,
        }));
        setStations(formatted);
      })
      .catch(console.error);
  }, []);

  // Update stations filter snapshot when target pipeline changes
  useEffect(() => {
    const url = runPipeline
      ? `http://127.0.0.1:8000/api/analysis/stations?pipeline_id=${runPipeline.pipeline_id}`
      : 'http://127.0.0.1:8000/api/analysis/stations';
    fetch(url)
      .then((res) => res.json())
      .then((data: any[]) => {
        const formatted = data.map((s) => ({
          station_code: s.station_code,
          station_name: s.station_name,
        }));
        setStations(formatted);
      })
      .catch(console.error);
  }, [runPipeline]);

  // Fetch Matrix Data when pipeline/experiment changes or on initialization
  const fetchMatrixData = (pId: string = 'default', eId: string = 'default') => {
    setIsLoadingMatrix(true);
    setErrorMsg(null);
    fetch(`http://127.0.0.1:8000/api/results/gravity_od/${pId}/${eId}/matrix`)
      .then((res) => {
        if (!res.ok) {
          throw new Error('Estimated matrix dataset not found for this execution.');
        }
        return res.json();
      })
      .then((data: MatrixDataResponse) => {
        setMatrixData(data);
        setSelectedTimeIdx(0);
        
        // Calculate global maximum values for scale mapping
        let maxF = 0.0001;
        let maxP = 0.0001;
        Object.values(data.flows).forEach((m) => {
          m.forEach((row) => {
            row.forEach((v) => { if (v > maxF) maxF = v; });
          });
        });
        Object.values(data.probabilities).forEach((m) => {
          m.forEach((row) => {
            row.forEach((v) => { if (v > maxP) maxP = v; });
          });
        });
        
        setGlobalMaxFlow(maxF);
        setGlobalMaxProb(maxP);
        setIsLoadingMatrix(false);
      })
      .catch((err) => {
        console.error(err);
        setErrorMsg(err.message || 'Failed to load matrix data.');
        setIsLoadingMatrix(false);
      });
  };

  // Load default matrix data on mount
  useEffect(() => {
    fetchMatrixData('default', 'default');
  }, []);

  // When a run completes, reload matrix data with the fresh experiment results
  useEffect(() => {
    if (status === 'completed' && lastExperimentId && runPipeline) {
      fetchMatrixData(runPipeline.pipeline_id, lastExperimentId);
    }
  }, [status, lastExperimentId]);

  // Handle Playback Loop
  useEffect(() => {
    if (isPlaying && matrixData) {
      animationIntervalRef.current = setInterval(() => {
        setSelectedTimeIdx((prev) => (prev >= matrixData.time_bins.length - 1 ? 0 : prev + 1));
      }, animationSpeed);
    } else {
      if (animationIntervalRef.current) {
        clearInterval(animationIntervalRef.current);
        animationIntervalRef.current = null;
      }
    }
    return () => {
      if (animationIntervalRef.current) {
        clearInterval(animationIntervalRef.current);
      }
    };
  }, [isPlaying, matrixData, animationSpeed]);

  const handleRun = async () => {
    try {
      const res = await fetch('http://127.0.0.1:8000/api/analysis/od/gravity_od/run', {
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

  const toggleStation = (code: string) => {
    setScriptParams((prev) => {
      const list = prev.stations || [];
      return {
        ...prev,
        stations: list.includes(code) ? list.filter((c) => c !== code) : [...list, code],
      };
    });
  };

  const formatPipelineLabel = (p: PipelineMeta) => {
    const step1 = (p as any).step1;
    const step3 = (p as any).step3;
    const range = step1 ? `${step1.start_date} → ${step1.end_date}` : '';
    const nStations = step3?.n_stations ? `${step3.n_stations} stations` : '';
    return `${p.pipeline_id} · ${[range, nStations].filter(Boolean).join(' · ')}`;
  };

  const formatTimeBin = (bin: string): string => {
    const clean = bin.replace('t_', '');
    const val = parseInt(clean);
    const hour = Math.floor(val / 100);
    const min = val % 100;
    const ampm = hour >= 12 ? 'PM' : 'AM';
    const displayHour = hour % 12 === 0 ? 12 : hour % 12;
    const displayMin = min.toString().padStart(2, '0');
    return `${displayHour}:${displayMin} ${ampm}`;
  };

  // Get active items for animator
  const activeTimeBin = matrixData?.time_bins[selectedTimeIdx] || '';
  const activeFlowMatrix = matrixData?.flows[activeTimeBin] || [];
  const activeProbMatrix = matrixData?.probabilities[activeTimeBin] || [];

  // Compute maximums in the active frame for local scaling
  const localMaxFlow = activeFlowMatrix.reduce((max, r) => Math.max(max, ...r), 0.0001);
  const localMaxProb = activeProbMatrix.reduce((max, r) => Math.max(max, ...r), 0.0001);

  const getCellColor = (val: number) => {
    if (val === 0) return 'rgba(255, 255, 255, 0.03)';
    
    // Select color range mapping based on scaleMode and metric
    const maxVal = metric === 'flows' 
      ? (scaleMode === 'global' ? globalMaxFlow : localMaxFlow)
      : (scaleMode === 'global' ? globalMaxProb : localMaxProb);

    // Compute logarithmic fraction for visual clarity, since flows differ in orders of magnitude
    const fraction = Math.log10(val + 1.0) / Math.log10(maxVal + 1.0);
    
    // Linear fallback if log fraction calculation goes awry
    const ratio = isNaN(fraction) ? Math.min(val / maxVal, 1.0) : Math.min(fraction, 1.0);

    // Dynamic harmonized color gradient: dark navy/purple to crimson red to orange to gold yellow
    if (ratio < 0.2) {
      return `rgba(52, 20, 92, ${ratio * 3.5 + 0.15})`; // Low values: subtle deep violet
    } else if (ratio < 0.6) {
      const red = Math.floor(100 + (120 * (ratio - 0.2)) / 0.4);
      return `rgba(${red}, 28, 76, ${ratio * 0.9 + 0.1})`; // Mid low: Crimson
    } else if (ratio < 0.85) {
      const orange = Math.floor(120 + (110 * (ratio - 0.6)) / 0.25);
      return `rgba(235, ${orange}, 40, ${ratio * 0.9 + 0.1})`; // Mid high: Vibrant Orange
    } else {
      const yellow = Math.floor(180 + (75 * (ratio - 0.85)) / 0.15);
      return `rgba(255, ${yellow}, 30, 0.95)`; // Peak: Glowing Gold/Yellow
    }
  };

  // Compute Top 5 flows for current frame
  const getTopFlows = () => {
    if (!matrixData || activeFlowMatrix.length === 0) return [];
    const list: { origin: string; dest: string; flow: number; prob: number }[] = [];
    
    const K = matrixData.station_codes.length;
    for (let i = 0; i < K; i++) {
      for (let j = 0; j < K; j++) {
        if (i === j) continue;
        const f = activeFlowMatrix[i][j];
        if (f > 0) {
          list.push({
            origin: matrixData.station_names[i],
            dest: matrixData.station_names[j],
            flow: f,
            prob: activeProbMatrix[i][j],
          });
        }
      }
    }
    return list.sort((a, b) => b.flow - a.flow).slice(0, 5);
  };

  const topFlows = getTopFlows();

  return (
    <div className="animate-fade-in" style={{ paddingBottom: '4rem' }}>
      {/* Title Header */}
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', gap: '2rem', marginBottom: '1.5rem', borderBottom: '2px solid var(--accent-primary)', paddingBottom: '0.5rem' }}>
        <div>
          <h1 style={{ fontSize: '1.75rem', fontWeight: 600, color: 'var(--accent-primary)', margin: 0 }}>Gravity OD Estimation</h1>
          <p style={{ color: 'var(--text-secondary)', fontSize: '0.9rem', marginTop: '0.25rem' }}>
            Estimate and animate time-varying Passenger Origin-Destination flow matrices using a Doubly-Constrained Entropy-Maximization Gravity Model.
          </p>
        </div>
        <button 
          className="btn btn-primary"
          onClick={() => setIsRunModalOpen(true)}
          style={{ padding: '0.65rem 1.25rem', fontSize: '0.9rem', flexShrink: 0, fontWeight: 600 }}
        >
          ▶ Run Gravity Model
        </button>
      </div>

      {/* Main Panel */}
      <div className="card glass" style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem', border: '1px solid var(--border-strong)' }}>
        
        {/* Loading / Error Overlays */}
        {isLoadingMatrix && (
          <div style={{ padding: '4rem', textAlign: 'center', color: 'var(--text-tertiary)' }}>
            <span style={{ fontSize: '1.5rem' }}>🔄 Loading dynamic OD Matrix dataset...</span>
          </div>
        )}

        {errorMsg && !isLoadingMatrix && (
          <div style={{ padding: '3rem', textAlign: 'center', background: 'rgba(239, 68, 68, 0.05)', border: '1px dashed var(--accent-error)', borderRadius: 'var(--radius-md)' }}>
            <h3 style={{ color: 'var(--accent-error)', fontSize: '1.1rem', fontWeight: 600, marginBottom: '0.5rem' }}>Matrix Data Not Available</h3>
            <p style={{ color: 'var(--text-secondary)', fontSize: '0.85rem', marginBottom: '1rem' }}>{errorMsg}</p>
            <button className="btn btn-secondary" onClick={() => fetchMatrixData('default', 'default')}>
              Load Pre-generated Baseline OD Data
            </button>
          </div>
        )}

        {/* Animator Screen */}
        {matrixData && !isLoadingMatrix && !errorMsg && (
          <div style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>
            
            {/* Control Bar */}
            <div style={{ display: 'flex', flexWrap: 'wrap', gap: '1rem', justifyContent: 'space-between', alignItems: 'center', background: 'var(--bg-secondary)', padding: '0.75rem 1rem', borderRadius: 'var(--radius-sm)', border: '1px solid var(--border-light)' }}>
              
              {/* Playback Controls */}
              <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                <button
                  onClick={() => setSelectedTimeIdx((prev) => (prev === 0 ? matrixData.time_bins.length - 1 : prev - 1))}
                  className="btn btn-secondary"
                  style={{ padding: '0.4rem 0.6rem', fontSize: '0.8rem' }}
                  title="Previous Step"
                >
                  ◀◀
                </button>
                <button
                  onClick={() => setIsPlaying(!isPlaying)}
                  className="btn btn-primary"
                  style={{
                    padding: '0.4rem 1rem',
                    fontSize: '0.85rem',
                    fontWeight: 600,
                    minWidth: '90px',
                    background: isPlaying ? '#d97706' : 'var(--accent-primary)',
                    borderColor: isPlaying ? '#b45309' : '#003366',
                  }}
                >
                  {isPlaying ? '⏸ Pause' : '▶ Play'}
                </button>
                <button
                  onClick={() => setSelectedTimeIdx((prev) => (prev === matrixData.time_bins.length - 1 ? 0 : prev + 1))}
                  className="btn btn-secondary"
                  style={{ padding: '0.4rem 0.6rem', fontSize: '0.8rem' }}
                  title="Next Step"
                >
                  ▶▶
                </button>
              </div>

              {/* Speed Controller */}
              <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                <span style={{ fontSize: '0.75rem', color: 'var(--text-tertiary)', fontWeight: 600 }}>Interval:</span>
                <input
                  type="range"
                  min="100"
                  max="1000"
                  step="50"
                  value={animationSpeed}
                  onChange={(e) => setAnimationSpeed(parseInt(e.target.value))}
                  style={{ width: '80px', accentColor: 'var(--accent-primary)' }}
                />
                <span style={{ fontSize: '0.75rem', fontFamily: 'monospace', color: 'var(--text-secondary)' }}>{animationSpeed}ms</span>
              </div>

              {/* Metric Selector */}
              <div style={{ display: 'flex', gap: '0.2rem', background: 'var(--bg-highlight)', padding: '2px', borderRadius: '4px' }}>
                <button
                  onClick={() => setMetric('flows')}
                  style={{
                    padding: '0.3rem 0.75rem',
                    fontSize: '0.75rem',
                    fontWeight: 600,
                    borderRadius: '3px',
                    background: metric === 'flows' ? 'var(--bg-primary)' : 'transparent',
                    color: metric === 'flows' ? 'var(--accent-primary)' : 'var(--text-secondary)',
                    boxShadow: metric === 'flows' ? 'var(--shadow-sm)' : 'none',
                    transition: 'all 100ms',
                  }}
                >
                  📊 Estimated Flows
                </button>
                <button
                  onClick={() => setMetric('probabilities')}
                  style={{
                    padding: '0.3rem 0.75rem',
                    fontSize: '0.75rem',
                    fontWeight: 600,
                    borderRadius: '3px',
                    background: metric === 'probabilities' ? 'var(--bg-primary)' : 'transparent',
                    color: metric === 'probabilities' ? 'var(--accent-primary)' : 'var(--text-secondary)',
                    boxShadow: metric === 'probabilities' ? 'var(--shadow-sm)' : 'none',
                    transition: 'all 100ms',
                  }}
                >
                  🕸️ Routing Probabilities
                </button>
              </div>

              {/* Scale mode Selector */}
              <div style={{ display: 'flex', alignItems: 'center', gap: '0.4rem' }}>
                <span style={{ fontSize: '0.75rem', color: 'var(--text-tertiary)', fontWeight: 600 }}>Color Scale:</span>
                <button
                  onClick={() => setScaleMode(scaleMode === 'global' ? 'local' : 'global')}
                  className="btn btn-secondary"
                  style={{
                    padding: '0.3rem 0.6rem',
                    fontSize: '0.75rem',
                    background: scaleMode === 'global' ? 'var(--bg-highlight)' : 'var(--bg-primary)',
                    borderColor: scaleMode === 'global' ? 'var(--border-strong)' : 'var(--border-light)',
                    color: 'var(--text-secondary)',
                    fontWeight: 600,
                  }}
                  title={scaleMode === 'global' ? 'Scaled across all 77 intervals' : 'Scaled dynamically per interval'}
                >
                  {scaleMode === 'global' ? '🌍 Global Scale' : '🎯 Local Scale'}
                </button>
              </div>
            </div>

            {/* Matrix Timeline Slider */}
            <div style={{ background: 'var(--bg-secondary)', border: '1px solid var(--border-light)', borderRadius: 'var(--radius-sm)', padding: '1rem 1.25rem' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.4rem' }}>
                <span style={{ fontSize: '0.85rem', color: 'var(--text-secondary)', fontWeight: 600 }}>
                  Timeline: <span style={{ fontFamily: 'monospace', color: 'var(--accent-primary)', fontSize: '0.95rem' }}>{formatTimeBin(activeTimeBin)}</span>
                </span>
                <span style={{ fontSize: '0.75rem', color: 'var(--text-tertiary)', fontFamily: 'monospace' }}>
                  Interval {selectedTimeIdx + 1} / {matrixData.time_bins.length}
                </span>
              </div>
              <input
                type="range"
                min="0"
                max={matrixData.time_bins.length - 1}
                value={selectedTimeIdx}
                onChange={(e) => setSelectedTimeIdx(parseInt(e.target.value))}
                style={{ width: '100%', height: '8px', accentColor: 'var(--accent-primary)', cursor: 'pointer', background: 'var(--border-light)' }}
              />
              <div style={{ display: 'flex', justifyContent: 'space-between', marginTop: '0.25rem', fontSize: '0.7rem', color: 'var(--text-tertiary)' }}>
                <span>04:00 AM</span>
                <span>07:30 AM (AM Peak)</span>
                <span>12:00 PM</span>
                <span>06:00 PM (PM Peak)</span>
                <span>11:00 PM</span>
              </div>
            </div>

            {/* Matrix Heatmap Grid Container */}
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 300px', gap: '1.5rem', alignItems: 'start' }}>
              
              {/* Left Column: Heatmap Grid */}
              <div 
                style={{
                  background: '#09090b',
                  border: '1px solid var(--border-strong)',
                  borderRadius: 'var(--radius-sm)',
                  padding: '1.5rem',
                  overflow: 'auto',
                  maxHeight: '620px',
                  display: 'flex',
                  justifyContent: 'center',
                }}
              >
                <div style={{ position: 'relative' }}>
                  {/* Grid Layout */}
                  <div
                    style={{
                      display: 'grid',
                      gridTemplateColumns: `repeat(${matrixData.station_codes.length}, minmax(18px, 1fr))`,
                      gap: '1px',
                      background: '#18181b',
                      padding: '1px',
                    }}
                  >
                    {activeFlowMatrix.map((rowValues, rIdx) =>
                      rowValues.map((val, cIdx) => {
                        const isHovered = hoveredCell?.rowIdx === rIdx && hoveredCell?.colIdx === cIdx;
                        const isRowColHovered = hoveredCell?.rowIdx === rIdx || hoveredCell?.colIdx === cIdx;
                        
                        return (
                          <div
                            key={`${rIdx}-${cIdx}`}
                            onMouseEnter={() => setHoveredCell({ rowIdx: rIdx, colIdx: cIdx })}
                            onMouseLeave={() => setHoveredCell(null)}
                            style={{
                              aspectRatio: '1',
                              minWidth: '18px',
                              background: getCellColor(metric === 'flows' ? val : activeProbMatrix[rIdx][cIdx]),
                              cursor: 'pointer',
                              border: isHovered 
                                ? '2px solid #ffffff' 
                                : (isRowColHovered ? '1px solid rgba(255, 255, 255, 0.2)' : 'none'),
                              boxShadow: isHovered ? '0 0 8px #ffffff' : 'none',
                              zIndex: isHovered ? 10 : 1,
                              transition: 'background-color 150ms ease-out',
                            }}
                          />
                        );
                      })
                    )}
                  </div>

                  {/* Matrix Axis Indicators (Labels overlay if hovered) */}
                  <div style={{ marginTop: '0.5rem', display: 'flex', justifyContent: 'space-between', fontSize: '0.65rem', color: '#a1a1aa', fontFamily: 'monospace' }}>
                    <span>Origin Boarding Stations (Row: Top to Bottom) ──&gt;</span>
                    <span>Destination Alighting (Col: Left to Right)</span>
                  </div>
                </div>
              </div>

              {/* Right Column: Analytical Panel */}
              <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
                
                {/* 1. Cell Hover Details Panel */}
                <div className="card" style={{ padding: '1rem', background: 'var(--bg-secondary)', border: '1px solid var(--border-strong)', margin: 0 }}>
                  <h3 style={{ fontSize: '0.85rem', fontWeight: 700, color: 'var(--accent-primary)', borderBottom: '1px solid var(--border-light)', paddingBottom: '0.4rem', marginBottom: '0.75rem', textTransform: 'uppercase' }}>
                    🔍 Cell Inspection
                  </h3>
                  {hoveredCell ? (
                    <div style={{ display: 'flex', flexDirection: 'column', gap: '0.6rem', fontSize: '0.8rem' }}>
                      <div>
                        <span style={{ color: 'var(--text-tertiary)', fontSize: '0.7rem', display: 'block', fontWeight: 600 }}>ORIGIN STATION</span>
                        <span style={{ fontWeight: 600, color: 'var(--text-primary)' }}>
                          [{matrixData.station_codes[hoveredCell.rowIdx]}] {matrixData.station_names[hoveredCell.rowIdx]}
                        </span>
                      </div>
                      <div>
                        <span style={{ color: 'var(--text-tertiary)', fontSize: '0.7rem', display: 'block', fontWeight: 600 }}>DESTINATION STATION</span>
                        <span style={{ fontWeight: 600, color: 'var(--text-primary)' }}>
                          [{matrixData.station_codes[hoveredCell.colIdx]}] {matrixData.station_names[hoveredCell.colIdx]}
                        </span>
                      </div>
                      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '0.5rem', marginTop: '0.25rem', borderTop: '1px dashed var(--border-light)', paddingTop: '0.5rem' }}>
                        <div>
                          <span style={{ color: 'var(--text-tertiary)', fontSize: '0.7rem', display: 'block', fontWeight: 600 }}>EST. FLOW</span>
                          <span style={{ fontSize: '0.95rem', fontWeight: 700, color: 'var(--accent-primary)' }}>
                            {activeFlowMatrix[hoveredCell.rowIdx][hoveredCell.colIdx].toFixed(1)} /15m
                          </span>
                        </div>
                        <div>
                          <span style={{ color: 'var(--text-tertiary)', fontSize: '0.7rem', display: 'block', fontWeight: 600 }}>PROBABILITY</span>
                          <span style={{ fontSize: '0.95rem', fontWeight: 700, color: 'var(--accent-success)' }}>
                            {(activeProbMatrix[hoveredCell.rowIdx][hoveredCell.colIdx] * 100).toFixed(2)}%
                          </span>
                        </div>
                      </div>
                    </div>
                  ) : (
                    <div style={{ padding: '1rem 0', textAlign: 'center', color: 'var(--text-tertiary)', fontSize: '0.8rem', fontStyle: 'italic' }}>
                      Hover over any cell in the 2D grid to read dynamic values.
                    </div>
                  )}
                </div>

                {/* 2. Top Commute Flows Panel */}
                <div className="card" style={{ padding: '1rem', background: 'var(--bg-secondary)', border: '1px solid var(--border-strong)', margin: 0, flex: 1 }}>
                  <h3 style={{ fontSize: '0.85rem', fontWeight: 700, color: 'var(--accent-primary)', borderBottom: '1px solid var(--border-light)', paddingBottom: '0.4rem', marginBottom: '0.75rem', textTransform: 'uppercase' }}>
                    🔥 Top 5 Active Flows
                  </h3>
                  <div style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem' }}>
                    {topFlows.map((tf, i) => (
                      <div
                        key={i}
                        style={{
                          background: 'var(--bg-primary)',
                          border: '1px solid var(--border-light)',
                          borderRadius: '3px',
                          padding: '0.5rem',
                          fontSize: '0.75rem',
                          position: 'relative',
                        }}
                      >
                        <div style={{ display: 'flex', justifyContent: 'space-between', fontWeight: 600, color: 'var(--text-primary)', marginBottom: '0.2rem' }}>
                          <span>{i + 1}. {tf.origin.split(' ')[0]} ➔ {tf.dest.split(' ')[0]}</span>
                          <span style={{ color: 'var(--accent-primary)' }}>{tf.flow.toFixed(1)}</span>
                        </div>
                        <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.68rem', color: 'var(--text-tertiary)' }}>
                          <span>Routing Probability:</span>
                          <span style={{ fontWeight: 600, color: 'var(--accent-success)' }}>{(tf.prob * 100).toFixed(1)}%</span>
                        </div>
                      </div>
                    ))}
                    {topFlows.length === 0 && (
                      <span style={{ fontStyle: 'italic', color: 'var(--text-tertiary)', fontSize: '0.75rem' }}>No flow active in this frame.</span>
                    )}
                  </div>
                </div>
              </div>

            </div>

          </div>
        )}
      </div>

      {/* Analytical PNG Plots below */}
      {matrixData && !isLoadingMatrix && !errorMsg && (
        <div style={{ marginTop: '2.5rem' }}>
          <h2 style={{ fontSize: '1.25rem', color: 'var(--accent-primary)', fontWeight: 600, marginBottom: '1rem', borderBottom: '1px solid var(--border-strong)', paddingBottom: '0.4rem' }}>
            📈 Generated Analytical Visualizations
          </h2>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(320px, 1fr))', gap: '1.5rem' }}>
            
            {/* Morning Peak */}
            <div className="card glass" style={{ padding: '0.75rem', border: '1px solid var(--border-strong)', background: 'var(--bg-primary)' }}>
              <h3 style={{ fontSize: '0.9rem', fontWeight: 600, color: 'var(--text-primary)', padding: '0.25rem 0.5rem', background: 'var(--bg-secondary)', borderBottom: '1px solid var(--border-light)', margin: '0 0 0.5rem 0', fontFamily: 'monospace' }}>
                gravity_od_flow_heatmap_am.png
              </h3>
              <img
                src={`http://127.0.0.1:8000/api/results/gravity_od/${runPipeline?.pipeline_id || 'default'}/${lastExperimentId || 'default'}/gravity_od_flow_heatmap_am.png/view`}
                alt="Morning Peak"
                style={{ width: '100%', height: 'auto', borderRadius: 'var(--radius-sm)' }}
                onError={(e) => {
                  // fallback to default
                  (e.target as HTMLImageElement).src = 'http://127.0.0.1:8000/api/results/gravity_od/default/default/gravity_od_flow_heatmap_am.png/view';
                }}
              />
            </div>

            {/* Evening Peak */}
            <div className="card glass" style={{ padding: '0.75rem', border: '1px solid var(--border-strong)', background: 'var(--bg-primary)' }}>
              <h3 style={{ fontSize: '0.9rem', fontWeight: 600, color: 'var(--text-primary)', padding: '0.25rem 0.5rem', background: 'var(--bg-secondary)', borderBottom: '1px solid var(--border-light)', margin: '0 0 0.5rem 0', fontFamily: 'monospace' }}>
                gravity_od_flow_heatmap_pm.png
              </h3>
              <img
                src={`http://127.0.0.1:8000/api/results/gravity_od/${runPipeline?.pipeline_id || 'default'}/${lastExperimentId || 'default'}/gravity_od_flow_heatmap_pm.png/view`}
                alt="Evening Peak"
                style={{ width: '100%', height: 'auto', borderRadius: 'var(--radius-sm)' }}
                onError={(e) => {
                  // fallback to default
                  (e.target as HTMLImageElement).src = 'http://127.0.0.1:8000/api/results/gravity_od/default/default/gravity_od_flow_heatmap_pm.png/view';
                }}
              />
            </div>

            {/* Portal Commute pattern Reversal */}
            <div className="card glass" style={{ padding: '0.75rem', border: '1px solid var(--border-strong)', background: 'var(--bg-primary)' }}>
              <h3 style={{ fontSize: '0.9rem', fontWeight: 600, color: 'var(--text-primary)', padding: '0.25rem 0.5rem', background: 'var(--bg-secondary)', borderBottom: '1px solid var(--border-light)', margin: '0 0 0.5rem 0', fontFamily: 'monospace' }}>
                portal_commute_reversal.png
              </h3>
              <img
                src={`http://127.0.0.1:8000/api/results/gravity_od/${runPipeline?.pipeline_id || 'default'}/${lastExperimentId || 'default'}/portal_commute_reversal.png/view`}
                alt="Portal commute pattern Reversal"
                style={{ width: '100%', height: 'auto', borderRadius: 'var(--radius-sm)' }}
                onError={(e) => {
                  // fallback to default
                  (e.target as HTMLImageElement).src = 'http://127.0.0.1:8000/api/results/gravity_od/default/default/portal_commute_reversal.png/view';
                }}
              />
            </div>

          </div>
        </div>
      )}

      {/* Configuration & Execution Modal */}
      {isRunModalOpen && (
        <div style={{ position: 'fixed', inset: 0, zIndex: 100, display: 'flex', alignItems: 'center', justifyContent: 'center', background: 'rgba(0,0,0,0.5)', backdropFilter: 'none' }}>
          <div style={{ background: 'var(--bg-primary)', border: '1px solid var(--border-strong)', borderRadius: 'var(--radius-md)', width: '90vw', height: '85vh', display: 'flex', flexDirection: 'column', overflow: 'hidden', boxShadow: 'var(--shadow-lg)' }}>
            
            {/* Modal Header */}
            <div style={{ padding: '1rem 1.5rem', borderBottom: '1px solid var(--border-strong)', display: 'flex', justifyContent: 'space-between', alignItems: 'center', background: 'var(--bg-secondary)' }}>
              <h2 style={{ margin: 0, fontSize: '1.25rem', color: 'var(--accent-primary)', fontWeight: 600 }}>Configure & Run: Gravity OD Model</h2>
              <button onClick={() => setIsRunModalOpen(false)} style={{ background: 'transparent', border: 'none', color: 'var(--text-tertiary)', fontSize: '1.5rem', cursor: 'pointer', lineHeight: 1 }}>&times;</button>
            </div>

            {/* Modal Body */}
            <div style={{ display: 'flex', flex: 1, overflow: 'hidden' }}>
              
              {/* Left Column: Config parameters */}
              <div style={{ flex: 1, padding: '1.5rem', overflowY: 'auto', borderRight: '1px solid var(--border-strong)', background: 'var(--bg-primary)' }}>
                
                {/* Pipeline Selector */}
                <div className="form-group" style={{ marginBottom: '2rem' }}>
                  <label className="form-label">Target Pipeline Data Context</label>
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

                {/* Script Parameters */}
                <h3 style={{ fontSize: '1.1rem', marginBottom: '1rem', paddingBottom: '0.5rem', borderBottom: '1px solid var(--border-strong)', color: 'var(--accent-primary)', fontWeight: 600 }}>Parameters</h3>
                <div style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>
                  
                  {/* day_type */}
                  <div className="form-group" style={{ marginBottom: 0 }}>
                    <label className="form-label">Day Type</label>
                    <p style={{ fontSize: '0.75rem', color: 'var(--text-tertiary)', marginBottom: '0.5rem' }}>Choose the binned day-type to run the gravity matrix balancing on.</p>
                    <select
                      className="form-select"
                      value={scriptParams.day_type}
                      onChange={(e) => updateParam('day_type', e.target.value)}
                    >
                      <option value="WD">Weekday (WD)</option>
                      <option value="SA">Saturday (SA)</option>
                      <option value="SU">Sunday (SU)</option>
                      <option value="HO">Holiday (HO)</option>
                    </select>
                  </div>

                  {/* cutoff_date */}
                  <div className="form-group" style={{ marginBottom: 0 }}>
                    <label className="form-label">Cutoff Date</label>
                    <p style={{ fontSize: '0.75rem', color: 'var(--text-tertiary)', marginBottom: '0.5rem' }}>YYYY-MM-DD filter for historical training counts.</p>
                    <input
                      type="text"
                      className="form-input"
                      value={scriptParams.cutoff_date}
                      onChange={(e) => updateParam('cutoff_date', e.target.value)}
                    />
                  </div>

                  {/* gamma friction decay */}
                  <div className="form-group" style={{ marginBottom: 0 }}>
                    <label className="form-label">Gamma Decay Parameter (Friction Factor)</label>
                    <p style={{ fontSize: '0.75rem', color: 'var(--text-tertiary)', marginBottom: '0.5rem' }}>Decay weight per meter for spatial interaction cost exp(-gamma * dist).</p>
                    <input
                      type="number"
                      step="0.00001"
                      className="form-input"
                      value={scriptParams.gamma}
                      onChange={(e) => updateParam('gamma', parseFloat(e.target.value))}
                    />
                  </div>

                  {/* min_days */}
                  <div className="form-group" style={{ marginBottom: 0 }}>
                    <label className="form-label">Minimum Days required</label>
                    <p style={{ fontSize: '0.75rem', color: 'var(--text-tertiary)', marginBottom: '0.5rem' }}>Filter out stations without sufficient replicate day binnings.</p>
                    <input
                      type="number"
                      className="form-input"
                      value={scriptParams.min_days}
                      onChange={(e) => updateParam('min_days', parseInt(e.target.value))}
                    />
                  </div>

                  {/* stations picker */}
                  <div className="form-group" style={{ marginBottom: 0 }}>
                    <label className="form-label">Choose Stations (Subset list)</label>
                    <p style={{ fontSize: '0.75rem', color: 'var(--text-tertiary)', marginBottom: '0.5rem' }}>Filter matrix generation to a subset of stations. Empty includes all available stations.</p>
                    <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.5rem', maxHeight: '180px', overflowY: 'auto', padding: '0.5rem', background: 'var(--bg-secondary)', border: '1px solid var(--border-strong)' }}>
                      <label style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', fontSize: '0.85rem', width: '100%', borderBottom: '1px solid var(--border-light)', paddingBottom: '0.25rem', marginBottom: '0.25rem' }}>
                        <input
                          type="checkbox"
                          checked={scriptParams.stations.length === 0}
                          onChange={() => updateParam('stations', [])}
                        />
                        <span style={{ color: 'var(--accent-primary)', fontWeight: 600 }}>All Stations (Default)</span>
                      </label>
                      {stations.map((s) => (
                        <label key={s.station_code} style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', fontSize: '0.8rem', background: 'var(--bg-primary)', padding: '0.2rem 0.4rem', border: '1px solid var(--border-light)', borderRadius: '2px' }}>
                          <input
                            type="checkbox"
                            checked={scriptParams.stations.includes(s.station_code)}
                            onChange={() => toggleStation(s.station_code)}
                          />
                          <span style={{ fontFamily: 'monospace', fontWeight: 600 }}>{s.station_code}</span> – {s.station_name.split(' ')[0]}
                        </label>
                      ))}
                    </div>
                  </div>

                </div>
              </div>

              {/* Right Column: Execution terminal logs */}
              <div style={{ flex: 1, display: 'flex', flexDirection: 'column', background: '#0a0a0a', borderLeft: '1px solid var(--border-strong)' }}>
                <div style={{ padding: '0.75rem 1rem', borderBottom: '1px solid #27272a', display: 'flex', justifyContent: 'space-between', alignItems: 'center', background: '#121212' }}>
                  <span style={{ fontSize: '0.8rem', color: '#a1a1aa', textTransform: 'uppercase', fontWeight: 600 }}>Execution Logs Console</span>
                  <span style={{ fontSize: '0.8rem', fontWeight: 600, color: status === 'running' ? '#f59e0b' : status === 'completed' ? '#10b981' : '#71717a' }}>
                    {status.toUpperCase()}
                  </span>
                </div>
                <div style={{ flex: 1, padding: '1rem', overflowY: 'auto', fontFamily: '"Courier New", Courier, monospace', fontSize: '0.82rem', color: '#e4e4e7', background: '#000' }}>
                  {logs.length === 0 && <div style={{ color: '#52525b', fontStyle: 'italic' }}>Subprocess CLI output will stream here on execution...</div>}
                  {logs.map((log, i) => (
                    <div key={i} style={{ color: log.type === 'stderr' ? '#ef4444' : log.type === 'status' ? '#10b981' : 'inherit', whiteSpace: 'pre-wrap', wordBreak: 'break-all', marginBottom: '0.2rem' }}>
                      {log.line || (log.type === 'status' ? `[Process Exited with status code: ${log.exit_code}]` : '')}
                    </div>
                  ))}
                </div>
                
                {/* Modal Run button actions */}
                <div style={{ padding: '1rem', borderTop: '1px solid #27272a', background: '#121212' }}>
                  <button
                    className="btn btn-primary"
                    style={{ width: '100%', padding: '0.75rem', fontSize: '1rem', fontWeight: 600 }}
                    onClick={handleRun}
                    disabled={status === 'running'}
                  >
                    {status === 'running' ? 'Balancing OD Matrices...' : 'Execute IPF Gravity Estimation'}
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
