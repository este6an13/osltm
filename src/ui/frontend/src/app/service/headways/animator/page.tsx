'use client';

import { useState, useEffect } from 'react';
import Link from 'next/link';

interface PipelineMeta {
  pipeline_id: string;
  created_at?: string;
}

interface ExperimentMeta {
  experiment_id: string;
  pipeline_id: string;
  created_at?: string;
  params?: any;
}

interface Station {
  station_id: string;
  name: string;
  x: number;
  y: number;
  troncal: string;
}

interface Edge {
  source: string;
  target: string;
  edge_type: string;
}

interface BusEvent {
  station_id: string;
  arrival: number;
  departure: number;
}

interface Bus {
  bus_id: string;
  timeline: BusEvent[];
}

interface RouteSimulation {
  route_code: string;
  color: string;
  stations: string[];
  views: {
    fitted: {
      description: string;
      buses: Bus[];
    };
    physical: {
      description: string;
      buses: Bus[];
    };
  };
}

interface SimulationData {
  metadata: {
    period: string;
    cv: number;
    simulated_routes: string[];
  };
  stations: Record<string, Station>;
  edges: Edge[];
  routes: Record<string, RouteSimulation>;
}

export default function ServiceAnimatorPage() {
  // Pipelines & Experiments selection
  const [pipelines, setPipelines] = useState<PipelineMeta[]>([]);
  const [historyExperiments, setHistoryExperiments] = useState<ExperimentMeta[]>([]);
  const [selectedPipelineId, setSelectedPipelineId] = useState<string>('default');
  const [selectedExpId, setSelectedExpId] = useState<string>('default');

  // Active loaded data
  const [simulationData, setSimulationData] = useState<SimulationData | null>(null);
  const [selectedRoute, setSelectedRoute] = useState<string>('');
  const [viewMode, setViewMode] = useState<'fitted' | 'physical'>('fitted');
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);

  // Playback Control state
  const [isPlaying, setIsPlaying] = useState<boolean>(false);
  const [currentTime, setCurrentTime] = useState<number>(0);
  const [speedMultiplier, setSpeedMultiplier] = useState<number>(10);
  const [maxTime, setMaxTime] = useState<number>(3600); // 1 hour default

  // Hover states
  const [hoveredStation, setHoveredStation] = useState<Station | null>(null);
  const [hoveredBus, setHoveredBus] = useState<{ id: string; status: string; nextStop: string } | null>(null);
  const [tooltipPos, setTooltipPos] = useState<{ x: number; y: number }>({ x: 0, y: 0 });

  // Load pipelines on mount
  useEffect(() => {
    fetch('http://127.0.0.1:8000/api/pipeline/experiments')
      .then((res) => res.json())
      .then((data: PipelineMeta[]) => {
        setPipelines(data);
      })
      .catch(console.error);

    fetch('http://127.0.0.1:8000/api/results/headway_fitting')
      .then((res) => (res.ok ? res.json() : []))
      .then((data: ExperimentMeta[]) => {
        setHistoryExperiments(data);
      })
      .catch(console.error);
  }, []);

  // Fetch simulation data function
  const fetchSimulationData = (pId: string = 'default', eId: string = 'default') => {
    setIsLoading(true);
    setErrorMsg(null);
    fetch(`http://127.0.0.1:8000/api/results/service/headway_fitting/${pId}/${eId}/traversal`)
      .then((res) => {
        if (!res.ok) {
          throw new Error('Traversal simulation timeline JSON not found for this execution.');
        }
        return res.json();
      })
      .then((data: SimulationData) => {
        setSimulationData(data);
        const routesList = Object.keys(data.routes);
        if (routesList.length > 0) {
          setSelectedRoute(routesList[0]);
        }
        setIsLoading(false);
      })
      .catch((err) => {
        console.error(err);
        setErrorMsg(err.message || 'Failed to load spatial simulation timeline.');
        setIsLoading(false);
      });
  };

  // Load default simulation on mount
  useEffect(() => {
    fetchSimulationData('default', 'default');
  }, []);

  // Recalculate max time when route or view changes
  useEffect(() => {
    if (!simulationData || !selectedRoute) return;
    const routeSim = simulationData.routes[selectedRoute];
    if (!routeSim) return;

    const buses = routeSim.views[viewMode].buses;
    if (buses.length === 0) return;

    let highestDep = 0;
    buses.forEach((b) => {
      const lastEvent = b.timeline[b.timeline.length - 1];
      if (lastEvent && lastEvent.departure > highestDep) {
        highestDep = lastEvent.departure;
      }
    });

    setMaxTime(highestDep);
    setCurrentTime(0);
    setIsPlaying(false);
  }, [simulationData, selectedRoute, viewMode]);

  // Delta clock ticker using requestAnimationFrame for smooth 60fps
  useEffect(() => {
    let animationFrameId: number;
    let lastTime = performance.now();

    const tick = () => {
      if (isPlaying) {
        const now = performance.now();
        const deltaSec = (now - lastTime) / 1000;
        lastTime = now;

        setCurrentTime((prev) => {
          const next = prev + deltaSec * speedMultiplier;
          if (next >= maxTime) {
            setIsPlaying(false);
            return maxTime;
          }
          return next;
        });
      } else {
        lastTime = performance.now();
      }
      animationFrameId = requestAnimationFrame(tick);
    };

    animationFrameId = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(animationFrameId);
  }, [isPlaying, speedMultiplier, maxTime]);

  const formatClock = (seconds: number) => {
    const hrs = Math.floor(seconds / 3600);
    const mins = Math.floor((seconds % 3600) / 60);
    const secs = Math.floor(seconds % 60);
    return `${hrs.toString().padStart(2, '0')}:${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
  };

  if (isLoading) {
    return (
      <div style={{ padding: '3rem', textAlign: 'center', color: 'var(--text-secondary)' }}>
        <div style={{ display: 'inline-block', width: '2rem', height: '2rem', border: '3px solid var(--border-strong)', borderTopColor: 'var(--accent-primary)', borderRadius: '50%', animation: 'spin 1s linear infinite', marginBottom: '1rem' }}></div>
        <p style={{ fontWeight: 500 }}>Loading vector spatial network & stochastic dispatches...</p>
        <style>{`@keyframes spin { to { transform: rotate(360deg); } }`}</style>
      </div>
    );
  }

  if (errorMsg || !simulationData) {
    return (
      <div style={{ padding: '2rem' }}>
        <div className="card" style={{ borderLeft: '4px solid var(--accent-error)', background: 'var(--bg-secondary)', padding: '1.5rem', maxWidth: '600px', margin: '0 auto', borderRadius: 'var(--radius-md)' }}>
          <h4 style={{ color: 'var(--accent-error)', fontWeight: 700, fontSize: '1.1rem', marginBottom: '0.5rem' }}>No Traversal Simulation Found</h4>
          <p style={{ fontSize: '0.88rem', color: 'var(--text-secondary)', marginBottom: '1.5rem', lineHeight: 1.5 }}>
            {errorMsg || 'Please run a headway fitting pipeline first to generate the simulation timeline.'}
          </p>
          <div style={{ display: 'flex', gap: '1rem' }}>
            <Link href="/service/headways" className="btn btn-secondary" style={{ padding: '0.5rem 1rem', fontSize: '0.85rem' }}>
              ← Go to Headway Fitting
            </Link>
            <button onClick={() => fetchSimulationData('default', 'default')} className="btn btn-primary" style={{ padding: '0.5rem 1rem', fontSize: '0.85rem' }}>
              Retry Default
            </button>
          </div>
        </div>
      </div>
    );
  }

  const stations = Object.values(simulationData.stations);
  const xCoords = stations.map((s) => s.x);
  const yCoords = stations.map((s) => s.y);
  
  const xMin = Math.min(...xCoords);
  const xMax = Math.max(...xCoords);
  const yMin = Math.min(...yCoords);
  const yMax = Math.max(...yCoords);

  // SVG Canvas dimensions
  const svgWidth = 900;
  const svgHeight = 620;
  const padding = 50;

  // Projection mapping scaling functions (Flipping Y so North is UP)
  const projectX = (x: number) => {
    if (xMax === xMin) return svgWidth / 2;
    return ((x - xMin) / (xMax - xMin)) * (svgWidth - 2 * padding) + padding;
  };

  const projectY = (y: number) => {
    if (yMax === yMin) return svgHeight / 2;
    return svgHeight - (((y - yMin) / (yMax - yMin)) * (svgHeight - 2 * padding) + padding);
  };

  // Selected Route Details
  const routeSim = simulationData.routes[selectedRoute];
  const activeView = routeSim?.views[viewMode];
  const routeStations = routeSim?.stations || [];
  const routeColor = routeSim?.color || '#3b82f6';

  // Compute active buses positions at currentTime
  const activeBusesPositions = activeView?.buses.map((bus) => {
    const timeline = bus.timeline;
    if (timeline.length === 0) return null;

    const tArr0 = timeline[0].arrival;
    const tDepLast = timeline[timeline.length - 1].departure;

    if (currentTime < tArr0 || currentTime > tDepLast) {
      return null; // Bus not active
    }

    // Check if dwelling at a station
    for (let j = 0; j < timeline.length; j++) {
      const event = timeline[j];
      if (currentTime >= event.arrival && currentTime <= event.departure) {
        const station = simulationData.stations[event.station_id];
        return {
          bus_id: bus.bus_id,
          x: station.x,
          y: station.y,
          status: 'dwelling',
          current_station: station.name,
          next_station: j < timeline.length - 1 ? simulationData.stations[timeline[j + 1].station_id].name : 'Terminus',
        };
      }
    }

    // Check if traveling between stations
    for (let j = 0; j < timeline.length - 1; j++) {
      const depEvent = timeline[j];
      const arrEvent = timeline[j + 1];

      if (currentTime > depEvent.departure && currentTime < arrEvent.arrival) {
        const sCurr = simulationData.stations[depEvent.station_id];
        const sNext = simulationData.stations[arrEvent.station_id];

        const duration = arrEvent.arrival - depEvent.departure;
        const elapsed = currentTime - depEvent.departure;
        const lambda = duration > 0 ? elapsed / duration : 0;

        const x = sCurr.x + (sNext.x - sCurr.x) * lambda;
        const y = sCurr.y + (sNext.y - sCurr.y) * lambda;

        return {
          bus_id: bus.bus_id,
          x,
          y,
          status: 'traveling',
          current_station: sCurr.name,
          next_station: sNext.name,
        };
      }
    }

    return null;
  }).filter((b) => b !== null) as Array<{
    bus_id: string;
    x: number;
    y: number;
    status: 'dwelling' | 'traveling';
    current_station: string;
    next_station: string;
  }> || [];

  return (
    <div style={{ padding: '1.5rem', paddingBottom: '4rem' }}>
      {/* Page Header */}
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', gap: '2rem', marginBottom: '1.5rem', borderBottom: '2px solid var(--accent-primary)', paddingBottom: '0.5rem' }}>
        <div>
          <h1 style={{ fontSize: '1.75rem', fontWeight: 600, color: 'var(--accent-primary)', margin: 0 }}>
            🚌 Stochastic 2D Spatial Network Bus Traversal Animator
          </h1>
          <p style={{ color: 'var(--text-secondary)', fontSize: '0.9rem', marginTop: '0.25rem' }}>
            Visualizing headway dispatches and en-route delay propagation along the projected TransMilenio grid.
          </p>
        </div>
      </div>

      {/* Main Grid Layout */}
      <div style={{ display: 'flex', gap: '1.5rem', flexWrap: 'wrap' }}>
        
        {/* Left Control Column */}
        <div style={{ flex: '1 1 300px', display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>
          
          {/* Execution Selection Card */}
          <div className="card glass" style={{ padding: '1.25rem', border: '1px solid var(--border-light)', borderRadius: 'var(--radius-md)' }}>
            <h3 style={{ fontSize: '0.85rem', fontWeight: 700, color: 'var(--accent-primary)', textTransform: 'uppercase', letterSpacing: '0.05em', marginBottom: '1rem' }}>
              📂 Pipeline Execution
            </h3>
            
            <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
              <div className="form-group" style={{ margin: 0 }}>
                <label style={{ display: 'block', fontSize: '0.78rem', fontWeight: 600, color: 'var(--text-secondary)', marginBottom: '0.35rem' }}>Pipeline ID</label>
                <select
                  className="form-select"
                  value={selectedPipelineId}
                  onChange={(e) => {
                    setSelectedPipelineId(e.target.value);
                    const exps = historyExperiments.filter((x) => x.pipeline_id === e.target.value);
                    if (exps.length > 0) {
                      setSelectedExpId(exps[0].experiment_id);
                      fetchSimulationData(e.target.value, exps[0].experiment_id);
                    } else {
                      setSelectedExpId('default');
                      fetchSimulationData(e.target.value, 'default');
                    }
                  }}
                  style={{ fontFamily: 'monospace', fontSize: '0.82rem', padding: '0.4rem 0.5rem' }}
                >
                  <option value="default">Default baseline context</option>
                  {Array.from(new Set(historyExperiments.map((e) => e.pipeline_id).filter(Boolean))).map((pId) => (
                    <option key={pId} value={pId}>
                      {pId}
                    </option>
                  ))}
                </select>
              </div>

              <div className="form-group" style={{ margin: 0 }}>
                <label style={{ display: 'block', fontSize: '0.78rem', fontWeight: 600, color: 'var(--text-secondary)', marginBottom: '0.35rem' }}>Experiment Run</label>
                <select
                  className="form-select"
                  value={selectedExpId}
                  onChange={(e) => {
                    setSelectedExpId(e.target.value);
                    fetchSimulationData(selectedPipelineId, e.target.value);
                  }}
                  style={{ fontFamily: 'monospace', fontSize: '0.82rem', padding: '0.4rem 0.5rem' }}
                >
                  {selectedPipelineId === 'default' ? (
                    <option value="default">Baseline Run</option>
                  ) : (
                    historyExperiments
                      .filter((x) => x.pipeline_id === selectedPipelineId)
                      .map((exp) => (
                        <option key={exp.experiment_id} value={exp.experiment_id}>
                          {exp.experiment_id}
                        </option>
                      ))
                  )}
                </select>
              </div>

              <div style={{ borderTop: '1px solid var(--border-light)', margin: '0.25rem 0' }}></div>

              <div className="form-group" style={{ margin: 0 }}>
                <label style={{ display: 'block', fontSize: '0.78rem', fontWeight: 600, color: 'var(--text-secondary)', marginBottom: '0.35rem' }}>Active Route Trace</label>
                <select
                  className="form-select font-medium"
                  value={selectedRoute}
                  onChange={(e) => setSelectedRoute(e.target.value)}
                  style={{ fontSize: '0.85rem', padding: '0.45rem 0.5rem', fontWeight: 600 }}
                >
                  {Object.keys(simulationData.routes).map((rCode) => (
                    <option key={rCode} value={rCode}>
                      Route {rCode}
                    </option>
                  ))}
                </select>
              </div>
            </div>
          </div>

          {/* Model View Explainer Card */}
          <div className="card glass" style={{ padding: '1.25rem', border: '1px solid var(--border-light)', borderRadius: 'var(--radius-md)', flex: 1, display: 'flex', flexDirection: 'column' }}>
            <h3 style={{ fontSize: '0.85rem', fontWeight: 700, color: 'var(--accent-primary)', textTransform: 'uppercase', letterSpacing: '0.05em', marginBottom: '0.75rem' }}>
              🧠 Model View Mode
            </h3>

            {/* Premium Selector Tabs */}
            <div style={{ display: 'flex', background: 'var(--bg-tertiary)', padding: '0.25rem', borderRadius: 'var(--radius-sm)', gap: '0.25rem', marginBottom: '1rem' }}>
              <button
                onClick={() => setViewMode('fitted')}
                style={{
                  flex: 1,
                  textAlign: 'center',
                  fontSize: '0.76rem',
                  padding: '0.4rem 0.25rem',
                  border: 'none',
                  borderRadius: 'var(--radius-sm)',
                  cursor: 'pointer',
                  fontWeight: 600,
                  transition: 'all 0.15s ease-in-out',
                  ...(viewMode === 'fitted'
                    ? { background: 'var(--bg-primary)', color: 'var(--accent-primary)', boxShadow: 'var(--shadow-sm)' }
                    : { background: 'transparent', color: 'var(--text-secondary)' }),
                }}
              >
                📊 Fitted MLE Model
              </button>
              <button
                onClick={() => setViewMode('physical')}
                style={{
                  flex: 1,
                  textAlign: 'center',
                  fontSize: '0.76rem',
                  padding: '0.4rem 0.25rem',
                  border: 'none',
                  borderRadius: 'var(--radius-sm)',
                  cursor: 'pointer',
                  fontWeight: 600,
                  transition: 'all 0.15s ease-in-out',
                  ...(viewMode === 'physical'
                    ? { background: 'var(--bg-primary)', color: 'var(--accent-primary)', boxShadow: 'var(--shadow-sm)' }
                    : { background: 'transparent', color: 'var(--text-secondary)' }),
                }}
              >
                🌀 Physical Degradation
              </button>
            </div>

            <div style={{ fontSize: '0.82rem', color: 'var(--text-secondary)', lineHeight: 1.5, display: 'flex', flexDirection: 'column', gap: '0.75rem', flex: 1 }}>
              {viewMode === 'fitted' ? (
                <>
                  <div style={{ fontWeight: 700, color: 'var(--accent-primary)' }}>View 1: Fitted MLE Model (Nominal Corridor)</div>
                  <p>
                    Buses are dispatched stochastically at the terminal stop. The dispatch headway spacing is drawn directly from the **winning MLE probability distribution** (Gamma, Erlang, or Log-Normal) fitted to arrival telemetry.
                  </p>
                  <div style={{ background: 'var(--bg-secondary)', padding: '0.75rem', borderLeft: '3px solid var(--accent-primary)', borderRadius: 'var(--radius-sm)', fontSize: '0.78rem' }}>
                    <strong>Corridor Physics</strong>: Cruising speeds are constant at 30 km/h and station dwells are fixed at 30s. No en-route traffic noise is applied, keeping spacing stable downstream.
                  </div>
                </>
              ) : (
                <>
                  <div style={{ fontWeight: 700, color: 'var(--accent-success)' }}>View 2: Physical Model (Stochastic Degradation)</div>
                  <p>
                    Buses are dispatched from the terminal stop at perfectly uniform planned intervals (e.g. exactly every {simulationData.routes[selectedRoute]?.views.physical.buses[0] ? Math.round((simulationData.routes[selectedRoute].views.physical.buses[0].timeline[0].departure - 30) / 60) : 6} minutes).
                  </p>
                  <div style={{ background: 'var(--bg-secondary)', padding: '0.75rem', borderLeft: '3px solid var(--accent-success)', borderRadius: 'var(--radius-sm)', fontSize: '0.78rem' }}>
                    <strong>Corridor Physics</strong>: Cruising speeds vary per driver ($v_b \sim N(30, 1.8)$), dwells are stochastic ($N(30, 5)$), and **log-normal traffic delay noise** ($C_v$) is applied segment-by-segment, causing bus bunching!
                  </div>
                </>
              )}
              <div style={{ borderTop: '1px solid var(--border-light)', paddingTop: '0.75rem', marginTop: 'auto', fontSize: '0.78rem' }}>
                <strong style={{ color: 'var(--text-primary)' }}>Thesis Impact:</strong> Demonstrates how en-route delay noise propagates, visually validating the degradation of scheduled headways into emergent bunching downstream.
              </div>
            </div>
          </div>
        </div>

        {/* Right Main Map Column */}
        <div style={{ flex: '3 1 600px', display: 'flex', flexDirection: 'column', border: '1px solid var(--border-light)', borderRadius: 'var(--radius-md)', background: 'var(--bg-primary)', overflow: 'hidden', boxShadow: 'var(--shadow-sm)' }}>
          
          {/* Playback Control Bar */}
          <div style={{ background: 'var(--bg-secondary)', borderBottom: '1px solid var(--border-light)', padding: '1rem', display: 'flex', flexWrap: 'wrap', gap: '1rem', alignItems: 'center', justifyContent: 'space-between' }}>
            {/* Left controls */}
            <div style={{ display: 'flex', gap: '0.5rem' }}>
              <button
                className="btn btn-primary"
                onClick={() => setIsPlaying(!isPlaying)}
                style={{
                  padding: '0.45rem 1rem',
                  fontSize: '0.82rem',
                  fontWeight: 700,
                  display: 'flex',
                  alignItems: 'center',
                  gap: '0.25rem',
                  background: isPlaying ? 'var(--accent-warning)' : 'var(--accent-primary)',
                }}
              >
                {isPlaying ? '⏸ Pause' : '▶ Play'}
              </button>
              <button
                className="btn btn-secondary"
                onClick={() => {
                  setCurrentTime(0);
                  setIsPlaying(false);
                }}
                style={{ padding: '0.45rem 1rem', fontSize: '0.82rem', fontWeight: 600 }}
              >
                🔄 Reset
              </button>
            </div>

            {/* Time Slider Scrubber */}
            <div style={{ flex: 1, minWidth: '220px', display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
              <input
                type="range"
                min="0"
                max={maxTime}
                value={currentTime}
                onChange={(e) => {
                  setCurrentTime(parseFloat(e.target.value));
                  setIsPlaying(false);
                }}
                style={{
                  flex: 1,
                  cursor: 'pointer',
                  height: '0.35rem',
                  borderRadius: 'var(--radius-full)',
                  accentColor: 'var(--accent-primary)',
                  background: 'var(--bg-tertiary)',
                  outline: 'none',
                }}
              />
              <span style={{ fontSize: '0.85rem', fontFamily: 'monospace', fontWeight: 700, background: 'var(--bg-tertiary)', color: 'var(--text-primary)', padding: '0.25rem 0.5rem', borderRadius: 'var(--radius-sm)', minWidth: '70px', textAlign: 'center' }}>
                {formatClock(currentTime)}
              </span>
            </div>

            {/* Speed Multipliers */}
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.25rem' }}>
              <span style={{ fontSize: '0.7rem', fontWeight: 700, color: 'var(--text-tertiary)', textTransform: 'uppercase', marginRight: '0.25rem' }}>Speed</span>
              {[1, 5, 10, 25, 50].map((spd) => (
                <button
                  key={spd}
                  onClick={() => setSpeedMultiplier(spd)}
                  style={{
                    fontSize: '0.72rem',
                    padding: '0.25rem 0.5rem',
                    border: 'none',
                    borderRadius: 'var(--radius-sm)',
                    cursor: 'pointer',
                    fontWeight: 700,
                    transition: 'all 0.1s ease-in-out',
                    ...(speedMultiplier === spd
                      ? { background: 'var(--accent-primary)', color: 'var(--bg-primary)' }
                      : { background: 'var(--bg-tertiary)', color: 'var(--text-secondary)' }),
                  }}
                >
                  {spd}x
                </button>
              ))}
            </div>
          </div>

          {/* SVG Map Render Canvas */}
          <div style={{ position: 'relative', background: 'var(--bg-secondary)', display: 'flex', alignItems: 'center', justifyContent: 'center', padding: '1rem', minHeight: '520px', flex: 1 }}>
            <svg
              width="100%"
              height="100%"
              viewBox={`0 0 ${svgWidth} ${svgHeight}`}
              style={{ maxHeight: '620px', width: '100%', display: 'block' }}
            >
              {/* Background Network Edges */}
              {simulationData.edges.map((edge, idx) => {
                const sNode = simulationData.stations[edge.source];
                const tNode = simulationData.stations[edge.target];
                if (!sNode || !tNode) return null;
                return (
                  <line
                    key={`edge-${idx}`}
                    x1={projectX(sNode.x)}
                    y1={projectY(sNode.y)}
                    x2={projectX(tNode.x)}
                    y2={projectY(tNode.y)}
                    stroke="var(--border-strong)"
                    strokeWidth={0.8}
                    opacity={0.35}
                  />
                );
              })}

              {/* Glowing Active Route sequence Track */}
              {routeStations.length > 1 && (
                <g>
                  {/* Glowing Underlay */}
                  <path
                    d={`M ${routeStations
                      .map((sid) => {
                        const s = simulationData.stations[sid];
                        return s ? `${projectX(s.x)} ${projectY(s.y)}` : '';
                      })
                      .filter(Boolean)
                      .join(' L ')}`}
                    fill="none"
                    stroke={routeColor}
                    strokeWidth={5}
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    opacity={0.25}
                    style={{ animation: 'pulse 2s infinite ease-in-out' }}
                  />
                  {/* Solid Overlay */}
                  <path
                    d={`M ${routeStations
                      .map((sid) => {
                        const s = simulationData.stations[sid];
                        return s ? `${projectX(s.x)} ${projectY(s.y)}` : '';
                      })
                      .filter(Boolean)
                      .join(' L ')}`}
                    fill="none"
                    stroke={routeColor}
                    strokeWidth={2}
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    opacity={0.8}
                  />
                </g>
              )}

              {/* Station Dots Layer */}
              {stations.map((station) => {
                const isRouteStop = routeStations.includes(station.station_id);
                return (
                  <circle
                    key={station.station_id}
                    cx={projectX(station.x)}
                    cy={projectY(station.y)}
                    r={isRouteStop ? 4 : 2}
                    fill={isRouteStop ? routeColor : 'var(--text-tertiary)'}
                    stroke={isRouteStop ? 'var(--bg-primary)' : 'none'}
                    strokeWidth={0.75}
                    opacity={isRouteStop ? 0.95 : 0.3}
                    style={{ cursor: 'pointer', transition: 'all 0.15s' }}
                    onMouseEnter={(e) => {
                      setHoveredStation(station);
                      setTooltipPos({ x: e.clientX, y: e.clientY });
                    }}
                    onMouseLeave={() => setHoveredStation(null)}
                  />
                );
              })}

              {/* Dwell pulsator ring */}
              {activeBusesPositions
                .filter((b) => b.status === 'dwelling')
                .map((bus) => (
                  <circle
                    key={`pulse-${bus.bus_id}`}
                    cx={projectX(bus.x)}
                    cy={projectY(bus.y)}
                    r={10}
                    fill="none"
                    stroke={routeColor}
                    strokeWidth={1.5}
                    opacity={0.5}
                    style={{ animation: 'ping 1.5s infinite linear' }}
                  />
                ))}

              {/* Bus indicators */}
              {activeBusesPositions.map((bus) => (
                <circle
                  key={bus.bus_id}
                  cx={projectX(bus.x)}
                  cy={projectY(bus.y)}
                  r={6}
                  fill={routeColor}
                  stroke="var(--bg-primary)"
                  strokeWidth={1.5}
                  style={{ cursor: 'pointer', filter: 'drop-shadow(0px 2px 3px rgba(0,0,0,0.25))' }}
                  onMouseEnter={(e) => {
                    setHoveredBus({
                      id: bus.bus_id,
                      status: bus.status === 'dwelling' ? `Dwelling at ${bus.current_station}` : `Traveling to ${bus.next_station}`,
                      nextStop: bus.next_station,
                    });
                    setTooltipPos({ x: e.clientX, y: e.clientY });
                  }}
                  onMouseLeave={() => setHoveredBus(null)}
                />
              ))}
            </svg>

            {/* HUD Overlay Info Box */}
            <div style={{ position: 'absolute', top: '1rem', left: '1rem', background: 'rgba(255,255,255,0.92)', border: '1px solid var(--border-light)', padding: '0.75rem', borderRadius: 'var(--radius-sm)', fontSize: '0.75rem', boxShadow: 'var(--shadow-sm)', display: 'flex', flexDirection: 'column', gap: '0.25rem', zIndex: 10 }}>
              <div style={{ fontWeight: 700, color: 'var(--accent-secondary)', textTransform: 'uppercase', fontSize: '0.7rem', borderBottom: '1px solid var(--border-light)', paddingBottom: '0.25rem', marginBottom: '0.25rem' }}>Parameters HUD</div>
              <div style={{ color: 'var(--text-secondary)' }}>Operating Period: <span style={{ fontWeight: 700, color: 'var(--text-primary)', textTransform: 'uppercase' }}>{simulationData.metadata.period}</span></div>
              <div style={{ color: 'var(--text-secondary)' }}>Traffic Delay Noise (Cv): <span style={{ fontWeight: 700, color: 'var(--text-primary)' }}>{simulationData.metadata.cv}</span></div>
              <div style={{ color: 'var(--text-secondary)' }}>Buses active: <span style={{ fontWeight: 700, color: 'var(--text-primary)' }}>{activeBusesPositions.length} / 30</span></div>
              <div style={{ color: 'var(--text-secondary)' }}>Route Color: <span style={{ fontWeight: 700, color: routeColor }}>{routeColor}</span></div>
            </div>

            {/* Hover Tooltips */}
            {hoveredStation && (
              <div
                style={{
                  position: 'fixed',
                  background: 'var(--accent-secondary)',
                  color: 'var(--bg-primary)',
                  fontSize: '0.76rem',
                  padding: '0.5rem 0.75rem',
                  borderRadius: 'var(--radius-sm)',
                  boxShadow: 'var(--shadow-lg)',
                  pointerEvents: 'none',
                  zIndex: 1000,
                  display: 'flex',
                  flexDirection: 'column',
                  gap: '0.15rem',
                  border: '1px solid rgba(255,255,255,0.1)',
                  left: tooltipPos.x + 12,
                  top: tooltipPos.y - 12,
                }}
              >
                <div style={{ fontWeight: 700 }}>{hoveredStation.name}</div>
                <div style={{ fontSize: '0.7rem', opacity: 0.85 }}>Troncal Corridor: {hoveredStation.troncal}</div>
                <div style={{ fontSize: '0.65rem', opacity: 0.65, fontFamily: 'monospace', borderTop: '1px solid rgba(255,255,255,0.2)', marginTop: '0.25rem', paddingTop: '0.15rem' }}>
                  X: {hoveredStation.x.toFixed(1)} | Y: {hoveredStation.y.toFixed(1)}
                </div>
              </div>
            )}

            {hoveredBus && (
              <div
                style={{
                  position: 'fixed',
                  background: 'var(--accent-secondary)',
                  color: 'var(--bg-primary)',
                  fontSize: '0.76rem',
                  padding: '0.5rem 0.75rem',
                  borderRadius: 'var(--radius-sm)',
                  boxShadow: 'var(--shadow-lg)',
                  pointerEvents: 'none',
                  zIndex: 1000,
                  display: 'flex',
                  flexDirection: 'column',
                  gap: '0.15rem',
                  border: '1px solid rgba(255,255,255,0.1)',
                  left: tooltipPos.x + 12,
                  top: tooltipPos.y - 12,
                }}
              >
                <div style={{ fontWeight: 700, color: '#f59e0b' }}>🚌 Bus ID: {hoveredBus.id}</div>
                <div style={{ fontSize: '0.72rem' }}>Status: {hoveredBus.status}</div>
                <div style={{ fontSize: '0.68rem', opacity: 0.85 }}>Target Stop: {hoveredBus.nextStop}</div>
              </div>
            )}
          </div>
        </div>
      </div>
      
      {/* Global CSS animations injections */}
      <style>{`
        @keyframes ping {
          0% { transform: scale(1); opacity: 0.6; }
          70% { transform: scale(1.6); opacity: 0; }
          100% { transform: scale(1.6); opacity: 0; }
        }
        @keyframes pulse {
          0% { opacity: 0.2; }
          50% { opacity: 0.45; }
          100% { opacity: 0.2; }
        }
      `}</style>
    </div>
  );
}
