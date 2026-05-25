"use client";

import { useState, useEffect } from "react";
import dynamic from "next/dynamic";
import { useWebSocket } from "@/hooks/useWebSocket";

// Dynamically import the force graph to avoid SSR issues with canvas
const ForceGraph2D = dynamic(() => import("react-force-graph-2d"), { ssr: false });

interface PipelineMeta {
  pipeline_id: string;
  created_at?: string;
}

interface ExperimentMeta {
  experiment_id: string;
  pipeline_id?: string;
  exit_code?: number;
}

export default function NetworkAnalysisPage() {
  const [script, setScript] = useState<any>(null);
  const [scriptParams, setScriptParams] = useState<any>({});
  
  // Pipeline context for running
  const [pipelines, setPipelines] = useState<PipelineMeta[]>([]);
  const [runPipeline, setRunPipeline] = useState<PipelineMeta | null>(null);

  // Modal & Run state
  const [isRunModalOpen, setIsRunModalOpen] = useState(false);
  const [runId, setRunId] = useState<string | null>(null);
  const [lastExperimentId, setLastExperimentId] = useState<string | undefined>(undefined);
  const { logs, status } = useWebSocket(runId);

  // Results state
  const [experiments, setExperiments] = useState<ExperimentMeta[]>([]);
  const [selectedPipelineId, setSelectedPipelineId] = useState<string | null>(null);
  const [selectedExp, setSelectedExp] = useState<string | null>(null);
  const [files, setFiles] = useState<string[]>([]);
  
  // View state
  const [viewMode, setViewMode] = useState<"interactive" | "static">("interactive");
  const [graphData, setGraphData] = useState<{ nodes: any[]; links: any[] } | null>(null);
  const [graphLoading, setGraphLoading] = useState(false);
  const [selectedStatic, setSelectedStatic] = useState<string | null>(null);

  const SCRIPT_KEY = "network/analyze";
  const OUTPUT_DIR = "network";

  // Initial load
  useEffect(() => {
    // Fetch script definition
    fetch("http://127.0.0.1:8000/api/analysis")
      .then(res => res.json())
      .then(data => {
        if (data[SCRIPT_KEY]) {
          setScript(data[SCRIPT_KEY]);
          const defs = data[SCRIPT_KEY].params || [];
          const initialParams: any = {};
          defs.forEach((p: any) => {
            initialParams[p.name] = p.default ?? "";
          });
          setScriptParams(initialParams);
        }
      })
      .catch(console.error);

    // Fetch pipelines
    fetch("http://127.0.0.1:8000/api/pipeline/experiments")
      .then(res => res.json())
      .then((data: PipelineMeta[]) => {
        setPipelines(data);
        if (data.length > 0) setRunPipeline(data[0]);
      })
      .catch(console.error);
  }, []);

  // Fetch past experiments
  const fetchExperiments = () => {
    fetch(`http://127.0.0.1:8000/api/results/${OUTPUT_DIR}`)
      .then(res => {
        if (!res.ok) return [];
        return res.json();
      })
      .then((data: ExperimentMeta[]) => {
        setExperiments(data);
        if (lastExperimentId && data.some(e => e.experiment_id === lastExperimentId)) {
          setSelectedExp(lastExperimentId);
          const match = data.find(e => e.experiment_id === lastExperimentId);
          if (match?.pipeline_id) setSelectedPipelineId(match.pipeline_id);
        } else if (data.length > 0 && !selectedExp) {
          const first = data[0];
          setSelectedExp(first.experiment_id);
          if (first.pipeline_id) setSelectedPipelineId(first.pipeline_id);
        }
      })
      .catch(console.error);
  };

  useEffect(() => {
    fetchExperiments();
  }, [lastExperimentId]); // Refetch when a run finishes

  // Fetch files when selected experiment changes
  useEffect(() => {
    if (!selectedExp || !selectedPipelineId) {
      setFiles([]);
      return;
    }
    fetch(`http://127.0.0.1:8000/api/results/${OUTPUT_DIR}/${selectedPipelineId}/${selectedExp}`)
      .then(res => res.json())
      .then(data => {
        setFiles(data.files ?? []);
        const imgs = (data.files ?? []).filter((f: string) => f.endsWith(".png"));
        if (imgs.length > 0) setSelectedStatic(imgs[0]);
      })
      .catch(console.error);
  }, [selectedExp, selectedPipelineId]);

  // Load Graph Data when in interactive mode and experiment is selected
  useEffect(() => {
    if (!selectedExp || !selectedPipelineId || viewMode !== "interactive") return;
    
    setGraphLoading(true);
    fetch(`http://127.0.0.1:8000/api/results/${OUTPUT_DIR}/${selectedPipelineId}/${selectedExp}/network_graph.json/view`)
      .then(res => {
        if (!res.ok) throw new Error("Graph data not found in this run");
        return res.json();
      })
      .then(data => {
        const links = data.links || data.edges || [];
        setGraphData({
          nodes: data.nodes.map((n: any) => ({ ...n, id: n.id })),
          links: links.map((l: any) => ({ ...l, source: l.source, target: l.target }))
        });
      })
      .catch(err => {
        console.error(err);
        setGraphData(null);
      })
      .finally(() => setGraphLoading(false));
  }, [selectedExp, selectedPipelineId, viewMode]);

  const handleRun = async () => {
    if (!script) return;
    try {
      const res = await fetch(`http://127.0.0.1:8000/api/analysis/${SCRIPT_KEY}/run`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          params: scriptParams,
          pipeline_id: runPipeline?.pipeline_id ?? null,
          exp_id: null,
        }),
      });
      const data = await res.json();
      if (data.run_id) setRunId(data.run_id);
      if (data.experiment_id) {
         // Clear graph so we show loading state while backend runs
         setGraphData(null); 
         setLastExperimentId(data.experiment_id);
      }
    } catch (err) {
      console.error(err);
    }
  };

  const updateParam = (name: string, value: any) => {
    setScriptParams((prev: any) => ({ ...prev, [name]: value }));
  };

  const images = files.filter(f => f.endsWith(".png"));
  const expsForPipeline = experiments.filter(e => e.pipeline_id === selectedPipelineId);
  const activePipelines = Array.from(new Set(experiments.map(e => e.pipeline_id).filter(Boolean))) as string[];

  return (
    <div className="animate-fade-in" style={{ paddingBottom: "4rem" }}>
      <h1 className="page-title">Network Analysis</h1>
      <p className="page-subtitle" style={{ marginBottom: "1.5rem" }}>
        Calculate node centralities and generate interactive graph layouts.
      </p>

      {/* Script Header (Description & Run Button) */}
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", gap: "2rem", marginBottom: "2rem", padding: "1rem", backgroundColor: "var(--bg-secondary)", borderRadius: "var(--radius-md)", border: "1px solid var(--border-strong)" }}>
        <div>
          <h2 style={{ fontSize: "1.5rem", color: "var(--text-primary)", marginBottom: "0.5rem", fontWeight: 600 }}>
            {script?.name || "Network Topology Analysis"}
          </h2>
          <p style={{ color: "var(--text-secondary)", fontSize: "0.9rem", maxWidth: "800px" }}>
            {script?.description || "Loading..."}
          </p>
        </div>
        <button 
          className="btn btn-primary"
          onClick={() => setIsRunModalOpen(true)}
          style={{ padding: "0.75rem 1.5rem", fontSize: "1rem", flexShrink: 0 }}
        >
          ▶ Run Experiment
        </button>
      </div>

      {/* Results Controls */}
      <div className="card glass" style={{ marginBottom: "2rem", padding: "1rem" }}>
        <div style={{ display: "flex", flexWrap: "wrap", gap: "1rem", alignItems: "center", justifyContent: "space-between" }}>
          
          <div style={{ display: "flex", gap: "1rem" }}>
            <div className="form-group" style={{ margin: 0, minWidth: "200px" }}>
              <label className="form-label" style={{ fontSize: "0.75rem" }}>Results Pipeline</label>
              <select 
                className="form-select" 
                value={selectedPipelineId || ""} 
                onChange={e => {
                  setSelectedPipelineId(e.target.value);
                  const firstExp = experiments.find(x => x.pipeline_id === e.target.value);
                  if (firstExp) setSelectedExp(firstExp.experiment_id);
                }}
                style={{ fontFamily: "monospace", fontSize: "0.85rem" }}
              >
                {!selectedPipelineId && <option value="">Select Pipeline...</option>}
                {activePipelines.map(p => <option key={p} value={p}>{p}</option>)}
              </select>
            </div>
            
            <div className="form-group" style={{ margin: 0, minWidth: "200px" }}>
              <label className="form-label" style={{ fontSize: "0.75rem" }}>Experiment Run</label>
              <select 
                className="form-select" 
                value={selectedExp || ""} 
                onChange={e => setSelectedExp(e.target.value)}
                style={{ fontFamily: "monospace", fontSize: "0.85rem" }}
              >
                {!selectedExp && <option value="">Select Experiment...</option>}
                {expsForPipeline.map(exp => (
                  <option key={exp.experiment_id} value={exp.experiment_id}>
                    {exp.experiment_id} {exp.exit_code === 0 ? "✓" : exp.exit_code !== undefined ? "✗" : ""}
                  </option>
                ))}
              </select>
            </div>
          </div>

          <div style={{ display: "flex", gap: "0.5rem" }}>
            <button 
              onClick={() => setViewMode("interactive")}
              className={`btn ${viewMode === "interactive" ? "btn-primary" : "btn-secondary"}`}
              style={{ padding: "0.5rem 1rem" }}
            >
              Interactive Map
            </button>
            <button 
              onClick={() => setViewMode("static")}
              className={`btn ${viewMode === "static" ? "btn-primary" : "btn-secondary"}`}
              style={{ padding: "0.5rem 1rem" }}
            >
              Static View
            </button>
          </div>
        </div>
      </div>

      {/* Main Content Area */}
      {selectedExp ? (
        <div style={{ display: "flex", gap: "2rem", height: "700px" }}>
          
          {/* Left Col: The View */}
          <div style={{ flex: 1, backgroundColor: "#000", border: "1px solid var(--border-strong)", borderRadius: "8px", overflow: "hidden", display: "flex", alignItems: "center", justifyContent: "center", position: "relative" }}>
            {viewMode === "interactive" ? (
              graphLoading ? (
                <p style={{ color: "#888" }}>Loading graph data...</p>
              ) : graphData ? (
                <ForceGraph2D
                  graphData={graphData}
                  nodeLabel={(node: any) => `${node.name || node.id}\nDegree: ${node.degree_centrality?.toFixed(3)}\nBetweenness: ${node.betweenness_centrality?.toFixed(3)}\nCloseness: ${node.closeness_centrality?.toFixed(3)}`}
                  nodeAutoColorBy="trazado"
                  nodeVal={(node: any) => (node.degree_centrality || 0) * 100 + 1}
                  linkColor={() => "rgba(255,255,255,0.2)"}
                  backgroundColor="#000000"
                  width={900}
                  height={700}
                />
              ) : (
                <p style={{ color: "#888" }}>Network JSON not found in this run.</p>
              )
            ) : (
              <div style={{ width: "100%", height: "100%", display: "flex", alignItems: "center", justifyContent: "center" }}>
                 {selectedStatic ? (
                   <img 
                     src={`http://127.0.0.1:8000/api/results/${OUTPUT_DIR}/${selectedPipelineId}/${selectedExp}/${selectedStatic}/view`}
                     alt="Network Plot" 
                     style={{ maxWidth: "100%", maxHeight: "100%", objectFit: "contain" }}
                   />
                 ) : (
                   <p style={{ color: "#888" }}>No images generated in this run.</p>
                 )}
              </div>
            )}
          </div>

          {/* Right Col: Stats / Selection */}
          <div style={{ width: "300px", display: "flex", flexDirection: "column", gap: "1rem" }}>
            
            <div style={{ padding: "1.5rem", backgroundColor: "var(--bg-secondary)", borderRadius: "8px", border: "1px solid var(--border-strong)" }}>
              <h3 style={{ marginTop: 0, marginBottom: "1rem", color: "var(--accent-primary)" }}>Network Stats</h3>
              {graphData ? (
                <ul style={{ listStyle: "none", padding: 0, margin: 0, color: "var(--text-secondary)", display: "flex", flexDirection: "column", gap: "0.5rem" }}>
                  <li><strong>Nodes:</strong> <span style={{ color: "var(--text-primary)" }}>{graphData.nodes.length}</span></li>
                  <li><strong>Edges:</strong> <span style={{ color: "var(--text-primary)" }}>{graphData.links.length}</span></li>
                </ul>
              ) : (
                <p style={{ color: "var(--text-tertiary)", margin: 0 }}>No data</p>
              )}
            </div>

            {viewMode === "static" && images.length > 0 && (
              <div style={{ padding: "1.5rem", backgroundColor: "var(--bg-secondary)", borderRadius: "8px", border: "1px solid var(--border-strong)", flex: 1, overflowY: "auto" }}>
                <h3 style={{ marginTop: 0, marginBottom: "1rem", color: "var(--accent-primary)" }}>Available Maps</h3>
                <div style={{ display: "flex", flexDirection: "column", gap: "0.5rem" }}>
                  {images.map(img => (
                    <button
                      key={img}
                      onClick={() => setSelectedStatic(img)}
                      style={{
                        padding: "0.5rem",
                        textAlign: "left",
                        backgroundColor: selectedStatic === img ? "var(--bg-highlight)" : "transparent",
                        color: selectedStatic === img ? "var(--accent-primary)" : "var(--text-secondary)",
                        border: selectedStatic === img ? "1px solid var(--accent-primary)" : "1px solid var(--border-strong)",
                        borderRadius: "4px",
                        cursor: "pointer",
                        fontSize: "0.85rem"
                      }}
                    >
                      {img.replace(".png", "")}
                    </button>
                  ))}
                </div>
              </div>
            )}
          </div>
        </div>
      ) : (
        <div style={{ padding: "3rem", textAlign: "center", color: "var(--text-tertiary)", backgroundColor: "var(--bg-secondary)", borderRadius: "8px" }}>
          No network analysis experiments found. Click "Run Experiment" to generate data.
        </div>
      )}

      {/* Run Modal */}
      {isRunModalOpen && script && (
        <div style={{ position: "fixed", inset: 0, zIndex: 100, display: "flex", alignItems: "center", justifyContent: "center", background: "rgba(0,0,0,0.4)", backdropFilter: "none" }}>
          <div style={{ background: "var(--bg-primary)", border: "1px solid var(--border-strong)", borderRadius: "var(--radius-md)", width: "90vw", height: "85vh", display: "flex", flexDirection: "column", overflow: "hidden", boxShadow: "var(--shadow-lg)" }}>
            
            {/* Modal Header */}
            <div style={{ padding: "1rem 1.5rem", borderBottom: "1px solid var(--border-strong)", display: "flex", justifyContent: "space-between", alignItems: "center", background: "var(--bg-secondary)" }}>
              <h2 style={{ margin: 0, fontSize: "1.25rem", color: "var(--accent-primary)" }}>Configure & Run: {script.name}</h2>
              <button onClick={() => setIsRunModalOpen(false)} style={{ background: "transparent", border: "none", color: "var(--text-tertiary)", fontSize: "1.5rem", cursor: "pointer", lineHeight: 1 }}>&times;</button>
            </div>

            {/* Modal Body */}
            <div style={{ display: "flex", flex: 1, overflow: "hidden" }}>
              
              {/* Left Column: Config */}
              <div style={{ flex: 1, padding: "1.5rem", overflowY: "auto", borderRight: "1px solid var(--border-strong)", background: "var(--bg-primary)" }}>
                
                {/* Pipeline Selector */}
                <div className="form-group" style={{ marginBottom: "2rem" }}>
                  <label className="form-label">Target Pipeline</label>
                  <p style={{ fontSize: "0.8rem", color: "var(--text-tertiary)", marginBottom: "0.5rem" }}>Select the pipeline data context to run this script on.</p>
                  <select 
                    className="form-select" 
                    value={runPipeline?.pipeline_id || ""} 
                    onChange={e => {
                      const p = pipelines.find(x => x.pipeline_id === e.target.value);
                      if (p) setRunPipeline(p);
                    }}
                    style={{ fontFamily: "monospace" }}
                  >
                    {pipelines.map(p => (
                      <option key={p.pipeline_id} value={p.pipeline_id}>{p.pipeline_id}</option>
                    ))}
                  </select>
                </div>

                {/* Script Parameters */}
                <h3 style={{ fontSize: "1.1rem", marginBottom: "1rem", paddingBottom: "0.5rem", borderBottom: "1px solid var(--border-strong)", color: "var(--accent-primary)" }}>Parameters</h3>
                <div style={{ display: "flex", flexDirection: "column", gap: "1.5rem" }}>
                  {script.params.map((param: any) => (
                    <div key={param.name} className="form-group" style={{ marginBottom: 0 }}>
                      <label className="form-label">{param.name.replace(/_/g, " ").replace(/\b\w/g, (l: string) => l.toUpperCase())}</label>
                      {param.description && <p style={{ fontSize: "0.75rem", color: "var(--text-tertiary)", marginBottom: "0.5rem" }}>{param.description}</p>}

                      {param.type === "choice" && (
                        <select className="form-select" value={scriptParams[param.name] || ""} onChange={e => updateParam(param.name, e.target.value)}>
                          {param.choices.map((c: string) => <option key={c} value={c}>{c}</option>)}
                        </select>
                      )}
                    </div>
                  ))}
                </div>
              </div>

              {/* Right Column: Terminal */}
              <div style={{ flex: 1, display: "flex", flexDirection: "column", background: "#0a0a0a", borderLeft: "1px solid var(--border-strong)" }}>
                <div style={{ padding: "0.75rem 1rem", borderBottom: "1px solid #27272a", display: "flex", justifyContent: "space-between", alignItems: "center", background: "#121212" }}>
                  <span style={{ fontSize: "0.8rem", color: "#a1a1aa", textTransform: "uppercase", fontWeight: 600 }}>Execution Log</span>
                  <span style={{ fontSize: "0.8rem", fontWeight: 600, color: status === "running" ? "#f59e0b" : status === "completed" ? "#10b981" : "#71717a" }}>
                    {status.toUpperCase()}
                  </span>
                </div>
                <div style={{ flex: 1, padding: "1rem", overflowY: "auto", fontFamily: '"Courier New", Courier, monospace', fontSize: "0.85rem", color: "#e4e4e7", background: "#000" }}>
                  {logs.length === 0 && <div style={{ color: "#52525b", fontStyle: "italic" }}>Console output will appear here...</div>}
                  {logs.map((log, i) => (
                    <div key={i} style={{ color: log.type === "stderr" ? "#ef4444" : log.type === "status" ? "#10b981" : "inherit", whiteSpace: "pre-wrap", wordBreak: "break-all", marginBottom: "0.2rem" }}>
                      {log.line || (log.type === "status" ? `[Exited: ${log.exit_code}]` : "")}
                    </div>
                  ))}
                </div>
                
                {/* Actions */}
                <div style={{ padding: "1rem", borderTop: "1px solid #27272a", background: "#121212" }}>
                  <button
                    className="btn btn-primary"
                    style={{ width: "100%", padding: "0.75rem", fontSize: "1rem" }}
                    onClick={handleRun}
                    disabled={status === "running"}
                  >
                    {status === "running" ? "Running Script..." : "Run Script"}
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
