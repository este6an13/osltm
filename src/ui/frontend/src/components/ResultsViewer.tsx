'use client';

import { useState, useEffect } from 'react';

interface ExperimentMeta {
  experiment_id: string;
  pipeline_id?: string;
  script?: string;
  created_at?: string;
  exit_code?: number;
  params?: any;
}

interface ResultsViewerProps {
  outputDir: string;
  experimentId?: string;   // auto-select after a fresh run
}

export default function ResultsViewer({ outputDir, experimentId }: ResultsViewerProps) {
  const [experiments, setExperiments] = useState<ExperimentMeta[]>([]);
  const [selectedPipelineId, setSelectedPipelineId] = useState<string | null>(null);
  const [selectedExp, setSelectedExp] = useState<string | null>(experimentId ?? null);
  
  const [files, setFiles] = useState<string[]>([]);
  const [selectedCsvFile, setSelectedCsvFile] = useState<string | null>(null);
  const [csvData, setCsvData] = useState<any | null>(null);
  const [refreshCounter, setRefreshCounter] = useState(0);

  // Gallery state
  const [isGridView, setIsGridView] = useState(false);
  const [currentImageIndex, setCurrentImageIndex] = useState(0);

  const fetchExperiments = () => {
    fetch(`http://127.0.0.1:8000/api/results/${outputDir}`)
      .then(res => res.json())
      .then((data: ExperimentMeta[]) => {
        setExperiments(data);
        // Auto-select logic
        if (experimentId && data.some(e => e.experiment_id === experimentId)) {
          setSelectedExp(experimentId);
          const match = data.find(e => e.experiment_id === experimentId);
          if (match?.pipeline_id) setSelectedPipelineId(match.pipeline_id);
        } else if (data.length > 0 && !selectedExp) {
          // Select most recent if none selected
          const first = data[0];
          setSelectedExp(first.experiment_id);
          if (first.pipeline_id) setSelectedPipelineId(first.pipeline_id);
        }
      })
      .catch(console.error);
  };

  useEffect(() => {
    if (outputDir) {
      setSelectedExp(null);
      setSelectedPipelineId(null);
      setFiles([]);
      fetchExperiments();
    }
  }, [outputDir]);

  // When experimentId prop changes (fresh run just finished), update selection
  useEffect(() => {
    if (experimentId) {
      setSelectedExp(experimentId);
      fetchExperiments();
    }
  }, [experimentId]);

  useEffect(() => {
    if (!selectedExp || !selectedPipelineId || !outputDir) return;
    fetch(`http://127.0.0.1:8000/api/results/${outputDir}/${selectedPipelineId}/${selectedExp}`)
      .then(res => res.json())
      .then(data => {
        setFiles(data.files ?? []);
        setCurrentImageIndex(0);
        setSelectedCsvFile(null);
      })
      .catch(console.error);
  }, [selectedExp, selectedPipelineId, outputDir, refreshCounter]);

  useEffect(() => {
    if (!selectedCsvFile || !selectedExp || !selectedPipelineId || !outputDir) {
      setCsvData(null);
      return;
    }
    fetch(`http://127.0.0.1:8000/api/results/${outputDir}/${selectedPipelineId}/${selectedExp}/${selectedCsvFile}/view`)
      .then(res => res.json())
      .then(setCsvData)
      .catch(console.error);
  }, [selectedCsvFile, selectedExp, selectedPipelineId, outputDir, refreshCounter]);

  if (!outputDir) return null;

  // Derive available pipelines and experiments
  const pipelines = Array.from(new Set(experiments.map(e => e.pipeline_id).filter(Boolean))) as string[];
  const expsForPipeline = experiments.filter(e => e.pipeline_id === selectedPipelineId);
  const activeExpMeta = experiments.find(e => e.experiment_id === selectedExp && e.pipeline_id === selectedPipelineId);

  // Separate files
  const images = files.filter(f => f.endsWith('.png') || f.endsWith('.jpg') || f.endsWith('.svg') || f.endsWith('.jpeg'));
  const csvs = files.filter(f => f.endsWith('.csv'));

  const getFileUrl = (filename: string) => `http://127.0.0.1:8000/api/results/${outputDir}/${selectedPipelineId}/${selectedExp}/${filename}/view`;

  return (
    <div className="card glass" style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>
      {/* Header & Controls */}
      <div style={{ display: 'flex', flexWrap: 'wrap', gap: '1rem', justifyContent: 'space-between', alignItems: 'center' }}>
        <h2 style={{ fontSize: '1.25rem', color: 'var(--accent-primary)', margin: 0, display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
          📊 Results <span style={{ fontSize: '0.9rem', color: 'var(--text-tertiary)', fontWeight: 400 }}>/ {outputDir}</span>
        </h2>
        <button
          className="btn btn-secondary"
          onClick={() => {
            fetchExperiments();
            setRefreshCounter(c => c + 1);
          }}
          style={{ fontSize: '0.75rem', padding: '0.35rem 0.75rem' }}
        >
          ↻ Refresh
        </button>
      </div>

      {experiments.length === 0 ? (
        <div style={{ color: 'var(--text-tertiary)', fontStyle: 'italic', fontSize: '0.9rem' }}>
          No runs found in <code>{outputDir}</code> yet.
        </div>
      ) : (
        <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
          {/* Dropdowns */}
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: '1rem', background: 'var(--bg-secondary)', padding: '1rem', borderRadius: 'var(--radius-md)', border: '1px solid var(--border-light)' }}>
            <div className="form-group" style={{ margin: 0, flex: 1, minWidth: '200px' }}>
              <label className="form-label" style={{ fontSize: '0.75rem' }}>Pipeline</label>
              <select 
                className="form-select" 
                value={selectedPipelineId || ''} 
                onChange={e => {
                  setSelectedPipelineId(e.target.value);
                  const firstExp = experiments.find(x => x.pipeline_id === e.target.value);
                  if (firstExp) setSelectedExp(firstExp.experiment_id);
                }}
                style={{ fontFamily: 'monospace', fontSize: '0.85rem' }}
              >
                {!selectedPipelineId && <option value="">Select Pipeline...</option>}
                {pipelines.map(p => <option key={p} value={p}>{p}</option>)}
              </select>
            </div>
            
            <div className="form-group" style={{ margin: 0, flex: 1, minWidth: '200px' }}>
              <label className="form-label" style={{ fontSize: '0.75rem' }}>Experiment Run</label>
              <select 
                className="form-select" 
                value={selectedExp || ''} 
                onChange={e => setSelectedExp(e.target.value)}
                style={{ fontFamily: 'monospace', fontSize: '0.85rem' }}
              >
                {!selectedExp && <option value="">Select Experiment...</option>}
                {expsForPipeline.map(exp => (
                  <option key={exp.experiment_id} value={exp.experiment_id}>
                    {exp.experiment_id} {exp.exit_code === 0 ? '✓' : exp.exit_code !== undefined ? '✗' : ''}
                  </option>
                ))}
              </select>
            </div>
          </div>

          {/* Parameters Toggle */}
          {activeExpMeta?.params && Object.keys(activeExpMeta.params).length > 0 && (
            <details style={{ background: 'var(--bg-primary)', border: '1px solid var(--border-light)', borderRadius: 'var(--radius-sm)', padding: '0.5rem 1rem' }}>
              <summary style={{ cursor: 'pointer', fontSize: '0.8rem', color: 'var(--text-secondary)', fontWeight: 600 }}>
                View Parameters
              </summary>
              <div style={{ marginTop: '0.75rem', display: 'flex', flexWrap: 'wrap', gap: '0.5rem' }}>
                {Object.entries(activeExpMeta.params).map(([k, v]) => (
                  <div key={k} style={{ background: 'var(--bg-secondary)', padding: '0.25rem 0.5rem', borderRadius: '4px', fontSize: '0.75rem', fontFamily: 'monospace', border: '1px solid var(--border-strong)' }}>
                    <span style={{ color: 'var(--text-tertiary)' }}>{k}:</span> <span style={{ color: 'var(--accent-primary)' }}>{JSON.stringify(v)}</span>
                  </div>
                ))}
              </div>
            </details>
          )}

          {/* Results Area */}
          {selectedExp && files.length === 0 ? (
            <div style={{ padding: '2rem', textAlign: 'center', color: 'var(--text-tertiary)', background: 'var(--bg-primary)', borderRadius: 'var(--radius-md)' }}>
              No files generated in this experiment.
            </div>
          ) : (
            <div style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem', marginTop: '0.5rem' }}>
              
              {/* Image Gallery */}
              {images.length > 0 && (
                <div style={{ border: '1px solid var(--border-strong)', borderRadius: 'var(--radius-sm)', background: 'var(--bg-secondary)', overflow: 'hidden' }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', padding: '0.5rem 1rem', background: 'var(--bg-tertiary)', borderBottom: '1px solid var(--border-strong)' }}>
                    <span style={{ fontSize: '0.85rem', color: 'var(--text-secondary)', fontFamily: 'monospace' }}>
                      {isGridView ? `${images.length} images` : images[currentImageIndex]}
                    </span>
                    <div style={{ display: 'flex', gap: '0.5rem' }}>
                      <button className="btn btn-secondary" style={{ padding: '0.25rem 0.5rem', fontSize: '0.75rem' }} onClick={() => setIsGridView(!isGridView)}>
                        {isGridView ? '🖼️ Single View' : '🔲 Grid View'}
                      </button>
                      {!isGridView && (
                        <a
                          href={`http://127.0.0.1:8000/api/results/${outputDir}/${selectedPipelineId}/${selectedExp}/${images[currentImageIndex]}/download`}
                          download
                          className="btn btn-primary"
                          style={{ padding: '0.25rem 0.5rem', fontSize: '0.75rem' }}
                        >
                          ⬇ Download
                        </a>
                      )}
                    </div>
                  </div>

                  <div style={{ padding: '1rem', minHeight: '300px', display: 'flex', justifyContent: 'center', alignItems: 'center', position: 'relative', background: '#fff' }}>
                    {isGridView ? (
                      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(200px, 1fr))', gap: '1rem', width: '100%' }}>
                        {images.map((img, i) => (
                          <div key={img} style={{ cursor: 'pointer', border: '1px solid var(--border-strong)', borderRadius: 'var(--radius-sm)', overflow: 'hidden', background: 'var(--bg-primary)' }} onClick={() => { setCurrentImageIndex(i); setIsGridView(false); }}>
                            <img src={getFileUrl(img)} alt={img} style={{ width: '100%', height: 'auto', display: 'block' }} loading="lazy" />
                            <div style={{ padding: '0.4rem', fontSize: '0.7rem', background: 'var(--bg-tertiary)', color: 'var(--text-secondary)', textAlign: 'center', fontFamily: 'monospace', textOverflow: 'ellipsis', overflow: 'hidden', whiteSpace: 'nowrap', borderTop: '1px solid var(--border-light)' }}>{img}</div>
                          </div>
                        ))}
                      </div>
                    ) : (
                      <>
                        <img src={getFileUrl(images[currentImageIndex])} alt={images[currentImageIndex]} style={{ maxWidth: '100%', maxHeight: '600px', borderRadius: 'var(--radius-sm)' }} />
                        {images.length > 1 && (
                          <>
                            <button 
                              onClick={() => setCurrentImageIndex(prev => (prev === 0 ? images.length - 1 : prev - 1))}
                              style={{ position: 'absolute', left: '1rem', top: '50%', transform: 'translateY(-50%)', background: 'rgba(255,255,255,0.8)', color: 'var(--text-primary)', border: '1px solid var(--border-strong)', borderRadius: '50%', width: '40px', height: '40px', display: 'flex', alignItems: 'center', justifyContent: 'center', cursor: 'pointer', fontSize: '1.2rem', boxShadow: 'var(--shadow-md)' }}
                            >
                              ◀
                            </button>
                            <button 
                              onClick={() => setCurrentImageIndex(prev => (prev === images.length - 1 ? 0 : prev + 1))}
                              style={{ position: 'absolute', right: '1rem', top: '50%', transform: 'translateY(-50%)', background: 'rgba(255,255,255,0.8)', color: 'var(--text-primary)', border: '1px solid var(--border-strong)', borderRadius: '50%', width: '40px', height: '40px', display: 'flex', alignItems: 'center', justifyContent: 'center', cursor: 'pointer', fontSize: '1.2rem', boxShadow: 'var(--shadow-md)' }}
                            >
                              ▶
                            </button>
                          </>
                        )}
                        {/* Gallery Indicators */}
                        {images.length > 1 && (
                          <div style={{ position: 'absolute', bottom: '1rem', display: 'flex', gap: '0.4rem', background: 'rgba(255,255,255,0.8)', padding: '0.3rem 0.6rem', borderRadius: '999px', border: '1px solid var(--border-strong)' }}>
                            {images.map((_, i) => (
                              <div key={i} style={{ width: '6px', height: '6px', borderRadius: '50%', background: i === currentImageIndex ? 'var(--accent-primary)' : 'var(--border-strong)' }} />
                            ))}
                          </div>
                        )}
                      </>
                    )}
                  </div>
                </div>
              )}

              {/* CSV Tables */}
              {csvs.length > 0 && (
                <details style={{ background: 'var(--bg-primary)', border: '1px solid var(--border-strong)', borderRadius: 'var(--radius-sm)' }} open={images.length === 0}>
                  <summary style={{ padding: '1rem', cursor: 'pointer', fontWeight: 600, color: 'var(--text-primary)', borderBottom: '1px solid transparent', display: 'flex', alignItems: 'center', gap: '0.5rem', background: 'var(--bg-secondary)' }}>
                    <span>📁 Data Tables ({csvs.length})</span>
                  </summary>
                  <div style={{ padding: '1rem', borderTop: '1px solid var(--border-strong)' }}>
                    <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.5rem', marginBottom: '1rem' }}>
                      {csvs.map(file => (
                        <button
                          key={file}
                          onClick={() => setSelectedCsvFile(file)}
                          style={{
                            padding: '0.375rem 0.75rem',
                            borderRadius: 'var(--radius-sm)',
                            border: selectedCsvFile === file ? '1px solid var(--accent-primary)' : '1px solid var(--border-strong)',
                            background: selectedCsvFile === file ? 'var(--bg-highlight)' : 'var(--bg-secondary)',
                            color: selectedCsvFile === file ? 'var(--accent-primary)' : 'var(--text-secondary)',
                            fontSize: '0.8rem',
                            fontWeight: selectedCsvFile === file ? 600 : 400
                          }}
                        >
                          📊 {file}
                        </button>
                      ))}
                    </div>

                    {selectedCsvFile && (
                      <div style={{ border: '1px solid var(--border-strong)', borderRadius: 'var(--radius-sm)', overflow: 'hidden' }}>
                        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', padding: '0.5rem 1rem', background: 'var(--bg-secondary)', borderBottom: '1px solid var(--border-light)' }}>
                          <span style={{ fontSize: '0.85rem', color: 'var(--text-secondary)', fontFamily: 'monospace' }}>{selectedCsvFile}</span>
                          <a
                            href={`http://127.0.0.1:8000/api/results/${outputDir}/${selectedPipelineId}/${selectedExp}/${selectedCsvFile}/download`}
                            download
                            className="btn btn-secondary"
                            style={{ padding: '0.2rem 0.5rem', fontSize: '0.7rem' }}
                          >
                            ⬇ Download CSV
                          </a>
                        </div>
                        <div style={{ padding: '0', maxHeight: '400px', overflow: 'auto' }}>
                          {csvData?.columns ? (
                            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '0.82rem', textAlign: 'left' }}>
                              <thead>
                                <tr>
                                  {csvData.columns.map((col: string) => (
                                    <th key={col} style={{ padding: '0.5rem 0.75rem', fontWeight: 600, color: 'var(--text-primary)', background: 'var(--bg-tertiary)', position: 'sticky', top: 0, borderBottom: '2px solid var(--border-strong)' }}>
                                      {col}
                                    </th>
                                  ))}
                                </tr>
                              </thead>
                              <tbody>
                                {csvData.data.map((row: any, i: number) => (
                                  <tr key={i} style={{ borderBottom: '1px solid var(--border-light)' }}>
                                    {csvData.columns.map((col: string) => (
                                      <td key={col} style={{ padding: '0.5rem 0.75rem', whiteSpace: 'nowrap' }}>
                                        {row[col] !== null ? String(row[col]) : ''}
                                      </td>
                                    ))}
                                  </tr>
                                ))}
                              </tbody>
                            </table>
                          ) : (
                            <div style={{ padding: '1rem', color: 'var(--text-tertiary)', textAlign: 'center' }}>Loading table data...</div>
                          )}
                        </div>
                      </div>
                    )}
                  </div>
                </details>
              )}

            </div>
          )}
        </div>
      )}
    </div>
  );
}
