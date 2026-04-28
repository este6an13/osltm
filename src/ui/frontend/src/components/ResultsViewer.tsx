'use client';

import { useState, useEffect } from 'react';

interface ExperimentMeta {
  experiment_id: string;
  pipeline_id?: string;
  script?: string;
  created_at?: string;
  exit_code?: number;
}

interface ResultsViewerProps {
  outputDir: string;
  experimentId?: string;   // auto-select after a fresh run
  pipelineId?: string;     // highlight experiments from this pipeline
}

export default function ResultsViewer({ outputDir, experimentId, pipelineId }: ResultsViewerProps) {
  const [experiments, setExperiments] = useState<ExperimentMeta[]>([]);
  const [selectedExp, setSelectedExp] = useState<string | null>(experimentId ?? null);
  const [selectedPipelineId, setSelectedPipelineId] = useState<string | null>(null);
  const [files, setFiles] = useState<string[]>([]);
  const [selectedFile, setSelectedFile] = useState<string | null>(null);
  const [csvData, setCsvData] = useState<any | null>(null);
  const [refreshCounter, setRefreshCounter] = useState(0);

  const fetchExperiments = () => {
    fetch(`http://127.0.0.1:8000/api/results/${outputDir}`)
      .then(res => res.json())
      .then((data: ExperimentMeta[]) => {
        setExperiments(data);
        // Auto-select if experimentId was provided and it's in the list
        if (experimentId && data.some(e => e.experiment_id === experimentId)) {
          setSelectedExp(experimentId);
          const match = data.find(e => e.experiment_id === experimentId);
          if (match?.pipeline_id) setSelectedPipelineId(match.pipeline_id);
        }
      })
      .catch(console.error);
  };

  useEffect(() => {
    if (outputDir) fetchExperiments();
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
    // Don't auto-deselect files on refresh, just update the list
    fetch(`http://127.0.0.1:8000/api/results/${outputDir}/${selectedPipelineId}/${selectedExp}`)
      .then(res => res.json())
      .then(data => setFiles(data.files ?? []))
      .catch(console.error);
  }, [selectedExp, selectedPipelineId, outputDir, refreshCounter]);

  useEffect(() => {
    if (!selectedFile || !selectedExp || !selectedPipelineId || !outputDir) return;
    if (selectedFile.endsWith('.csv')) {
      fetch(`http://127.0.0.1:8000/api/results/${outputDir}/${selectedPipelineId}/${selectedExp}/${selectedFile}/view`)
        .then(res => res.json())
        .then(setCsvData)
        .catch(console.error);
    } else {
      setCsvData(null);
    }
  }, [selectedFile, selectedExp, selectedPipelineId, outputDir, refreshCounter]);

  if (!outputDir) return null;

  // Group experiments: matching pipeline first, then "other"
  const matchingExps = pipelineId ? experiments.filter(e => e.pipeline_id === pipelineId) : experiments;
  const otherExps = pipelineId ? experiments.filter(e => e.pipeline_id !== pipelineId) : [];

  const fileUrl = selectedExp && selectedPipelineId && selectedFile
    ? `http://127.0.0.1:8000/api/results/${outputDir}/${selectedPipelineId}/${selectedExp}/${selectedFile}/view`
    : '';

  return (
    <div className="card glass" style={{ display: 'flex', flexDirection: 'column', gap: '1rem', marginTop: '1rem' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <h2 style={{ fontSize: '1.25rem', color: 'var(--accent-primary)' }}>
          Results — <span style={{ fontFamily: 'monospace', fontSize: '1rem' }}>{outputDir}</span>
        </h2>
        <button
          className="btn btn-secondary"
          onClick={() => {
            fetchExperiments();
            setRefreshCounter(c => c + 1);
          }}
          style={{ fontSize: '0.75rem' }}
        >
          ↻ Refresh
        </button>
      </div>

      {experiments.length === 0 ? (
        <div style={{ color: 'var(--text-tertiary)', fontStyle: 'italic', fontSize: '0.9rem' }}>
          No runs found in <code>{outputDir}</code> yet. Run the script above first.
        </div>
      ) : (
        <>
          {/* Experiment Selector */}
          <div>
            <p style={{ fontSize: '0.75rem', color: 'var(--text-tertiary)', marginBottom: '0.5rem', textTransform: 'uppercase', letterSpacing: '0.05em' }}>
              Select experiment run
            </p>
            <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.5rem' }}>
              {matchingExps.map(exp => (
                <button
                  key={exp.experiment_id}
                  onClick={() => { setSelectedExp(exp.experiment_id); setSelectedPipelineId(exp.pipeline_id ?? null); }}
                  style={{
                    padding: '0.375rem 0.75rem',
                    borderRadius: 'var(--radius-sm)',
                    border: selectedExp === exp.experiment_id ? '1px solid var(--accent-primary)' : '1px solid var(--border-light)',
                    background: selectedExp === exp.experiment_id ? 'rgba(59,130,246,0.1)' : 'var(--bg-primary)',
                    color: selectedExp === exp.experiment_id ? 'var(--accent-primary)' : 'var(--text-secondary)',
                    fontSize: '0.8rem',
                    fontFamily: 'monospace',
                  }}
                  title={`Pipeline: ${exp.pipeline_id ?? 'unknown'} · ${exp.created_at ?? ''}`}
                >
                  {exp.experiment_id}
                  {exp.exit_code === 0 && <span style={{ marginLeft: '0.5rem', color: 'var(--accent-success)' }}>✓</span>}
                </button>
              ))}
              {otherExps.length > 0 && (
                <details style={{ width: '100%' }}>
                  <summary style={{ fontSize: '0.75rem', color: 'var(--text-tertiary)', cursor: 'pointer' }}>
                    {otherExps.length} run(s) from other pipelines
                  </summary>
                  <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.5rem', marginTop: '0.5rem' }}>
                    {otherExps.map(exp => (
                      <button
                        key={exp.experiment_id}
                        onClick={() => { setSelectedExp(exp.experiment_id); setSelectedPipelineId(exp.pipeline_id ?? null); }}
                        style={{
                          padding: '0.375rem 0.75rem',
                          borderRadius: 'var(--radius-sm)',
                          border: '1px solid var(--border-light)',
                          background: 'transparent',
                          color: 'var(--text-tertiary)',
                          fontSize: '0.8rem',
                          fontFamily: 'monospace',
                          opacity: 0.7,
                        }}
                      >
                        {exp.experiment_id}
                      </button>
                    ))}
                  </div>
                </details>
              )}
            </div>
          </div>

          {/* Files for selected experiment */}
          {selectedExp && files.length > 0 && (
            <div>
              <p style={{ fontSize: '0.75rem', color: 'var(--text-tertiary)', marginBottom: '0.5rem', textTransform: 'uppercase', letterSpacing: '0.05em' }}>
                Files in {selectedExp}
              </p>
              <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.5rem' }}>
                {files.map(file => (
                  <button
                    key={file}
                    onClick={() => {
                      setSelectedFile(file);
                      setCsvData(null);
                    }}
                    style={{
                      padding: '0.375rem 0.75rem',
                      borderRadius: 'var(--radius-sm)',
                      border: selectedFile === file ? '1px solid var(--accent-secondary)' : '1px solid var(--border-light)',
                      background: selectedFile === file ? 'rgba(139,92,246,0.1)' : 'var(--bg-primary)',
                      color: selectedFile === file ? 'var(--accent-secondary)' : 'var(--text-secondary)',
                      fontSize: '0.8rem',
                    }}
                  >
                    {file.endsWith('.csv') ? '📊 ' : '🖼️ '}{file}
                  </button>
                ))}
              </div>
            </div>
          )}
          {selectedExp && files.length === 0 && (
            <div style={{ color: 'var(--text-tertiary)', fontSize: '0.85rem' }}>No files found in this experiment run.</div>
          )}
        </>
      )}

      {/* File Preview */}
      {selectedFile && selectedExp && (
    <div style={{ background: 'var(--bg-primary)', borderRadius: 'var(--radius-md)', border: '1px solid var(--border-strong)', overflow: 'hidden' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', padding: '0.75rem 1rem', borderBottom: '1px solid var(--border-light)' }}>
        <span style={{ fontSize: '0.85rem', color: 'var(--text-secondary)', fontFamily: 'monospace' }}>{selectedFile}</span>
        <a
          href={`http://127.0.0.1:8000/api/results/${outputDir}/${selectedPipelineId}/${selectedExp}/${selectedFile}/download`}
          download
          className="btn btn-primary"
          style={{ padding: '0.25rem 0.5rem', fontSize: '0.75rem' }}
        >
          Download
        </a>
      </div>
      <div style={{ padding: '1rem', maxHeight: '500px', overflow: 'auto', display: 'flex', justifyContent: 'center' }}>
        {selectedFile.endsWith('.png') && (
          <img src={fileUrl} alt={selectedFile} style={{ maxWidth: '100%', borderRadius: 'var(--radius-sm)' }} />
        )}
        {selectedFile.endsWith('.csv') && csvData?.columns && (
          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '0.82rem', textAlign: 'left' }}>
            <thead>
              <tr>
                {csvData.columns.map((col: string) => (
                  <th key={col} style={{ padding: '0.4rem 0.6rem', fontWeight: 600, color: 'var(--text-secondary)', background: 'var(--bg-secondary)', position: 'sticky', top: 0, borderBottom: '1px solid var(--border-strong)' }}>
                    {col}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {csvData.data.map((row: any, i: number) => (
                <tr key={i} style={{ borderBottom: '1px solid var(--border-light)' }}>
                  {csvData.columns.map((col: string) => (
                    <td key={col} style={{ padding: '0.4rem 0.6rem', whiteSpace: 'nowrap' }}>
                      {row[col] !== null ? String(row[col]) : ''}
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        )}
        {selectedFile.endsWith('.csv') && !csvData?.columns && (
          <div style={{ color: 'var(--text-tertiary)' }}>Loading...</div>
        )}
      </div>
    </div>
      )}
    </div>
  );
}
