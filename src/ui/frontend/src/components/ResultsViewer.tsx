'use client';

import { useState, useEffect } from 'react';

interface ResultsViewerProps {
  outputDir: string;
}

export default function ResultsViewer({ outputDir }: ResultsViewerProps) {
  const [files, setFiles] = useState<string[]>([]);
  const [selectedFile, setSelectedFile] = useState<string | null>(null);
  const [csvData, setCsvData] = useState<any | null>(null);

  const fetchFiles = () => {
    fetch(`http://127.0.0.1:8000/api/results/${outputDir}`)
      .then(res => res.json())
      .then(data => {
        if (Array.isArray(data)) {
          setFiles(data);
        }
      })
      .catch(console.error);
  };

  // Initially fetch, but user might re-run scripts so we provide a refresh button
  useEffect(() => {
    if (outputDir) {
      fetchFiles();
    }
  }, [outputDir]);

  useEffect(() => {
    if (!selectedFile || !outputDir) return;
    if (selectedFile.endsWith('.csv')) {
      fetch(`http://127.0.0.1:8000/api/results/${outputDir}/${selectedFile}/view`)
        .then(res => res.json())
        .then(setCsvData)
        .catch(console.error);
    } else {
      setCsvData(null);
    }
  }, [selectedFile, outputDir]);

  const fileUrl = outputDir && selectedFile 
    ? `http://127.0.0.1:8000/api/results/${outputDir}/${selectedFile}/view` 
    : '';

  if (!outputDir) return null;

  return (
    <div className="card glass" style={{ display: 'flex', flexDirection: 'column', gap: '1rem', marginTop: '1rem' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <h2 style={{ fontSize: '1.25rem', color: 'var(--accent-primary)' }}>Generated Results ({outputDir})</h2>
        <button className="btn btn-secondary" onClick={fetchFiles}>Refresh Results</button>
      </div>

      {files.length === 0 ? (
        <div style={{ color: 'var(--text-tertiary)', fontSize: '0.9rem', fontStyle: 'italic' }}>
          No files generated in {outputDir} yet. Run the script above first.
        </div>
      ) : (
        <div style={{ display: 'flex', gap: '1rem', flexWrap: 'wrap' }}>
          {files.map(file => (
            <button 
              key={file}
              onClick={() => setSelectedFile(file)}
              style={{
                padding: '0.5rem 0.75rem',
                borderRadius: 'var(--radius-sm)',
                backgroundColor: selectedFile === file ? 'var(--bg-highlight)' : 'var(--bg-primary)',
                color: selectedFile === file ? 'var(--accent-primary)' : 'var(--text-secondary)',
                border: selectedFile === file ? '1px solid var(--accent-primary)' : '1px solid var(--border-light)',
                fontSize: '0.85rem'
              }}
            >
              {file.endsWith('.csv') ? '📊' : '🖼️'} {file}
            </button>
          ))}
        </div>
      )}

      {selectedFile && (
        <div style={{ marginTop: '1rem', padding: '1rem', background: 'var(--bg-primary)', borderRadius: 'var(--radius-md)', border: '1px solid var(--border-strong)', overflow: 'auto' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1rem' }}>
            <h3 style={{ fontSize: '1rem', color: 'var(--text-secondary)' }}>PREVIEW: {selectedFile}</h3>
            <a href={`http://127.0.0.1:8000/api/results/${outputDir}/${selectedFile}/download`} download className="btn btn-primary" style={{ padding: '0.25rem 0.5rem', fontSize: '0.75rem' }}>
              Download File
            </a>
          </div>

          <div style={{ display: 'flex', justifyContent: 'center' }}>
            {selectedFile.endsWith('.png') && (
              <img src={fileUrl} alt={selectedFile} style={{ maxWidth: '100%', borderRadius: 'var(--radius-sm)' }} />
            )}
            
            {selectedFile.endsWith('.csv') && csvData && csvData.columns && (
              <div style={{ maxHeight: '400px', width: '100%', overflow: 'auto' }}>
                <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '0.85rem', textAlign: 'left' }}>
                  <thead>
                    <tr style={{ borderBottom: '1px solid var(--border-strong)' }}>
                      {csvData.columns.map((col: string) => (
                        <th key={col} style={{ padding: '0.5rem', fontWeight: 600, color: 'var(--text-secondary)', background: 'var(--bg-secondary)', position: 'sticky', top: 0 }}>{col}</th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {csvData.data.map((row: any, i: number) => (
                      <tr key={i} style={{ borderBottom: '1px solid var(--border-light)' }}>
                        {csvData.columns.map((col: string) => (
                          <td key={col} style={{ padding: '0.5rem', color: 'var(--text-primary)', whiteSpace: 'nowrap' }}>
                            {row[col] !== null ? String(row[col]) : ''}
                          </td>
                        ))}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
            {selectedFile.endsWith('.csv') && (!csvData || !csvData.columns) && (
              <div style={{ color: 'var(--text-tertiary)' }}>Loading CSV contents...</div>
            )}
          </div>
        </div>
      )}
    </div>
  );
}
