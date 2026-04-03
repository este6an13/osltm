'use client';

import { useEffect, useState } from 'react';

export default function Dashboard() {
  const [status, setStatus] = useState<any>(null);

  useEffect(() => {
    fetch('http://127.0.0.1:8000/api/status')
      .then(res => res.json())
      .then(data => setStatus(data))
      .catch(err => console.error("Could not fetch status", err));
  }, []);

  return (
    <div className="animate-fade-in">
      <h1 className="page-title">Dashboard</h1>
      <p className="page-subtitle">OSLTM Stochastic Processing System Overview</p>
      
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))', gap: '1.5rem', marginBottom: '2rem' }}>
        <div className="card">
          <h3 className="form-label">Database Size</h3>
          <div style={{ fontSize: '2rem', fontWeight: 700, color: 'var(--text-primary)' }}>
            {status ? `${status.db_size_mb} MB` : '...'}
          </div>
        </div>
        
        <div className="card">
          <h3 className="form-label">Sampled Dates</h3>
          <div style={{ fontSize: '2rem', fontWeight: 700, color: 'var(--text-primary)' }}>
            {status ? status.sampled_dates : '...'}
          </div>
        </div>
        
        <div className="card">
          <h3 className="form-label">Sampled Stations</h3>
          <div style={{ fontSize: '2rem', fontWeight: 700, color: 'var(--text-primary)' }}>
            {status ? status.sampled_stations : '...'}
          </div>
        </div>
      </div>
      
      <div className="card glass">
        <h2 style={{ fontSize: '1.25rem', marginBottom: '1rem' }}>Quick Actions</h2>
        <div style={{ display: 'flex', gap: '1rem', flexWrap: 'wrap' }}>
          <button className="btn btn-primary" onClick={() => window.location.href='/pipeline'}>
            Go to Data Pipeline
          </button>
          <button className="btn btn-secondary" onClick={() => window.location.href='/analysis/profiles'}>
            Profile Analysis
          </button>
          <button className="btn btn-secondary" onClick={() => window.location.href='/analysis/intensity'}>
            Intensity Analysis
          </button>
          <button className="btn btn-secondary" onClick={() => window.location.href='/models/hawkes'}>
            Hawkes Modeling
          </button>
        </div>
      </div>
    </div>
  );
}
