import type { Metadata } from "next";
import "./globals.css";
import Link from 'next/link';

export const metadata: Metadata = {
  title: "OSLTM Workflow UI",
  description: "Web interface for OSLTM stochastic modeling pipeline",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body>
        <div className="app-container">
          <aside className="sidebar">
            <div className="sidebar-header">
              <h2>OSLTM</h2>
              <p>Workflow Engine</p>
            </div>
            
            <nav className="sidebar-nav">
              <Link href="/" className="nav-link">Dashboard</Link>
              <div className="nav-section">Pipeline</div>
              <Link href="/pipeline" className="nav-link">Data Pipeline</Link>
              
              <div className="nav-section">Analysis</div>
              <Link href="/analysis/network" className="nav-link">Network</Link>
              <Link href="/analysis/profiles" className="nav-link">Profiles</Link>
              <Link href="/analysis/intensity" className="nav-link">Intensity</Link>
              
              <div className="nav-section">Models</div>
              <Link href="/models/hawkes" className="nav-link">Hawkes Process</Link>
              <Link href="/models/lgcp" className="nav-link">LGCP Pipeline</Link>
              <Link href="/models/avg_profile" className="nav-link">Average Profile</Link>
              <Link href="/models/cluster" className="nav-link">Cluster Process</Link>

              <div className="nav-section">Real-Time</div>
              <Link href="/realtime" className="nav-link">⏱ Live Simulation</Link>
            </nav>
          </aside>
          
          <main className="main-content flex-1">
            {children}
          </main>
        </div>
      </body>
    </html>
  );
}
