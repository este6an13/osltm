# OSLTM Web Interface

The OSLTM Web UI provides a graphical dashboard and execution environment to visually run the stochastic point-process pipeline, browse configurations, and instantly see intensity, models, and profiles metrics.

The architecture is split into two parts:
1. **FastAPI Backend (`src/ui/backend`)**: Orchestrates the existing raw Python scripts using child process execution (`subprocess`). Streams stdout logs through WebSockets in real time.
2. **Next.js Frontend (`src/ui/frontend`)**: A React-based Single Page Application providing a fully dynamic Dark Mode UI.

---

## 🚀 How to Start the Web App

The application runs using two local development servers. For the best experience, open two separate terminals at the root of the repository.

### Terminal 1: Start the Backend (API & Subprocesses)
```bash
# Ensure you are at the repository root
# Launch the FastAPI orchestrator
uv run uvicorn src.ui.backend.main:app --reload
```
*The backend will be available at `http://localhost:8000`*

### Terminal 2: Start the Frontend (Next.js Dashboard)
```bash
# Navigate to the frontend directory
cd src/ui/frontend

# Install dependencies strictly the first time:
# npm install

# Launch the Next.js development server
npm run dev
```
*The frontend interface will become available at `http://localhost:3000`*

---

## 🧭 How to Use the UI

Navigate your browser to **`http://localhost:3000`** to access the dashboard.

### 1. Data Pipeline
Go to the **Data Pipeline** tab to execute the foundational 4-step pipeline that builds the SQLite DB, populates checkins/checkouts, and samples stations.
- You can tune parameters (like *Stations*, *Dates per Stratum*) directly before executing.

### 2. Analysis & Models
The UI automatically fetches the selected stations from `sampled_stations.csv` under the hood. You can configure individual analysis constraints and then run them live.
- Expand specific module pages (e.g., **Hawkes Process**, **LGCP Pipeline**, **Profiles**).
- Adjust the dynamically rendered UI toggles (`--date_percentage`, `--n-clusters`, etc). 
- Hit **Run**. A terminal block will show you Python logs in real-time.

### 3. Native Results 
When your script finishes running, you don't even need to leave the page. 
- Click the **"Refresh Results"** button located right underneath the execution terminal.
- It will automatically read the corresponding directory (e.g. `fpca_results`, `fano_factor_within_bins`) and instantly display any generated `.csv` arrays as interactive HTML tables, and `.png` metric plots visually inside your browser.
