import asyncio
import json
import os
import uuid
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional
from src.ui.backend.ws import manager

RESULTS_BASE = Path("src/workflow/results")
DATA_BASE = Path("src/workflow/data")


class RunnerService:
    def __init__(self):
        self.runs = {}

    def _make_pipeline_id(self) -> str:
        return f"pipeline_{datetime.now().strftime('%Y%m%d_%H%M')}"

    def _make_experiment_id(self) -> str:
        return f"exp_{datetime.now().strftime('%Y%m%d_%H%M')}"

    def build_cli_args(self, params: Dict[str, Any]) -> List[str]:
        args = []
        for k, v in params.items():
            if isinstance(v, list):
                if v:
                    args.append(f"--{k}")
                    args.extend(str(item) for item in v)
            elif isinstance(v, bool):
                # Boolean flags: only append the flag when True; omit entirely when False
                if v is True:
                    args.append(f"--{k}")
            elif v is not None and v != "":
                args.append(f"--{k}")
                args.append(str(v))
        return args

    def get_active_pipeline(self) -> Optional[Dict]:
        """Return the most recently created pipeline params."""
        pipeline_dirs = sorted(
            [d for d in DATA_BASE.iterdir() if d.is_dir() and d.name.startswith("pipeline_")],
            key=lambda d: d.name,
            reverse=True,
        )
        if not pipeline_dirs:
            return None
        params_file = pipeline_dirs[0] / "params.json"
        if not params_file.exists():
            return None
        try:
            with open(params_file) as f:
                return json.load(f)
        except Exception:
            return None

    def list_pipeline_experiments(self) -> List[Dict]:
        """List all pipeline runs sorted newest-first."""
        pipeline_dirs = sorted(
            [d for d in DATA_BASE.iterdir() if d.is_dir() and d.name.startswith("pipeline_")],
            key=lambda d: d.name,
            reverse=True,
        )
        results = []
        for pd_ in pipeline_dirs:
            params_file = pd_ / "params.json"
            if not params_file.exists():
                continue
            try:
                with open(params_file) as f:
                    results.append(json.load(f))
            except Exception:
                continue
        return results

    async def run_pipeline(self, steps: List[int], params: Dict[str, Any]) -> Dict[str, str]:
        pipeline_id = self._make_pipeline_id()
        run_id = str(uuid.uuid4())

        # Inject pipeline_id and created_at into params
        params["pipeline_id"] = pipeline_id
        params["created_at"] = datetime.now().isoformat()

        self.runs[run_id] = {
            "type": "pipeline",
            "steps": steps,
            "pipeline_id": pipeline_id,
            "status": "started",
        }

        # Persist params into a pipeline-specific subdirectory
        pipeline_dir = DATA_BASE / pipeline_id
        pipeline_dir.mkdir(parents=True, exist_ok=True)
        params_path = pipeline_dir / "params.json"
        with open(params_path, "w") as f:
            json.dump(params, f, indent=2)

        steps_str = ",".join(map(str, steps))
        cmd = [
            sys.executable, "-m", "src.workflow.workflow",
            "--params", str(params_path),
            "--steps", steps_str,
        ]
        asyncio.create_task(self._execute(cmd, run_id))
        return {"run_id": run_id, "pipeline_id": pipeline_id}

    async def run_script(
        self,
        module: str,
        script_key: str,
        output_dir: str,
        params: Dict[str, Any],
        pipeline_id: Optional[str] = None,
        exp_id: Optional[str] = None,       # upstream experiment to inherit
        depends_on: str = "",               # output_dir of upstream step
        input_arg: str = "",               # CLI flag for upstream path
    ) -> Dict[str, str]:
        # Resolve pipeline_id — fall back to active pipeline
        if not pipeline_id:
            active = self.get_active_pipeline()
            pipeline_id = active["pipeline_id"] if active else "unknown"

        # Determine this run's experiment_id
        # Root scripts: generate fresh id. Downstream scripts: inherit from upstream.
        current_exp_id = exp_id if exp_id else self._make_experiment_id()

        run_id = str(uuid.uuid4())

        # Scoped output directory: output_dir / pipeline_id / exp_id
        scoped_output = RESULTS_BASE / output_dir / pipeline_id / current_exp_id
        scoped_output.mkdir(parents=True, exist_ok=True)

        self.runs[run_id] = {
            "type": "script",
            "module": module,
            "experiment_id": current_exp_id,
            "pipeline_id": pipeline_id,
            "status": "started",
        }

        cmd = [sys.executable, "-m", module]
        cmd.extend(self.build_cli_args(params))
        cmd.extend(["--output_dir", str(scoped_output)])

        # Pass the pipeline-specific params.json so scripts know where to find their sampled data
        if pipeline_id != "unknown":
            pipeline_params_path = Path("src/workflow/data") / pipeline_id / "params.json"
            cmd.extend(["--params", str(pipeline_params_path)])

        # Inject upstream path if this script depends on a previous step's results
        if exp_id and depends_on and input_arg:
            upstream_dir = RESULTS_BASE / depends_on / pipeline_id / exp_id
            if input_arg == "--input":
                # Hawkes step2: needs exact CSV file path, not directory
                count_type = params.get("count_type", "checkins")
                upstream_path = upstream_dir / f"hawkes_params_{count_type}.csv"
            else:
                upstream_path = upstream_dir
            cmd.extend([input_arg, str(upstream_path)])

        run_meta = {
            "experiment_id": current_exp_id,
            "pipeline_id": pipeline_id,
            "script": script_key,
            "params": params,
            "created_at": datetime.now().isoformat(),
            "output_dir": output_dir,
        }
        if exp_id and depends_on:
            run_meta["upstream_experiment_id"] = exp_id
            run_meta["depends_on"] = depends_on

        asyncio.create_task(
            self._execute(cmd, run_id, scoped_output=scoped_output, run_meta=run_meta)
        )
        return {
            "run_id": run_id,
            "experiment_id": current_exp_id,
            "output_subdir": str(scoped_output),
        }


    async def _execute(
        self,
        cmd: List[str],
        run_id: str,
        scoped_output: Optional[Path] = None,
        run_meta: Optional[Dict] = None,
    ):
        import subprocess
        import threading

        await manager.send_json({"type": "stdout", "line": f"$ {' '.join(cmd)}"}, run_id)

        loop = asyncio.get_event_loop()
        queue: asyncio.Queue = asyncio.Queue()

        def _run_in_thread():
            try:
                env = {**os.environ, "PYTHONUNBUFFERED": "1", "PYTHONIOENCODING": "utf-8"}
                proc = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    cwd="d:/dequi/repositories/osltm",
                    env=env,
                    bufsize=1,  # line-buffered
                )

                def reader(stream, stream_type):
                    for raw_line in iter(stream.readline, b""):
                        decoded = raw_line.decode("utf-8", errors="replace").rstrip()
                        loop.call_soon_threadsafe(queue.put_nowait, (stream_type, decoded))
                    stream.close()

                t_out = threading.Thread(target=reader, args=(proc.stdout, "stdout"), daemon=True)
                t_err = threading.Thread(target=reader, args=(proc.stderr, "stderr"), daemon=True)
                t_out.start()
                t_err.start()
                t_out.join()
                t_err.join()

                exit_code = proc.wait()
                loop.call_soon_threadsafe(queue.put_nowait, ("_done", exit_code))
            except Exception as e:
                loop.call_soon_threadsafe(queue.put_nowait, ("_error", str(e)))

        threading.Thread(target=_run_in_thread, daemon=True).start()

        # Consume messages from the thread
        try:
            while True:
                msg_type, payload = await queue.get()
                if msg_type == "_done":
                    exit_code = payload
                    self.runs[run_id]["status"] = "completed" if exit_code == 0 else "failed"

                    if scoped_output and run_meta:
                        run_meta["exit_code"] = exit_code
                        run_meta["completed_at"] = datetime.now().isoformat()
                        with open(scoped_output / "run_meta.json", "w") as f:
                            json.dump(run_meta, f, indent=2)

                    final_status = "completed" if exit_code == 0 else "failed"
                    await manager.send_json(
                        {"type": "status", "status": final_status, "exit_code": exit_code}, run_id
                    )
                    asyncio.create_task(self._delayed_clear(run_id))
                    break
                elif msg_type == "_error":
                    raise RuntimeError(payload)
                else:
                    await manager.send_json({"type": msg_type, "line": payload}, run_id)
        except Exception as e:
            self.runs[run_id]["status"] = "error"
            import traceback
            tb = traceback.format_exc()
            err_msg = f"{type(e).__name__}: {e}" if str(e) else f"{type(e).__name__}: {repr(e)}"
            await manager.send_json({"type": "stderr", "line": f"❌ Runner error: {err_msg}"}, run_id)
            await manager.send_json({"type": "stderr", "line": tb}, run_id)
            await manager.send_json(
                {"type": "status", "status": "error", "exit_code": -1, "message": err_msg}, run_id
            )
            asyncio.create_task(self._delayed_clear(run_id))

    async def _delayed_clear(self, run_id: str, delay: int = 30):
        """Clear the WS message buffer after a delay so late clients can still replay logs."""
        await asyncio.sleep(delay)
        manager.clear_buffer(run_id)


runner = RunnerService()
