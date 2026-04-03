import asyncio
import uuid
import sys
from typing import List, Dict, Any
from src.ui.backend.ws import manager

class RunnerService:
    def __init__(self):
        # Tracking runs by id for history
        self.runs = {}

    def build_cli_args(self, params: Dict[str, Any]) -> List[str]:
        args = []
        for k, v in params.items():
            if isinstance(v, bool):
                if v:
                    # Note: We replace checking for True/False and converting back to python flags 
                    # If arg is required as --flag we do --flag (Python argparse `action='store_true'`)
                    # We should match the snake_case keys from the API to kebab-case or what script needs
                    pass # We do it per param below
            if k == "stations" and isinstance(v, list):
                if v:
                    args.append("--stations")
                    args.extend(v)
            elif k == "no_standardize" and v is True:
                args.append("--no_standardize")
            elif k == "no_normalize" and v is True:
                args.append("--no-normalize")
            elif k == "no_plot" and v is True:
                args.append("--no_plot")
            elif isinstance(v, bool) and v is True:
                # generically convert `--flag`
                args.append(f"--{k.replace('_', '-')}")
            elif v is not None and v != "":
                # Convert underscore to dash, argparse actually depends on how the script was written!
                # Most scripts use underscores in argparse if we look at playbook
                args.append(f"--{k.replace('_', '-')}")
                args.append(str(v))
        return args

    async def run_pipeline(self, steps: List[int]) -> str:
        run_id = str(uuid.uuid4())
        steps_str = ",".join(map(str, steps))
        self.runs[run_id] = {"type": "pipeline", "steps": steps, "status": "started"}

        cmd = [sys.executable, "-m", "src.workflow.workflow", "--params", "src/workflow/params.json", "--steps", steps_str]
        asyncio.create_task(self._execute(cmd, run_id))
        return run_id

    async def run_script(self, module: str, params: Dict[str, Any]) -> str:
        run_id = str(uuid.uuid4())
        self.runs[run_id] = {"type": "script", "module": module, "status": "started"}

        cmd = [sys.executable, "-m", module]
        cmd.extend(self.build_cli_args(params))
        asyncio.create_task(self._execute(cmd, run_id))
        return run_id

    async def _execute(self, cmd: List[str], run_id: str):
        # Send initial cmd
        await manager.send_json({"type": "stdout", "line": f"$ {' '.join(cmd)}"}, run_id)

        try:
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd="d:/dequi/repositories/osltm" # execute in project root
            )

            async def read_stream(stream, stream_type):
                while True:
                    line = await stream.readline()
                    if not line:
                        break
                    decoded_line = line.decode('utf-8', errors='replace').rstrip()
                    await manager.send_json({"type": stream_type, "line": decoded_line}, run_id)

            await asyncio.gather(
                read_stream(process.stdout, "stdout"),
                read_stream(process.stderr, "stderr")
            )

            exit_code = await process.wait()
            self.runs[run_id]["status"] = "completed" if exit_code == 0 else "failed"
            await manager.send_json({"type": "status", "status": "completed", "exit_code": exit_code}, run_id)
        except Exception as e:
            self.runs[run_id]["status"] = "failed"
            await manager.send_json({"type": "status", "status": "error", "message": str(e)}, run_id)

runner = RunnerService()
