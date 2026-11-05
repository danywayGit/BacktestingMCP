# GPU MCP Server — Start Here (Windows 4090 box)

The ONLY place you need. Full background is in `docs/GPU_GUIDE.md`; you don't need it to start the server.

## Start the server — Windows 11 (RTX 4090)

Open **Command Prompt**, from your home directory (NOT the repo folder):

```bat
cd C:\Users\danyw
set PYTHONPATH=
python C:\Users\danyw\Documents\Git\DanywayGit\BacktestingMCP\src\mcp\gpu_server.py
```

Expected output (keep this window open):
```
INFO: Uvicorn running on http://0.0.0.0:8765
```

That's it. Hermes connects to `192.168.1.7:8765`.

## Why this exact command (so it works every time)

- Run from a directory OUTSIDE the repo and clear `PYTHONPATH`, or Python imports the
  repo's own `src/mcp/server.py` instead of the installed `mcp` library and dies with
  `ImportError: cannot import name 'Server'`. cd out + `set PYTHONPATH=` avoids it.
- Serves Streamable HTTP MCP at `http://0.0.0.0:8765/mcp`.

## Once it's running, optionally verify from Hermes

```bash
hermes mcp test gpu4090   # expect ✓ Connected + 5 tools
```

## Troubleshooting

- **Connection dropped** → the window closed or the PC slept. Re-run the start command.
- **`ImportError: cannot import name 'Server'`** → you're in the repo dir or PYTHONPATH is set. cd out + `set PYTHONPATH=` first.
- **`FATAL: FastMCP import failed`** → one-time deps missing. Run once:
  ```bat
  pip install "fastmcp" "mcp[server]"
  ```
- **Computation still runs if GPU asleep** — a CPU fallback (same math) kicks in; results are identical.