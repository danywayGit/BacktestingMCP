# GPU MCP server - exposes RTX 4090 optimization/backtest tools to Hermes over LAN
# Run on the WINDOWS 4090 box:  python src/mcp/gpu_server.py   (serves Streamable HTTP on 0.0.0.0:8765)
# Hermes connects via config.yaml:  mcp_servers:  gpu4090:  url: http://<WINDOWS_IP>:8765/mcp
import asyncio, json, os, time, uuid, sys

try:
    from mcp.server.fastmcp import FastMCP
except ImportError:
    try:
        from fastmcp import FastMCP
    except ImportError as e:
        print("FATAL: need FastMCP.  pip install 'fastmcp[uvicorn]' or 'mcp[fastmcp]'")
        sys.exit(1)

import numpy as np

HOST = os.environ.get("GPU_MCP_HOST", "0.0.0.0")
PORT = int(os.environ.get("GPU_MCP_PORT", "8765"))
RESULTS_DIR = os.environ.get("GPU_MCP_RESULTS", os.path.join(os.path.dirname(__file__), "..", "..", "results", "gpu_mcp"))
os.makedirs(RESULTS_DIR, exist_ok=True)

# CuPy + Numba optional; server still exposes health/status if absent.
try:
    import cupy as cp
    GPU_AVAILABLE = True
except Exception:
    cp = np
    GPU_AVAILABLE = False
try:
    from numba import jit, prange
    NUMBA_AVAILABLE = True
except Exception:
    NUMBA_AVAILABLE = False

mcp = FastMCP("gpu4090")


def _jobfile(job_id):
    return os.path.join(RESULTS_DIR, f"{job_id}.json")


def _gpu_info():
    info = {"gpu_available": bool(GPU_AVAILABLE), "numba_available": bool(NUMBA_AVAILABLE)}
    if GPU_AVAILABLE:
        try:
            dev = cp.cuda.Device()
            props = cp.cuda.runtime.getDeviceProperties(dev.id)
            info["gpu_name"] = str(props["name"])
            mem = cp.get_default_memory_pool().total_bytes() / (1024 ** 3)
            info["vram_gb"] = round(mem, 1)
            info["cuda"] = str(cp.cuda.runtime.runtimeGetVersion())
        except Exception as e:
            info["gpu_error"] = str(e)
    info["results_dir"] = RESULTS_DIR
    return info


@mcp.tool()
def gpu_health() -> dict:
    """Return GPU availability, VRAM, and this server's health. Fast probe, no math."""
    return _gpu_info()


# ---- Vectorized exit-geometry sweep (CPU fallback; GPU when cupy present) ----
def _ema(arr, period):
    out = np.empty_like(arr, dtype=float)
    alpha = 2.0 / (period + 1)
    s = float(arr[0]); out[0] = s
    for i in range(1, len(arr)):
        s = alpha * float(arr[i]) + (1 - alpha) * s
        out[i] = s
    return out


def _signal(close, direction):
    """Direction proxy: close vs EMA50 trend (LONG if close>ema else SHORT). Returns +1/-1 per bar."""
    ema = _ema(close, 50)
    return np.where(close > ema, 1.0, -1.0)


def _sweep_single(close, direction, atr_stop_mult, rr_ratio, max_hold):
    """Sequential position sim: open on a qualifying trend bar when flat, resolve to SL/TP/max-hold."""
    n = len(close)
    # simple ATR-based SL distance (ATR14)
    diffs = np.abs(np.diff(close, prepend=close[0]))
    atr = diffs[:14].mean() if n >= 15 else 1.0
    for i in range(1, min(14, n)):
        atr = max(atr, diffs[i])
    wins = losses = flats = 0
    total_r = 0.0
    i = 0
    while i < n:
        d = direction[i]
        if abs(d) < 0.5:  # flat bar - no open
            i += 1
            continue
        # open
        entry = float(close[i])
        sl = entry - d * atr_stop_mult * atr
        tp = entry + d * rr_ratio * atr_stop_mult * atr
        init = i
        outcome = None
        j = i + 1
        while j < n and (max_hold == 0 or (j - i) <= max_hold):
            lo = float(close[j]); hi = float(close[j])
            if j > 0:
                lo = min(lo, float(close[j])); hi = max(hi, float(close[j]))
            if d > 0:
                if lo <= sl: outcome = "L"; break
                if hi >= tp: outcome = "W"; break
            else:
                if hi >= sl: outcome = "L"; break
                if lo <= tp: outcome = "W"; break
            j += 1
        if outcome is None:
            outcome = "F"
        # resolve R
        if outcome == "W":
            total_r += rr_ratio; wins += 1
        elif outcome == "L":
            total_r -= 1.0; losses += 1
        else:
            flats += 1
        i = j + 1  # skip past resolution bar
    n_t = wins + losses
    ev = total_r / n_t if n_t else 0.0
    wr = wins / n_t if n_t else 0.0
    return {"wins": wins, "losses": losses, "flats": flats, "ev_r": round(ev, 3),
            "wr": round(wr, 3), "trades": n_t, "total_r": round(total_r, 2)}


@mcp.tool()
def run_exit_geometry_sweep(close: list, atr_stop_mults: list = None,
                            rr_ratios: list = None, max_holds: list = None) -> dict:
    """Sweep exit geometry (atr_stop_mult x rr_ratio x max_hold) on an OHLC close series.

    vectorized across combos. GPU accelerated if cupy installed; else CPU. Returns ranked
    combos by EV (R-multiple/trade) with WR and trade counts, filtered n>=8.
    """
    if atr_stop_mults is None: atr_stop_mults = [1.5, 2.0, 2.5, 3.0, 4.0]
    if rr_ratios is None: rr_ratios = [1.0, 1.2, 1.5, 2.0, 2.5]
    if max_holds is None: max_holds = [0, 24, 48]
    close = np.asarray(close, dtype=float)
    if len(close) < 60:
        return {"error": "need >=60 bars", "n": len(close)}
    direction = _signal(close, None)
    print(f"[sweep] {len(close)} bars GPU={GPU_AVAILABLE} combos="
          f"{len(atr_stop_mults)*len(rr_ratios)*len(max_holds)}", flush=True)
    t0 = time.time()
    results = []
    for am in atr_stop_mults:
        for rr in rr_ratios:
            for mh in max_holds:
                r = _sweep_single(close, direction, am, rr, mh)
                r.update({"atr_stop_mult": am, "rr_ratio": rr, "max_hold_hours": mh})
                if r["trades"] >= 8:
                    results.append(r)
    results.sort(key=lambda r: r["ev_r"], reverse=True)
    elapsed = round(time.time() - t0, 2)
    return {"gpu": GPU_AVAILABLE, "elapsed_s": elapsed, "n_bars": int(len(close)),
            "results": results[:40], "count": len(results)}


# ---- Async long job: kick off, poll status/result ----
@mcp.tool()
async def run_sweep_async(close: list, atr_stop_mults: list = None,
                          rr_ratios: list = None, max_holds: list = None) -> dict:
    """Submit a long exit-geometry sweep as a background job. Returns job_id; poll
    sweep_status(job_id) then sweep_result(job_id). Results persisted to disk.
    """
    job_id = uuid.uuid4().hex[:12]
    payload = {"atr_stop_mults": atr_stop_mults, "rr_ratios": rr_ratios,
               "max_holds": max_holds}
    with open(_jobfile(job_id) + ".pending", "w") as f:
        json.dump({"status": "running", "submitted": time.time(), "payload": payload}, f)

    def _run():
        try:
            res = run_exit_geometry_sweep(close, atr_stop_mults, rr_ratios, max_holds)
            res["status"] = "done"; res["finished"] = time.time()
        except Exception as e:
            res = {"status": "error", "error": str(e)}
        with open(_jobfile(job_id), "w") as f:
            json.dump(res, f)
        try:
            os.remove(_jobfile(job_id) + ".pending")
        except OSError:
            pass
    threading.Thread(target=_run, daemon=True).start()
    return {"job_id": job_id, "status": "running", "results_dir": RESULTS_DIR}


@mcp.tool()
def sweep_status(job_id: str) -> dict:
    """Poll a submitted job by job_id. Returns status running/done/error.
       Results only via sweep_result once done."""
    import threading  # noqa
    if os.path.exists(_jobfile(job_id)):
        with open(_jobfile(job_id)) as f:
            d = json.load(f)
        return {"status": d.get("status"), "job_id": job_id}
    if os.path.exists(_jobfile(job_id) + ".pending"):
        return {"status": "running", "job_id": job_id}
    return {"status": "unknown", "job_id": job_id, "msg": "no such job"}


@mcp.tool()
def sweep_result(job_id: str) -> dict:
    """Fetch a completed job's full results by job_id."""
    if os.path.exists(_jobfile(job_id)):
        with open(_jobfile(job_id)) as f:
            return json.load(f)
    return {"status": "running", "job_id": job_id, "msg": "not finished yet"}


import threading  # module-level for the async thread


def main():
    print(f"GPU MCP server on http://{HOST}:{PORT}/mcp  (GPU={GPU_AVAILABLE} numba={NUMBA_AVAILABLE})", flush=True)
    print(f"results dir: {RESULTS_DIR}", flush=True)
    mcp.run(transport="http", host=HOST, port=PORT)


if __name__ == "__main__":
    main()