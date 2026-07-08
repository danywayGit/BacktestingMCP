"""
LLM-driven config generation — reads evolution stats and uses an LLM
to auto-generate improved ScoringConfig parameters.

Inspired by AlgoEvolve's meta-evolution loop: the LLM analyses
under/over-performing configs and suggests targeted parameter changes.
"""

import json
import logging
import os
import re
from typing import Any, Dict, List, Optional
from datetime import datetime

logger = logging.getLogger(__name__)

# ScoringConfig parameter schema — tells the LLM what it can tune
CONFIG_SCHEMA = {
    "min_abs_score": {"type": "float", "range": [5.0, 9.0], "default": 7.0},
    "min_adx": {"type": "float", "range": [15, 30], "default": 20},
    "min_rsi": {"type": "float", "range": [20, 40], "default": 30},
    "max_rsi": {"type": "float", "range": [60, 80], "default": 70},
    "min_atr_pct": {"type": "float", "range": [0.0, 1.0], "default": 0.3},
    "atr_stop_mult": {"type": "float", "range": [0.5, 4.0], "default": 1.5},
    "rr_ratio": {"type": "float", "range": [1.0, 4.0], "default": 2.0},
    "trend_weight": {"type": "float", "range": [0.0, 0.6], "default": 0.4},
    "volume_relative_weight": {"type": "float", "range": [0.0, 0.4], "default": 0.2},
    "signal_feed_weight": {"type": "float", "range": [0.0, 0.4], "default": 0.3},
    "onchain_netflow_weight": {"type": "float", "range": [0.0, 0.3], "default": 0.1},
    "volume_divergence_weight": {"type": "float", "range": [0.0, 5.0], "default": 3.0},
    "smart_money_index_weight": {"type": "float", "range": [0.0, 4.0], "default": 2.0},
    "low_float_squeeze_weight": {"type": "float", "range": [0.0, 3.0], "default": 1.5},
    "regime_dir_bear_short_bonus": {"type": "float", "range": [1.0, 4.0], "default": 2.0},
    "regime_dir_bear_long_penalty": {"type": "float", "range": [1.0, 4.0], "default": 2.0},
    "regime_dir_bull_long_bonus": {"type": "float", "range": [1.0, 4.0], "default": 2.0},
    "regime_dir_bull_short_penalty": {"type": "float", "range": [1.0, 4.0], "default": 2.0},
}


def build_evolution_context(stats: Dict[str, Any]) -> str:
    """Build a readable summary of all configs' performance for the LLM."""
    lines = ["## Current Config Performance", ""]
    # Convert ConfigStats objects to dict access
    sorted_versions = sorted(stats.keys(), key=lambda v: stats[v].win_rate if hasattr(stats[v], 'win_rate') else 0, reverse=True)
    for ver in sorted_versions:
        s = stats[ver]
        wr = s.win_rate if hasattr(s, 'win_rate') else getattr(s, 'get', lambda k, d=0: d)("win_rate", 0)
        pf = s.profit_factor if hasattr(s, 'profit_factor') else 0
        flats = s.flat_rate if hasattr(s, 'flat_rate') else 0
        n = s.non_flat_trades if hasattr(s, 'non_flat_trades') else 0
        avg_r = s.avg_return_pct if hasattr(s, 'avg_return_pct') else 0
        lines.append(f"| {ver} | {wr:.1f}% | {pf:.2f} | {flats:.1f}% | {n} | {avg_r:+.2f}% |")
    if len(lines) > 1:
        lines.insert(1, "| Config | WR | PF | Flat% | Trades | Avg Ret |")
        lines.insert(2, "|--------|----|----|-------|--------|---------|")
    return "\n".join(lines)


def build_prompt(stats: Dict[str, Any], active_version: str) -> str:
    """Build the LLM prompt for config generation."""
    context = build_evolution_context(stats)
    schema_str = json.dumps(CONFIG_SCHEMA, indent=2)
    active = stats.get(active_version)
    active_wr = active.win_rate if active and hasattr(active, 'win_rate') else 0
    active_flat = active.flat_rate if active and hasattr(active, 'flat_rate') else 0

    prompt = f"""You are a quantitative trading strategist optimizing a scoring config for a crypto edge scanner.

## Current State
Active config: **{active_version}** (WR={active_wr:.1f}%, Flat={active_flat:.1f}%)

{context}

## Config Parameter Schema
{schema_str}

## Objective
Create ONE new config ("V{active_version.replace('.', '_')}") that improves upon the current active config ({active_version}).

### Strategy Guidelines
- If flat rate > 60%: relax filters (lower min_abs_score, min_adx, min_atr_pct)
- If WR < 30%: tighten filters (raise min_abs_score, min_adx, narrow RSI range)
- If low trade count: relax filters to let more signals through
- Weights must sum to 1.0 (trend + volume + signal_feed + onchain)
- If a particular weight parameter is 0.0, consider activating it if it could help
- Keep ALL parameters within their schema ranges

### Output Format
Return ONLY a valid JSON object with these exact keys:
{json.dumps({k: v["default"] for k, v in CONFIG_SCHEMA.items()}, indent=2)}

No explanation, no markdown — just the JSON object.
"""
    return prompt


def parse_llm_response(response: str) -> Optional[Dict[str, Any]]:
    """Extract JSON config from LLM response, handling markdown code blocks."""
    # Try to find a JSON block
    json_match = re.search(r"```(?:json)?\s*\n?({.*?})\s*\n?```", response, re.DOTALL)
    if json_match:
        response = json_match.group(1)
    else:
        # Try raw JSON parse
        json_match = re.search(r"({.*})", response, re.DOTALL)
        if json_match:
            response = json_match.group(1)

    try:
        config = json.loads(response)
    except json.JSONDecodeError:
        logger.error("Could not parse LLM response as JSON: %s", response[:200])
        return None

    # Validate all required keys exist and are in range
    for key, schema in CONFIG_SCHEMA.items():
        if key not in config:
            logger.warning("Missing key %s in LLM config, using default", key)
            config[key] = schema["default"]
        else:
            val = config[key]
            rmin, rmax = schema["range"]
            if val < rmin:
                logger.warning("Clamping %s from %.2f to %.2f", key, val, rmin)
                config[key] = rmin
            elif val > rmax:
                logger.warning("Clamping %s from %.2f to %.2f", key, val, rmax)
                config[key] = rmax

    # Ensure weights sum to 1.0
    weight_keys = ["trend_weight", "volume_relative_weight", "signal_feed_weight", "onchain_netflow_weight"]
    weight_sum = sum(config.get(k, 0) for k in weight_keys)
    if abs(weight_sum - 1.0) > 0.01:
        # Normalize
        for k in weight_keys:
            config[k] = config.get(k, 0) / weight_sum if weight_sum > 0 else 0.25
        logger.info("Normalized weights to sum to 1.0 (was %.2f)", weight_sum)

    return config


def _get_api_key() -> Optional[str]:
    """Get OpenRouter API key — tries multiple sources in order."""
    # 1. Try Hermes credential store
    hermes_env = "/home/hermes/.hermes/.env"
    if os.path.exists(hermes_env):
        try:
            with open(hermes_env) as f:
                for line in f:
                    line = line.strip()
                    if line.startswith("OPENROUTER_API_KEY=") or line.startswith("export OPENROUTER_API_KEY="):
                        val = line.split("=", 1)[1].strip('"').strip("'")
                        if val:
                            return val
        except (OSError, IOError):
            pass
    # 2. Try environment variable
    key = os.environ.get("OPENROUTER_API_KEY")
    if key and len(key) > 10:
        return key
    # 3. Try BacktestingMCP .env
    btmcp_env = os.path.join(os.path.dirname(__file__), "..", "..", ".env")
    if os.path.exists(btmcp_env):
        try:
            with open(btmcp_env) as f:
                for line in f:
                    line = line.strip()
                    if line.startswith("OPENROUTER_API_KEY=") or line.startswith("export OPENROUTER_API_KEY="):
                        val = line.split("=", 1)[1].strip('"').strip("'")
                        if val:
                            return val
        except (OSError, IOError):
            pass
    return None


def generate_new_config(stats: Dict[str, Any], active_version: str,
                        provider: str = "openrouter", model: str = "deepseek/deepseek-v4-flash") -> Optional[Dict[str, Any]]:
    """Use an LLM to generate a new scoring config based on evolution stats.

    Args:
        stats: Dict of {config_version: ConfigStats} from evolution engine
        active_version: Current active config version string (e.g. "7.0")
        provider: API provider name
        model: Model name

    Returns:
        Dict of validated config parameters, or None on failure
    """
    prompt = build_prompt(stats, active_version)

    try:
        import httpx
        api_key = _get_api_key()
        if not api_key:
            logger.error("OPENROUTER_API_KEY not set")
            return None

        resp = httpx.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.7,
                "max_tokens": 2048,
            },
            timeout=30,
        )
        resp.raise_for_status()
        content = resp.json()["choices"][0]["message"]["content"]
    except Exception as e:
        logger.error("LLM call failed: %s", e)
        return None

    config = parse_llm_response(content)
    if config:
        logger.info("LLM generated new config: %s", json.dumps(config, indent=2))
    return config


def register_config_as_code(config: Dict[str, Any], version_str: str = "7.3") -> bool:
    """Write a new config constant into scoring_config.py and register in ALL_CONFIGS.

    Args:
        config: Dict of config parameters
        version_str: Version string, e.g. "7.3" → CONFIG_V7_3

    Returns:
        True if successfully written and registered.
    """
    const_name = f"CONFIG_V{version_str.replace('.', '_')}"
    filepath = "/home/hermes/BacktestingMCP/src/edge_scanner/scoring_config.py"

    # Read existing file
    try:
        with open(filepath, "r") as f:
            content = f.read()
    except FileNotFoundError:
        logger.error("scoring_config.py not found at %s", filepath)
        return False

    # Check if this config already exists
    if const_name in content:
        logger.warning("Config %s already exists — not overwriting", const_name)
        return False

    # Build the config block — inject version and description explicitly
    block_lines = [
        f"\n\n# ── {const_name} — Auto-generated {datetime.now().strftime('%Y-%m-%d %H:%M')} ──",
        f"{const_name} = ScoringConfig(",
        f'    version="{version_str}",',
        f'    description="LLM-evolved: relaxed ADX/ATR/RSI filters, enabled multi-factor weights, boosted trend weight to 0.5",',
    ]
    for key, val in config.items():
        if key in ("version", "description"):
            continue  # injected above
        if isinstance(val, float):
            block_lines.append(f"    {key}={val},")
        else:
            block_lines.append(f"    {key}={val},")
    block_lines.append(")")

    # Find insertion point: before ACTIVE_CONFIG assignment
    active_line = content.find("\nACTIVE_CONFIG = ")
    if active_line == -1:
        active_line = content.find("ACTIVE_CONFIG = ")
    if active_line == -1:
        # Fallback: append at end
        content += "\n" + "\n".join(block_lines)
    else:
        content = content[:active_line] + "\n" + "\n".join(block_lines) + "\n" + content[active_line:]

    # Also register in ALL_CONFIGS if it exists — append to the list before the closing ]
    all_configs_line = content.find("ALL_CONFIGS: dict[str, ScoringConfig] = {")
    if all_configs_line == -1:
        all_configs_line = content.find("ALL_CONFIGS = {")
    if all_configs_line != -1:
        # Find the closing bracket of the innermost list inside the dict value
        # Pattern: {c.version: c for c in [..., CONFIG_V8_0, ]}
        prefix = f"{const_name},\n"
        # Simple approach: find the first standalone `]` after the dict body opener
        body_start = content.find("{", all_configs_line)
        if body_start != -1:
            # Find the LAST `]` before the closing `}`
            # Crawl forward to find the closing braces structure
            inner_list_end = content.rfind("]", body_start, content.rfind("}"))
            if inner_list_end != -1:
                content = content[:inner_list_end] + "\n        " + prefix + content[inner_list_end:]

    try:
        with open(filepath, "w") as f:
            f.write(content)
        logger.info("Successfully registered %s in scoring_config.py", const_name)
        return True
    except Exception as e:
        logger.error("Failed to write scoring_config.py: %s", e)
        return False


def _resolve_active_version() -> str:
    """Read the actual active config version from scoring_config.py."""
    filepath = "/home/hermes/BacktestingMCP/src/edge_scanner/scoring_config.py"
    try:
        with open(filepath) as f:
            content = f.read()
        m = re.search(r'ACTIVE_CONFIG\s*=\s*"?(CONFIG_V?[\d_]+)"?', content)
        if m:
            const_name = m.group(1)
            # Convert CONFIG_V7_0 -> 7.0, CONFIG_V1_0 -> 1.0
            v = const_name.replace("CONFIG_V", "").replace("_", ".")
            logger.info("Resolved active version: %s (from %s)", v, const_name)
            return v
    except Exception as e:
        logger.warning("Could not read ACTIVE_CONFIG: %s", e)
    return "1.0"


def _next_available_version(base: str = "1.0") -> str:
    """Find the NEXT version after the highest existing in the same major family.

    For active config "7.0", if existing are 7.0, 7.2, 7.3, 7.4, 7.5,
    generates 7.6 (not 7.1). Always creates a NEW version, never re-uses gaps.
    """
    filepath = "/home/hermes/BacktestingMCP/src/edge_scanner/scoring_config.py"
    try:
        with open(filepath) as f:
            content = f.read()
        existing_consts = set(re.findall(r"CONFIG_V([\d_]+)", content))

        # Extract the major version from the base (e.g., "7.0" → major=7)
        parts = base.split(".")
        major = parts[0]

        # Find the maximum existing minor in this major family
        max_minor = -1
        for const in existing_consts:
            if const.startswith(f"{major}_"):
                try:
                    minor_part = const.split("_")[1]
                    minor_val = int(minor_part)
                    if minor_val > max_minor:
                        max_minor = minor_val
                except (IndexError, ValueError):
                    continue

        # Next version = major.max_minor+1
        next_minor = max_minor + 1
        dotted = f"{major}.{next_minor}"
        logger.info("Next available version: %s (max existing minor: %d)", dotted, max_minor)
        return dotted

    except Exception as e:
        logger.warning("Could not scan config versions: %s", e)
    import time
    return f"{major if 'major' in dir() else '1'}_{int(time.time())}"


def utils_ts():
    from datetime import datetime
    return datetime.now().strftime("%H%M%S")


def auto_evolve_with_llm(db_path: str = "data/crypto.db", dry_run: bool = True) -> Dict[str, Any]:
    """Full pipeline: analyze → rank → LLM suggests new config → register.

    This is the end-to-end equivalent of AlgoEvolve's outer loop.

    Args:
        db_path: Path to SQLite database
        dry_run: If True, only log what would be done

    Returns:
        Dict with keys: action (llm_generate|noop|error), new_config, reason
    """
    from .evolution import analyze_configs, rank_configs

    result = {"action": "noop", "new_config": None, "reason": ""}

    stats = analyze_configs(db_path)
    if not stats:
        result["reason"] = "No config stats available"
        return result

    ranked = rank_configs(stats)
    if not ranked:
        result["reason"] = "No configs have enough data for ranking"
        return result

    active_version = _resolve_active_version()
    new_version = _next_available_version(active_version)
    active = stats.get(active_version)

    # Check if active has enough data — but still generate even if not
    # (we can learn from all configs, not just the active one)
    if active and active.non_flat_trades < 20:
        logger.info("Active config %s only has %d trades — but generating anyway from all config data", active_version, active.non_flat_trades)

    logger.info("Calling LLM to generate improved config...")
    new_config = generate_new_config(stats, active_version)
    if not new_config:
        result["reason"] = "LLM failed to produce valid config"
        return result

    result["new_config"] = new_config

    if dry_run:
        logger.info("DRY RUN: would register config: %s", json.dumps(new_config))
        result["action"] = "llm_generate"
        result["reason"] = "DRY RUN — LLM generated config ready for review"
        return result

    success = register_config_as_code(new_config, new_version)
    if success:
        result["action"] = "promote"
        result["reason"] = f"LLM-generated config V{new_version.replace('.', '_')} registered"
    else:
        result["action"] = "error"
        result["reason"] = "Failed to register config"

    return result