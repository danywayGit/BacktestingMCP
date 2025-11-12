# Repository Cleanup Plan
**Date**: November 12, 2025

## 📊 Repository Analysis Summary

### Current Structure Issues Found:
1. ✅ **Empty folders**: `exports/`, `logs/`, `strategies/custom/`
2. ⚠️ **Duplicate/obsolete test files**: Multiple test scripts with unclear purpose
3. ⚠️ **Old analysis scripts**: `analyze_dca_frequency.py`, `analyze_dca_trades.py` (created during session)
4. ⚠️ **Redundant documentation**: Multiple summary/guide files
5. ⚠️ **Unused examples**: Some example files may be outdated

---

## 🗑️ Cleanup Actions

### A. EMPTY FOLDERS TO REMOVE
- [x] `exports/` - Empty, no purpose defined
- [x] `logs/` - Empty, logging not implemented
- [x] `strategies/custom/` - Empty subdirectory (keep parent `strategies/` folder structure)

### B. OBSOLETE TEST FILES
**Keep**:
- `test_backtest.py` - Active backtest testing
- `run_simple_backtest.py` - Working demo script

**Remove**:
- `test_simple.py` - Duplicate functionality
- `test_venv.py` - Empty file
- `test_ollama_strategy.py` - One-off test for AI generation

### C. ANALYSIS SCRIPTS (Session-Created)
**Remove** (created during debugging, not part of core system):
- `analyze_dca_frequency.py` - Buy frequency analysis (one-time use)
- `analyze_dca_trades.py` - Trade diagnosis script (debugging only)

### D. DOCUMENTATION CONSOLIDATION
**Keep** (Active documentation):
- `README.md` - Main project documentation
- `QUICKSTART.md` - Quick start guide
- `DCA_STRATEGIES_GUIDE.md` - DCA strategy reference
- `OLLAMA_RTX4090_SETUP.md` - AI setup guide
- `PARAMETER_FORMAT_GUIDE.md` - Parameter optimization guide
- `src/ai/README.md` - AI generation documentation

**Consider Consolidating**:
- `IMPLEMENTATION_SUMMARY.md` - Move to docs/ folder or merge into README
- `CLEANUP_SUMMARY.md` - Archive or move to docs/archive/

### E. REDUNDANT RUNNER SCRIPTS
**Keep**:
- `run_dca_comparison.py` - Active DCA comparison tool
- `download_dca_data.py` - Data download utility
- `demo_ai_strategies.py` - AI strategy demo
- `optimize_example.py` - Optimization example
- `optimize_simple.py` - Simplified optimization

**Review**:
- `run.py` - General launcher (may be redundant with CLI)
- `quickstart_dca.py` - Just prints instructions (could be in README)
- `examples.py` - Partially disabled examples

### F. OLD BACKTEST RESULTS
**Action**: Clean up old JSON files in `backtest_results/`
- Keep: Latest 2-3 results
- Remove: Older duplicates (9 old dca_comparison files)

---

## 📁 Proposed Final Structure

```
BacktestingMCP/
├── src/                             # Core source code ✅
│   ├── core/                        # Backtesting engine ✅
│   ├── data/                        # Data management ✅
│   ├── strategies/                  # Strategy templates ✅
│   │   └── generated/               # AI-generated strategies ✅
│   ├── risk/                        # Risk management ✅
│   ├── ai/                          # AI strategy generation ✅
│   ├── mcp/                         # MCP server ✅
│   └── cli/                         # CLI interface ✅
├── config/                          # Configuration ✅
├── data/                            # Data storage ✅
│   └── crypto.db                    # SQLite database ✅
├── backtest_results/                # Results (cleaned) 🧹
├── docs/                            # Documentation folder 📚 NEW
│   ├── archive/                     # Old docs/summaries 📦 NEW
│   └── guides/                      # User guides 📖 NEW
├── examples/                        # Working examples 📝 NEW
│   ├── run_simple_backtest.py      # Basic demo ✅
│   ├── demo_ai_strategies.py       # AI demo ✅
│   ├── optimize_example.py         # Optimization demo ✅
│   └── optimize_simple.py          # Simple optimization ✅
├── scripts/                         # Utility scripts 🔧 NEW
│   ├── download_dca_data.py        # Data downloader ✅
│   ├── run_dca_comparison.py       # DCA comparison ✅
│   ├── compare_dca_strategies.py   # Scaling approach ✅
│   ├── compare_both_approaches.py  # Compare results ✅
│   └── fractional_dca_comparison.py # Fractional approach ✅
├── .gitignore                       # Git ignore ✅
├── requirements.txt                 # Dependencies ✅
├── README.md                        # Main documentation ✅
├── QUICKSTART.md                    # Quick start ✅
└── venv/                            # Virtual environment ✅
```

---

## 🎯 Cleanup Benefits

1. **Clearer Organization**: Separate examples, scripts, and docs
2. **Easier Navigation**: Less clutter in root directory
3. **Better Discoverability**: Related files grouped together
4. **Reduced Confusion**: Remove obsolete/duplicate files
5. **Cleaner Git History**: Remove temporary debugging files

---

## ⚡ Execution Order

1. Remove empty folders
2. Remove obsolete test files
3. Remove session analysis scripts
4. Create new folder structure (docs/, examples/, scripts/)
5. Move files to appropriate locations
6. Clean up old backtest results
7. Update .gitignore if needed
8. Update README.md with new structure
9. Commit changes

---

## 📝 Notes

- All core functionality remains intact
- Only removing duplicates, empties, and debugging scripts
- Documentation is being organized, not deleted
- Working examples preserved and better organized
