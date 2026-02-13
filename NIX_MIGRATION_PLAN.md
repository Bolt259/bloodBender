# bloodBender Nix Migration Plan

**Branch:** `cleanup/deprecated-modules-and-nix-migration`  
**Date:** February 13, 2026  
**Status:** Planning & Execution

---

## 📋 Overview

This document outlines the comprehensive plan to:

1. Remove deprecated modules and legacy code
2. Migrate from Python venv (`bloodBath-env/`) to Nix-based dependency management
3. Set up a reproducible development environment with Nix flakes

---

## 🗑️ Phase 1: Deprecated Code Removal

### Folders to Remove

#### High Priority (Safe to Remove)

1. **`sweetBloodDeprecated/`**
   - **Status:** Fully deprecated, replaced by bloodBank v2.0
   - **Size:** ~10+ subdirectories
   - **Contains:** Old sweetBlood module structure
   - **Notes:** All functionality migrated to `bloodBath/bloodBank/`

2. **`bloodBath-env.bak/`**
   - **Status:** Backup of old virtual environment
   - **Size:** ~500MB+
   - **Contains:** Old Python packages
   - **Notes:** Superseded by current `bloodBath-env/`, will be replaced by Nix

3. **`training_data_legacy/`**
   - **Status:** Old training data format
   - **Size:** Unknown
   - **Contains:** Pre-v2.0 LSTM training data
   - **Notes:** Current data in `bloodBath/bloodBank/lstm_pump_data/`

4. **`test_fixed_v2/`**
   - **Status:** Old test output directory
   - **Size:** Small
   - **Contains:** Test artifacts from v2.0 development
   - **Notes:** Current tests in `bloodBath/test_scripts/`

5. **`test_logs/`**
   - **Status:** Historical test logs
   - **Notes:** Can be archived if needed before removal

6. **`test_monthly_validation/`**
   - **Status:** Old validation test artifacts
   - **Notes:** Current validation in `bloodBath/validation/`

7. **`test_results/`**
   - **Status:** Historical test results
   - **Notes:** Can be archived if needed

8. **`unified_lstm_training/`**
   - **Status:** Old training directory (replaced by bloodTwin)
   - **Notes:** Current training handled by `bloodTwin/pipelines/`

#### Root-Level Test Scripts (Consider Consolidation)

- `test_*.py` files in root (30+ files)
- **Action:** Review and either move to `bloodBath/test_scripts/` or remove if obsolete
- **Examples:**
  - `test_bloodbath.py`
  - `test_comprehensive.py`
  - `test_sweetblood.py` (likely deprecated)
  - `test_synthetic_100_system.py`
  - etc.

#### Log Files (Archive or Clean)

- `bloodbank_*.log` (40+ log files)
- `csv_repair_*.log`
- `csv_repair_summary_*.json`
- **Action:** Archive critical logs, remove routine operation logs

#### Deprecated Scripts

- `batch_csv_repair.py` (if replaced by new validation)
- `bloodBathMaker.py` (purpose unclear, likely old)
- `bloodBath_data_consolidator.py` (if replaced)
- `simple_bloodBath_consolidator.py` (if replaced)
- Various `*_incremental_resync.py` scripts (check if duplicates)

### Code Cleanup

#### Files with Deprecated Stubs

- **`bloodBath/cli/main.py`**: Contains deprecated sweetBlood compatibility stubs

  ```python
  # Lines 38-56: Remove deprecated stubs after confirming no usage
  def add_sweetblood_args(parser): ...
  def handle_sweetblood_commands(args, client): ...
  class SweetBloodIntegration: ...
  ```

- **`bloodBath/utils/structure_utils.py`**:
  - `setup_sweetblood_environment()` - marked as deprecated
  - `create_sweetblood_structure()` - check for usage

- **`bloodBath/data/processors.py`**:
  - Legacy `DataProcessor` class (lines 921+)
  - Wraps `UnifiedDataProcessor` for backward compatibility

#### Deprecation Strategy

1. ✅ Search codebase for usage of deprecated functions
2. ✅ Update callers to use new v2.0 APIs
3. ✅ Remove deprecated compatibility layers
4. ✅ Update documentation

---

## 🔧 Phase 2: Nix Flake Setup

### Current Python Environment Analysis

**Dependencies (from `bloodBath-env/`):**

- Core: `numpy==2.2.6`, `pandas==2.3.3`
- ML: `torch==2.7.1+cu118`, `pytorch-lightning==2.5.5`, `torchmetrics==1.8.2`
- ONNX: `onnx==1.19.1`, `onnxruntime-gpu==1.23.0`
- CUDA: Multiple `nvidia-*` packages (CUDA 11.8)
- Utils: `arrow`, `coloredlogs`, `humanfriendly`, `certifi`
- API: `bloodBath.api` (integrated tconnectsync functionality)

**Total:** 71 packages

### Nix Flake Architecture

```
flake.nix                    # Main flake definition
├── inputs
│   ├── nixpkgs              # Main package repository
│   ├── flake-utils          # Helper utilities
│   └── poetry2nix (optional) # Python dependency management
├── outputs
│   ├── devShells            # Development environments
│   │   ├── default          # Full dev environment
│   │   ├── python-only      # Python without CUDA
│   │   └── cuda             # Full GPU support
│   ├── packages             # Build outputs
│   │   ├── bloodBath        # Python package
│   │   ├── bloodTwin        # ML module
│   │   └── bareMetalBender  # C++ executable
│   └── apps                 # Runnable applications
```

### Nix Files Structure

```
bloodBender/
├── flake.nix                # Main flake
├── flake.lock               # Lock file (generated)
├── nix/                     # Nix configuration modules
│   ├── python-env.nix       # Python environment
│   ├── cuda-env.nix         # CUDA/GPU configuration
│   ├── cpp-env.nix          # C++ build environment
│   └── shell.nix            # Legacy nix-shell support
├── pyproject.toml           # Python project metadata
└── requirements/            # Split requirements
    ├── base.txt             # Core dependencies
    ├── ml.txt               # ML/PyTorch dependencies
    ├── dev.txt              # Development tools
    └── test.txt             # Testing dependencies
```

### Nix Flake Implementation

#### Step 1: Create `pyproject.toml`

Convert project to modern Python package with proper metadata.

#### Step 2: Create `flake.nix`

Define reproducible development environments:

- Python environment with all dependencies
- CUDA support for GPU acceleration
- C++ build tools for bareMetalBender
- Multiple shells for different use cases

#### Step 3: Pin Dependencies

Lock all package versions for reproducibility.

#### Step 4: Create Helper Scripts

- `nix/scripts/enter-env.sh` - Enter development shell
- `nix/scripts/build-cpp.sh` - Build C++ components
- `nix/scripts/run-training.sh` - Run ML training

#### Step 5: Documentation

- `NIX_QUICK_START.md` - Getting started guide
- `NIX_TROUBLESHOOTING.md` - Common issues
- Update main README with Nix instructions

---

## 🚀 Phase 3: Migration Execution

### Pre-Migration Checklist

- [x] Create git branch for changes
- [ ] Archive critical logs and data
- [ ] Document current environment state
- [ ] Test current functionality baseline
- [ ] Identify all import paths using deprecated code

### Migration Steps

#### 1. Deprecated Code Removal (Safe)

```bash
# Remove deprecated folders
git rm -rf sweetBloodDeprecated/
git rm -rf bloodBath-env.bak/
git rm -rf training_data_legacy/
git rm -rf test_fixed_v2/
git rm -rf test_logs/
git rm -rf test_monthly_validation/
git rm -rf test_results/
git rm -rf unified_lstm_training/

# Remove old log files (after archiving if needed)
git rm bloodbank_*.log
git rm csv_repair_*.log
git rm csv_repair_summary_*.json

# Commit removal
git commit -m "Remove deprecated modules and legacy code"
```

#### 2. Code Cleanup

```bash
# Remove deprecated function stubs
# Update imports and references
# Run tests to verify no breakage
git commit -m "Remove deprecated compatibility stubs"
```

#### 3. Nix Flake Creation

```bash
# Create flake structure
# Define dependencies
# Test development shell
git add flake.nix nix/
git commit -m "Add Nix flake for reproducible builds"
```

#### 4. Replace venv with Nix

```bash
# Update .gitignore
# Add bloodBath-env/ to ignore (will be removed after migration)
# Document Nix usage
git commit -m "Migrate from venv to Nix-based dependency management"
```

#### 5. Verify Everything Works

```bash
# Enter Nix shell
nix develop

# Run tests
python -m pytest bloodBath/test_scripts/

# Test CLI
python -m bloodBath status

# Test LSTM training
python bloodTwin/pipelines/train_lstm.py --config bloodTwin/configs/smoke_test.yaml

# Build C++ component
cd bareMetalBender && make clean && make
```

---

## 🎯 Success Criteria

### Code Cleanup

- ✅ All deprecated folders removed
- ✅ No broken import statements
- ✅ All tests pass
- ✅ CLI functionality intact
- ✅ Reduced repository size

### Nix Migration

- ✅ `nix develop` enters working environment
- ✅ All Python dependencies available
- ✅ CUDA/PyTorch GPU support works
- ✅ C++ compilation succeeds
- ✅ All existing scripts run without modification
- ✅ Development environment reproducible across machines

---

## 📊 Benefits

### After Cleanup

- **Smaller repository**: Remove ~500MB+ of deprecated code
- **Clearer structure**: No confusion about which modules to use
- **Easier maintenance**: Less code to maintain
- **Faster CI/CD**: Fewer files to process

### After Nix Migration

- **Reproducibility**: Exact same environment on any machine
- **Version control**: Dependencies locked in Git
- **No more "works on my machine"**: Hermetic builds
- **Easy onboarding**: New developers run `nix develop`
- **Multiple environments**: Easy switching between configs
- **No venv conflicts**: Isolated per-project dependencies

---

## 🚨 Risks & Mitigation

### Risks

1. **Breaking changes**: Removing code that's still used
   - **Mitigation**: Thorough grep search, test suite before merge
2. **Lost historical context**: Important logs deleted
   - **Mitigation**: Archive critical logs before deletion
3. **Nix learning curve**: Team unfamiliar with Nix
   - **Mitigation**: Comprehensive documentation, fallback scripts

4. **CUDA/GPU compatibility**: Nix CUDA support can be tricky
   - **Mitigation**: Test on actual hardware, provide CPU-only option

### Rollback Plan

```bash
# If anything breaks, rollback is simple:
git checkout main
git branch -D cleanup/deprecated-modules-and-nix-migration
```

---

## 📅 Timeline

- **Phase 1 (Cleanup)**: 2-4 hours
- **Phase 2 (Nix Setup)**: 4-6 hours
- **Phase 3 (Testing)**: 2-3 hours
- **Total**: ~1-2 days

---

## 🔗 References

- Nix Flakes: https://nixos.wiki/wiki/Flakes
- poetry2nix: https://github.com/nix-community/poetry2nix
- PyTorch on Nix: https://nixos.wiki/wiki/Python#PyTorch
- bloodBath v2.0 Spec: `bloodBath/spec/bloodBath_Design_Specification_v2.0.md`

---

**Next Steps:**

1. Review this plan with team
2. Archive any critical data
3. Execute Phase 1 (cleanup)
4. Execute Phase 2 (Nix setup)
5. Test thoroughly
6. Merge to main when validated
