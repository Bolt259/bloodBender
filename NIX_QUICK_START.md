# Nix Quick Start Guide for bloodBender

This guide will help you get started with the Nix-based development environment for bloodBender.

---

## 🚀 Installation

### 1. Install Nix (if not already installed)

#### Linux / macOS

```bash
# Multi-user installation (recommended)
sh <(curl -L https://nixos.org/nix/install) --daemon

# Or single-user installation
sh <(curl -L https://nixos.org/nix/install) --no-daemon
```

#### Enable Flakes

Add to `~/.config/nix/nix.conf` (create if doesn't exist):

```
experimental-features = nix-command flakes
```

Or set environment variable:

```bash
export NIX_CONFIG="experimental-features = nix-command flakes"
```

### 2. Install direnv (Optional but Recommended)

```bash
# On NixOS
nix-env -iA nixpkgs.direnv

# On other systems with Nix
nix profile install nixpkgs#direnv

# Or with your system package manager
# apt install direnv       # Debian/Ubuntu
# brew install direnv      # macOS
```

Configure your shell:

```bash
# For bash, add to ~/.bashrc:
eval "$(direnv hook bash)"

# For zsh, add to ~/.zshrc:
eval "$(direnv hook zsh)"
```

---

## 🎯 Usage

### Method 1: direnv (Automatic, Recommended)

```bash
# Navigate to project directory
cd /path/to/bloodBender

# Allow direnv (first time only)
direnv allow

# Environment automatically activates when you cd into directory!
```

### Method 2: Manual Nix Development Shell

```bash
# Enter default development shell (full environment)
nix develop

# Or specify a specific shell
nix develop .#python      # Python-only, no CUDA
nix develop .#cpp         # C++ only for bareMetalBender
```

### Method 3: One-off Commands

```bash
# Run a command in the Nix environment without entering shell
nix develop --command python -m bloodBath status
nix develop --command make -C bareMetalBender
```

---

## 🧪 Verify Installation

```bash
# Enter Nix shell
nix develop

# Check Python and packages
python --version          # Should be Python 3.10.x
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

# Check bloodBath CLI
python -m bloodBath status

# Check C++ build environment
cd bareMetalBender
make clean && make
./ivp
```

---

## 📦 Available Environments

### Default (Full)

```bash
nix develop
```

- Python 3.10 with all dependencies
- PyTorch with CUDA support
- C++ build tools
- Development utilities

### Python-only (CPU)

```bash
nix develop .#python
```

- Python environment without CUDA
- Faster to build, smaller download
- Good for data processing tasks

### C++-only

```bash
nix develop .#cpp
```

- C++ compiler and build tools
- For bareMetalBender development

---

## 🔧 Common Tasks

### Sync Pump Data

```bash
nix develop --command python -m bloodBath sync \
  --pump-serial YOUR_SERIAL \
  --start-date 2024-01-01
```

### Train LSTM Model

```bash
nix develop --command python bloodTwin/pipelines/train_lstm.py
```

### Build C++ Solver

```bash
nix develop .#cpp --command make -C bareMetalBender
```

### Run Tests

```bash
nix develop --command python -m pytest bloodBath/test_scripts/
```

---

## 🛠️ Configuration

### Environment Variables

Create a `.env` file in the project root:

```bash
cp .env.example .env
# Edit .env with your credentials
```

The Nix shell will automatically load this file.

### Python Path

The Nix shell automatically sets `PYTHONPATH` to include the project root, so you can import modules directly:

```python
from bloodBath import TandemHistoricalSyncClient
from bloodTwin.models.lstm import BloodGlucoseLSTM
```

---

## 🐛 Troubleshooting

### "experimental-features" Error

**Problem:** `error: experimental Nix feature 'nix-command' is disabled`

**Solution:** Enable flakes in your Nix configuration:

```bash
mkdir -p ~/.config/nix
echo "experimental-features = nix-command flakes" >> ~/.config/nix/nix.conf
```

### CUDA Not Available

**Problem:** `torch.cuda.is_available()` returns `False`

**Possible causes:**

1. No NVIDIA GPU in system → Use `nix develop .#python` for CPU-only
2. NVIDIA drivers not installed → Install system drivers
3. Driver/CUDA version mismatch → Check compatibility

### Slow First Build

**Problem:** First `nix develop` takes a long time

**Explanation:** Nix is downloading and building all dependencies. This only happens once! Subsequent runs use cached builds.

**Tip:** Use binary caches:

```bash
# Add to ~/.config/nix/nix.conf
substituters = https://cache.nixos.org https://cuda-maintainers.cachix.org
trusted-public-keys = cache.nixos.org-1:6NCHdD59X431o0gWypbMrAURkbJ16ZPMQFGspcDShjY= cuda-maintainers.cachix.org-1:0dq3bujKpuEPMCX6U4WylrUDZ9JyUG0VpVZa7CNfq5E=
```

### Module Not Found Errors

**Problem:** `ModuleNotFoundError: No module named 'bloodBath'`

**Solution:** Make sure you're in a Nix shell and PYTHONPATH is set:

```bash
echo $PYTHONPATH  # Should include project directory
export PYTHONPATH=".:$PYTHONPATH"
```

---

## 📚 Next Steps

- Read the main [README.md](README.md) for project overview
- Check [NIX_MIGRATION_PLAN.md](NIX_MIGRATION_PLAN.md) for migration details
- See [bloodBath/README.md](bloodBath/README.md) for data sync guide
- See [bloodTwin/README.md](bloodTwin/README.md) for ML training guide

---

## 🔄 Updating Dependencies

### Update all packages to latest compatible versions:

```bash
nix flake update
```

### Update specific input:

```bash
nix flake lock --update-input nixpkgs
```

### Pin to specific nixpkgs commit:

Edit `flake.nix`:

```nix
inputs.nixpkgs.url = "github:NixOS/nixpkgs/COMMIT_HASH";
```

---

## 🎓 Learn More

- [Nix Flakes Guide](https://nixos.wiki/wiki/Flakes)
- [Nixpkgs Python Documentation](https://nixos.org/manual/nixpkgs/stable/#python)
- [direnv Documentation](https://direnv.net/)

---

**Happy Hacking! 🩸**
