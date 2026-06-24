# Nix environment

bloodBender uses a Nix flake for a reproducible toolchain: Python with PyTorch/CUDA, and the C++ build environment for bareMetalBender.

## Setup

Install Nix with flakes enabled, then:

```bash
nix develop            # full environment (Python + CUDA + C++)
nix develop .#python   # Python only, no CUDA (smaller, faster build)
nix develop .#cpp      # C++ build tools only
```

With direnv, `direnv allow` activates the environment on entering the directory.

The shell sets `PYTHONPATH` to the project root and loads `.env` automatically (copy from `.env.example`).

## Verify

```bash
nix develop --command python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
nix develop --command python -m bloodBath status
nix develop .#cpp --command make -C bareMetalBender
```

## Apps

```bash
nix run .#bloodBath -- status
nix run .#sync -- --pump-serial 881235
nix run .#trainLSTM
```

## Dependencies

`nix flake update` refreshes all inputs; `nix flake lock --update-input nixpkgs` updates one. The lock file is committed for reproducibility.

If `torch.cuda.is_available()` is `False`: confirm an NVIDIA GPU and host drivers are present, or use `nix develop .#python` for CPU-only work. In containers, `nix develop --impure` exposes host GPU.

## Status and known issues

- The flake is currently a single monolithic `flake.nix` (flake-utils). A dendritic rework (flake-parts + import-tree under `nix/`) is planned on a separate branch.
- `nixpkgs` is pinned to `nixos-23.11`, which is end-of-life. The flake evaluates, but the PyTorch + CUDA build on this pin is fragile; bumping to a current nixpkgs is part of the dendritic rework.
- The legacy `bloodBath-env/` venv is no longer tracked in git. It may still exist on disk; Nix is the supported path.
