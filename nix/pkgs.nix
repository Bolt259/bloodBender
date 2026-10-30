# Shared CPU nixpkgs instance (unfree allowed for completeness; no cudaSupport
# so torch/numpy/etc. resolve to Hydra's prebuilt binaries). CUDA lives in
# nix/cuda.nix with its own pkgs instance.
{ inputs, ... }:
{
  perSystem = { system, ... }: {
    _module.args.pkgs = import inputs.nixpkgs {
      inherit system;
      config.allowUnfree = true;
    };
  };
}
