# CPU Python environments, shared with other modules via _module.args.
#   pythonSyncEnv      - data sync/processing (bloodBath)
#   pythonInferenceEnv - CPU inference (stock interpreter => torch from cache)
{ ... }:
{
  perSystem = { pkgs, ... }:
    let
      # fsspec's test suite is flaky in the sandbox; skip it for the sync env.
      # This override only touches syncPython, so it does NOT poison the
      # inference env's hashes (which must match nixpkgs' prebuilt binaries).
      syncPython = pkgs.python311.override {
        packageOverrides = _self: super: {
          fsspec = super.fsspec.overridePythonAttrs (_: { doCheck = false; });
        };
      };

      pythonSyncEnv = syncPython.withPackages (ps: with ps; [
        arrow
        cryptography
        numpy
        pandas
        pyjwt
        python-dotenv
        pyyaml
        requests
        scikit-learn
        scipy
      ]);

      # Stock interpreter so torch/numpy/etc. match nixpkgs' binary cache
      # instead of rebuilding from source.
      pythonInferenceEnv = pkgs.python311.withPackages (ps: with ps; [
        numpy
        pandas
        scikit-learn
        scipy
        torch
      ]);
    in
    {
      _module.args = { inherit pythonSyncEnv pythonInferenceEnv; };
    };
}
