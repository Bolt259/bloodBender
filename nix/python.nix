# CPU Python environments, shared with other modules via _module.args.
#   pythonSyncEnv      - data sync/processing (bloodBath)
#   pythonInferenceEnv - CPU inference via onnxruntime on the exported model.onnx
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

      # Inference runs the exported model.onnx through onnxruntime, so torch is
      # not needed at inference time. onnxruntime is prebuilt in the binary cache
      # (torch/CUDA are not, for this pin), so the closure is far smaller and
      # substitutes instead of compiling from source. scikit-learn stays for
      # unpickling the RobustScaler; scipy is its dependency.
      pythonInferenceEnv = pkgs.python311.withPackages (ps: with ps; [
        numpy
        pandas
        scikit-learn
        scipy
        onnxruntime
      ]);
    in
    {
      _module.args = { inherit pythonSyncEnv pythonInferenceEnv; };
    };
}
