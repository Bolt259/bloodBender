# Separate GPU module: CUDA-enabled torch for local accelerated training
# (e.g. RTX 2070 SUPER). Isolated from the CPU path so it can't poison the
# default cache hits. x86_64-linux only.
#
# To get prebuilt CUDA torch instead of a multi-hour source build, you must be
# a Nix trusted user so the cuda-maintainers substituter (see flake nixConfig)
# is honored. As root, once:
#   echo "trusted-users = root $USER" >> /etc/nix/nix.conf
#   systemctl restart nix-daemon
{ inputs, lib, ... }:
{
  perSystem = { system, ... }:
    lib.optionalAttrs (system == "x86_64-linux") (
      let
        pkgsCuda = import inputs.nixpkgs {
          inherit system;
          config = {
            allowUnfree = true;
            cudaSupport = true;
          };
        };

        cudaEnv = pkgsCuda.python311.withPackages (ps: with ps; [
          numpy
          pandas
          scikit-learn
          scipy
          torch
          torchvision
          torchaudio
          pytorch-lightning
          torchmetrics
          tensorboard
          onnx
        ]);
      in
      {
        packages.bloodbender-cuda-env = cudaEnv;

        devShells.cuda = pkgsCuda.mkShell {
          packages = [ cudaEnv ];
          shellHook = ''
            export PYTHONPATH="$PWD''${PYTHONPATH:+:$PYTHONPATH}"
            echo "bloodBender CUDA shell (GPU torch)"
            python -c 'import torch; print("CUDA available:", torch.cuda.is_available())' || true
          '';
        };
      }
    );
}
