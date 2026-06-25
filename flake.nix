{
  description = "bloodBender reproducible environment and run targets";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-25.05";
  };

  outputs = { self, nixpkgs }:
    let
      systems = [
        "x86_64-linux"
        "aarch64-linux"
      ];
      forAllSystems = f:
        nixpkgs.lib.genAttrs systems (system:
          f {
            pkgs = import nixpkgs {
              inherit system;
              config.allowUnfree = true;
            };
          });
    in
    {
      packages = forAllSystems ({ pkgs }:
        let
          syncPython = pkgs.python311.override {
            packageOverrides = self: super: {
              fsspec = super.fsspec.overridePythonAttrs (_: {
                doCheck = false;
              });
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
          # Uses the stock interpreter (not syncPython) so torch/numpy/etc. match
          # nixpkgs' prebuilt binary cache instead of rebuilding from source.
          pythonInferenceEnv = pkgs.python311.withPackages (ps: with ps; [
            numpy
            pandas
            scikit-learn
            scipy
            torch
          ]);
        in
        {
          default = pythonSyncEnv;
          bloodbender-env = pythonSyncEnv;
          bloodbender-inference-env = pythonInferenceEnv;

          sync-data = pkgs.writeShellApplication {
            name = "sync-data";
            runtimeInputs = [ pythonSyncEnv ];
            text = ''
              if [[ ! -d "$PWD/bloodBath" ]]; then
                echo "ERR: Run from the bloodBender repository root (missing ./bloodBath)." >&2
                exit 1
              fi
              export PYTHONPATH="$PWD''${PYTHONPATH:+:$PYTHONPATH}"
              exec python -m bloodBath.cli.main sync "$@"
            '';
          };

          normal-full-sync = pkgs.writeShellApplication {
            name = "normal-full-sync";
            runtimeInputs = [ pythonSyncEnv ];
            text = ''
              if [[ ! -d "$PWD/bloodBath" ]]; then
                echo "ERR: Run from the bloodBender repository root (missing ./bloodBath)." >&2
                exit 1
              fi
              export PYTHONPATH="$PWD''${PYTHONPATH:+:$PYTHONPATH}"
              exec python -m bloodBath.cli.main sync --full-sync --chunk-days "''${SYNC_CHUNK_DAYS:-15}" "$@"
            '';
          };

          available-data = pkgs.writeShellApplication {
            name = "available-data";
            runtimeInputs = [ pythonSyncEnv ];
            text = ''
              if [[ ! -d "$PWD/bloodBath" ]]; then
                echo "ERR: Run from the bloodBender repository root (missing ./bloodBath)." >&2
                exit 1
              fi
              export PYTHONPATH="$PWD''${PYTHONPATH:+:$PYTHONPATH}"
              exec python -m bloodBath.cli.main available-data "$@"
            '';
          };

          run-inference = pkgs.writeShellApplication {
            name = "run-inference";
            runtimeInputs = [ pythonInferenceEnv pkgs.bash pkgs.coreutils pkgs.findutils pkgs.gnugrep pkgs.gnused ];
            text = ''
              if [[ ! -f "$PWD/run_lstm_inference.sh" ]]; then
                echo "ERR: Run from the bloodBender repository root (missing ./run_lstm_inference.sh)." >&2
                exit 1
              fi
              export NO_VENV=1
              export BLOODBENDER_ROOT="$PWD"
              export PYTHONPATH="$PWD''${PYTHONPATH:+:$PYTHONPATH}"
              exec bash "$PWD/run_lstm_inference.sh" "$@"
            '';
          };
        });

      apps = forAllSystems ({ pkgs }:
        {
          sync-data = {
            type = "app";
            program = "${self.packages.${pkgs.system}.sync-data}/bin/sync-data";
          };

          normal-full-sync = {
            type = "app";
            program = "${self.packages.${pkgs.system}.normal-full-sync}/bin/normal-full-sync";
          };

          run-inference = {
            type = "app";
            program = "${self.packages.${pkgs.system}.run-inference}/bin/run-inference";
          };

          available-data = {
            type = "app";
            program = "${self.packages.${pkgs.system}.available-data}/bin/available-data";
          };

          default = self.apps.${pkgs.system}.sync-data;
        });

      devShells = forAllSystems ({ pkgs }:
        {
          default = pkgs.mkShell {
            packages = [
              self.packages.${pkgs.system}.bloodbender-env
            ];

            shellHook = ''
              export PYTHONPATH="$PWD''${PYTHONPATH:+:$PYTHONPATH}"
              echo "bloodBender nix shell active"
            '';
          };

          inference = pkgs.mkShell {
            packages = [
              self.packages.${pkgs.system}.bloodbender-inference-env
            ];

            shellHook = ''
              export PYTHONPATH="$PWD''${PYTHONPATH:+:$PYTHONPATH}"
              echo "bloodBender inference nix shell active"
            '';
          };
        });
    };
}