# Package outputs: the Python envs, the sync/inference run wrappers (ported
# verbatim from main), and the bareMetalBender C++ solver.
{ ... }:
{
  perSystem = { pkgs, pythonSyncEnv, pythonInferenceEnv, ... }: {
    packages = {
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

      # C++ glucose-insulin IVP solver. Makefile target is `system` (not `ivp`).
      bareMetalBender = pkgs.stdenv.mkDerivation {
        pname = "bareMetalBender";
        version = "1.0.0";
        src = ../bareMetalBender;
        nativeBuildInputs = [ pkgs.gnumake pkgs.gcc ];
        buildPhase = "make";
        installPhase = ''
          mkdir -p $out/bin
          cp system $out/bin/bareMetalBender
        '';
        meta = {
          description = "C++ glucose-insulin IVP solver";
          license = pkgs.lib.licenses.mit;
          mainProgram = "bareMetalBender";
        };
      };
    };
  };
}
