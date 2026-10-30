# Package outputs: the Python envs, the sync/inference run wrappers (ported
# verbatim from main), and the bareMetalBender C++ solver.
{ ... }:
{
  perSystem = { pkgs, config, pythonSyncEnv, pythonInferenceEnv, ... }: {
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

      # ── Daemon service wrappers (used by systemd units + homeManagerModules) ──
      # These bake in the correct Nix Python env so the unit's ExecStart never
      # needs to call `nix develop` at runtime — cold-start drops from ~4s to <1s.

      bloodbender-sync = pkgs.writeShellApplication {
        name = "bloodbender-sync";
        runtimeInputs = [ pythonSyncEnv pkgs.coreutils ];
        text = ''
          REPO="''${BLOODBENDER_ROOT:-$PWD}"
          PUMP="''${PUMP:-1539514}"
          if [[ ! -d "$REPO/bloodBath" ]]; then
            echo "ERR: BLOODBENDER_ROOT ($REPO) missing ./bloodBath" >&2
            exit 1
          fi
          cd "$REPO"
          export PYTHONPATH="$REPO''${PYTHONPATH:+:$PYTHONPATH}"
          exec python -m bloodBath.cli.main sync --pump-serial "$PUMP" --update
        '';
      };

      bloodbender-inference-cron = pkgs.writeShellApplication {
        name = "bloodbender-inference-cron";
        runtimeInputs = [ pythonInferenceEnv pkgs.coreutils ];
        text = ''
          REPO="''${BLOODBENDER_ROOT:-$PWD}"
          PUMP="''${PUMP:-1539514}"
          LOG_DIR="$REPO/bloodBath/logs"
          OUT="$LOG_DIR/inference_latest.json"
          ERR="$LOG_DIR/inference_latest.err"
          TMP="$LOG_DIR/.inference_latest.json.tmp"
          mkdir -p "$LOG_DIR"
          cd "$REPO"
          PYTHONPATH="$REPO''${PYTHONPATH:+:$PYTHONPATH}" \
            python bloodTwin/pipelines/run_inference.py --pump "$PUMP" \
            >"$TMP" 2>"$ERR"
          rc=$?
          if [[ "$rc" -eq 0 && -s "$TMP" ]]; then
            mv -f "$TMP" "$OUT"
          else
            rm -f "$TMP"
            echo "$(date -Iseconds) inference exited $rc; kept previous $OUT" >>"$ERR"
          fi
          exit "$rc"
        '';
      };

      # ── Monthly conformal recalibration (#9) ──
      # Re-fits conformal_cal.pkl on the latest merged data so prediction
      # intervals track the patient's shifting distribution over months.
      bloodbender-calibrate = pkgs.writeShellApplication {
        name = "bloodbender-calibrate";
        runtimeInputs = [ pythonInferenceEnv pkgs.coreutils ];
        text = ''
          REPO="''${BLOODBENDER_ROOT:-$PWD}"
          ARTIFACTS="''${BLOODTWIN_ARTIFACTS:-$REPO/bloodTwin/artifacts/bloodTwin_unified_lstm_v1_3}"
          cd "$REPO"
          PYTHONPATH="$REPO''${PYTHONPATH:+:$PYTHONPATH}" \
            python bloodTwin/pipelines/calibrate_conformal.py \
            --artifacts "$ARTIFACTS" --coverage 0.90 --stride 6
        '';
      };

      # ── Weekly ISF re-estimation (#11) ──
      # Re-fits the ISFModel so it tracks the patient's current insulin
      # sensitivity rather than the value frozen at training time.
      bloodbender-fit-isf = pkgs.writeShellApplication {
        name = "bloodbender-fit-isf";
        runtimeInputs = [ pythonInferenceEnv pkgs.coreutils ];
        text = ''
          REPO="''${BLOODBENDER_ROOT:-$PWD}"
          ARTIFACTS="''${BLOODTWIN_ARTIFACTS:-$REPO/bloodTwin/artifacts/bloodTwin_unified_lstm_v1_3}"
          PUMP="''${PUMP:-1539514}"
          cd "$REPO"
          PYTHONPATH="$REPO''${PYTHONPATH:+:$PYTHONPATH}" \
            python bloodTwin/pipelines/fit_isf_model.py \
            --pump "$PUMP" --artifacts "$ARTIFACTS"
        '';
      };

      # `nix run .#install-services` — write + enable systemd user units that
      # point at the Nix store paths above.  Idempotent; safe to re-run after
      # a flake update (units are rewritten with the new store paths).
      install-services = pkgs.writeShellApplication {
        name = "install-services";
        runtimeInputs = [ pkgs.systemd pkgs.coreutils ];
        text =
          let
            syncBin = "${config.packages.bloodbender-sync}/bin/bloodbender-sync";
            infBin  = "${config.packages.bloodbender-inference-cron}/bin/bloodbender-inference-cron";
            calBin  = "${config.packages.bloodbender-calibrate}/bin/bloodbender-calibrate";
            isfBin  = "${config.packages.bloodbender-fit-isf}/bin/bloodbender-fit-isf";
          in ''
            REPO="''${BLOODBENDER_ROOT:-$PWD}"
            UNIT="$HOME/.config/systemd/user"
            mkdir -p "$UNIT"

            cat >"$UNIT/bloodbender-sync.service" <<EOF
            [Unit]
            Description=bloodBender incremental pump sync
            After=network-online.target

            [Service]
            Type=oneshot
            Environment=BLOODBENDER_ROOT=$REPO
            ExecStart=${syncBin}
            StandardOutput=journal
            StandardError=journal
            EOF

            cat >"$UNIT/bloodbender-sync.timer" <<EOF
            [Unit]
            Description=bloodBender sync every 15 minutes

            [Timer]
            OnBootSec=2min
            OnUnitActiveSec=15min
            AccuracySec=60

            [Install]
            WantedBy=timers.target
            EOF

            cat >"$UNIT/bloodbender-inference.service" <<EOF
            [Unit]
            Description=bloodBender inference — refresh dashboard JSON
            After=network.target

            [Service]
            Type=oneshot
            Environment=BLOODBENDER_ROOT=$REPO
            ExecStart=${infBin}
            StandardOutput=journal
            StandardError=journal
            EOF

            cat >"$UNIT/bloodbender-inference.timer" <<EOF
            [Unit]
            Description=bloodBender inference every 5 minutes

            [Timer]
            OnBootSec=60
            OnUnitActiveSec=5min
            AccuracySec=30

            [Install]
            WantedBy=timers.target
            EOF

            cat >"$UNIT/bloodbender-calibrate.service" <<EOF
            [Unit]
            Description=bloodBender monthly conformal recalibration

            [Service]
            Type=oneshot
            Environment=BLOODBENDER_ROOT=$REPO
            ExecStart=${calBin}
            StandardOutput=journal
            StandardError=journal
            EOF

            cat >"$UNIT/bloodbender-calibrate.timer" <<EOF
            [Unit]
            Description=bloodBender conformal recalibration (monthly)

            [Timer]
            OnCalendar=monthly
            Persistent=true
            AccuracySec=1h

            [Install]
            WantedBy=timers.target
            EOF

            cat >"$UNIT/bloodbender-fit-isf.service" <<EOF
            [Unit]
            Description=bloodBender weekly ISF re-estimation

            [Service]
            Type=oneshot
            Environment=BLOODBENDER_ROOT=$REPO
            ExecStart=${isfBin}
            StandardOutput=journal
            StandardError=journal
            EOF

            cat >"$UNIT/bloodbender-fit-isf.timer" <<EOF
            [Unit]
            Description=bloodBender ISF re-estimation (weekly)

            [Timer]
            OnCalendar=weekly
            Persistent=true
            AccuracySec=1h

            [Install]
            WantedBy=timers.target
            EOF

            systemctl --user daemon-reload
            systemctl --user enable --now bloodbender-sync.timer bloodbender-inference.timer bloodbender-calibrate.timer bloodbender-fit-isf.timer
            echo "Services installed and enabled."
            systemctl --user list-timers bloodbender-sync.timer bloodbender-inference.timer bloodbender-calibrate.timer bloodbender-fit-isf.timer --no-pager
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
