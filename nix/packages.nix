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
          # SRC = Python source (bloodBender repo); REPO = runtime data root.
          SRC="''${BLOODBENDER_SOURCE:-''${BLOODBENDER_ROOT:-$PWD}}"
          REPO="''${BLOODBENDER_ROOT:-$SRC}"
          PUMP="''${PUMP:-1539514}"
          if [[ ! -d "$SRC/bloodBath" ]]; then
            echo "ERR: BLOODBENDER_SOURCE ($SRC) missing ./bloodBath" >&2
            exit 1
          fi
          cd "$SRC"
          export BLOODBENDER_ROOT="$REPO"
          export PYTHONPATH="$SRC''${PYTHONPATH:+:$PYTHONPATH}"
          exec python -m bloodBath.cli.main sync --pump-serial "$PUMP" --update
        '';
      };

      bloodbender-inference-cron = pkgs.writeShellApplication {
        name = "bloodbender-inference-cron";
        runtimeInputs = [ pythonInferenceEnv pkgs.coreutils ];
        text = ''
          SRC="''${BLOODBENDER_SOURCE:-''${BLOODBENDER_ROOT:-$PWD}}"
          REPO="''${BLOODBENDER_ROOT:-$SRC}"
          PUMP="''${PUMP:-1539514}"
          LOG_DIR="$REPO/bloodBath/logs"
          OUT="$LOG_DIR/inference_latest.json"
          ERR="$LOG_DIR/inference_latest.err"
          TMP="$LOG_DIR/.inference_latest.json.tmp"
          mkdir -p "$LOG_DIR"
          cd "$SRC"
          export BLOODBENDER_ROOT="$REPO"
          export PYTHONPATH="$SRC''${PYTHONPATH:+:$PYTHONPATH}"
          python bloodTwin/pipelines/run_inference.py --pump "$PUMP" \
            >"$TMP" 2>"$ERR"
          rc=$?
          if [[ "$rc" -eq 0 && -s "$TMP" ]]; then
            mv -f "$TMP" "$OUT"
            # Archive every inference for production accuracy tracking (BT-09)
            python bloodTwin/inference/archive.py \
              --latest "$OUT" --archive-dir "$LOG_DIR/inference_archive" \
              >>"$ERR" 2>&1 || true
            # Fire push notifications if NTFY_TOPIC is set (IN-01)
            if [[ -n "''${NTFY_TOPIC:-}" ]]; then
              python bloodTwin/inference/notify.py \
                --latest "$OUT" --log-dir "$LOG_DIR" \
                >>"$ERR" 2>&1 || true
            fi
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
          SRC="''${BLOODBENDER_SOURCE:-''${BLOODBENDER_ROOT:-$PWD}}"
          REPO="''${BLOODBENDER_ROOT:-$SRC}"
          ARTIFACTS="''${BLOODTWIN_ARTIFACTS:-$REPO/bloodTwin/artifacts/current}"
          cd "$SRC"
          PYTHONPATH="$SRC''${PYTHONPATH:+:$PYTHONPATH}" \
            python bloodTwin/pipelines/calibrate_conformal.py \
            --artifacts "$ARTIFACTS" --coverage 0.90 --stride 6
        '';
      };

      # ── Weekly ISF re-estimation (#11) ──
      bloodbender-fit-isf = pkgs.writeShellApplication {
        name = "bloodbender-fit-isf";
        runtimeInputs = [ pythonInferenceEnv pkgs.coreutils ];
        text = ''
          SRC="''${BLOODBENDER_SOURCE:-''${BLOODBENDER_ROOT:-$PWD}}"
          REPO="''${BLOODBENDER_ROOT:-$SRC}"
          ARTIFACTS="''${BLOODTWIN_ARTIFACTS:-$REPO/bloodTwin/artifacts/current}"
          PUMP="''${PUMP:-1539514}"
          cd "$SRC"
          PYTHONPATH="$SRC''${PYTHONPATH:+:$PYTHONPATH}" \
            python bloodTwin/pipelines/fit_isf_model.py \
            --pump "$PUMP" --artifacts "$ARTIFACTS"
        '';
      };

      # ── Dashboard HTTP server ──
      bloodbender-dashboard = pkgs.writeShellApplication {
        name = "bloodbender-dashboard";
        runtimeInputs = [ pythonInferenceEnv pkgs.coreutils ];
        text = ''
          SRC="''${BLOODBENDER_SOURCE:-''${BLOODBENDER_ROOT:-$PWD}}"
          REPO="''${BLOODBENDER_ROOT:-$SRC}"
          if [[ ! -d "$SRC/bloodTwin" ]]; then
            echo "ERR: BLOODBENDER_SOURCE ($SRC) missing ./bloodTwin" >&2
            exit 1
          fi
          cd "$SRC"
          export BLOODBENDER_ROOT="$REPO"
          export PYTHONPATH="$SRC''${PYTHONPATH:+:$PYTHONPATH}"
          export DASHBOARD_HOST="''${DASHBOARD_HOST:-0.0.0.0}"
          exec python bloodTwin/dashboard/server.py
        '';
      };

      # ── Production accuracy tracker (BT-09) ──
      bloodbender-accuracy = pkgs.writeShellApplication {
        name = "bloodbender-accuracy";
        runtimeInputs = [ pythonInferenceEnv pkgs.coreutils ];
        text = ''
          SRC="''${BLOODBENDER_SOURCE:-''${BLOODBENDER_ROOT:-$PWD}}"
          REPO="''${BLOODBENDER_ROOT:-$SRC}"
          PUMP="''${PUMP:-1539514}"
          cd "$SRC"
          BLOODBENDER_ROOT="$REPO" PYTHONPATH="$SRC''${PYTHONPATH:+:$PYTHONPATH}" \
            python scripts/accuracy_tracker.py
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
            syncBin  = "${config.packages.bloodbender-sync}/bin/bloodbender-sync";
            infBin   = "${config.packages.bloodbender-inference-cron}/bin/bloodbender-inference-cron";
            calBin   = "${config.packages.bloodbender-calibrate}/bin/bloodbender-calibrate";
            isfBin   = "${config.packages.bloodbender-fit-isf}/bin/bloodbender-fit-isf";
            dashBin  = "${config.packages.bloodbender-dashboard}/bin/bloodbender-dashboard";
          in ''
            # SRC = bloodBender source repo (where nix run is called from, or $BLOODBENDER_SOURCE).
            # DATA = runtime data root (merged CSVs, logs, artifacts).
            # These are the same dir in the legacy two-dir setup; separated by DR-11.
            SRC="''${BLOODBENDER_SOURCE:-$PWD}"
            DATA="''${BLOODBENDER_ROOT:-$SRC}"
            ARTIFACTS_REL="''${BLOODTWIN_ARTIFACTS:-bloodTwin/artifacts/current}"
            UNIT="$HOME/.config/systemd/user"
            mkdir -p "$UNIT"

            # Ensure artifacts/current symlink exists (points to the active model).
            # Create it if missing; leave it alone if already set (operator manages it).
            ARTS_DIR="$DATA/bloodTwin/artifacts"
            if [[ -d "$ARTS_DIR" && ! -e "$ARTS_DIR/current" ]]; then
              LATEST=$(find "$ARTS_DIR" -maxdepth 1 -name 'bloodTwin_unified_lstm_v*' -type d | sort -r | head -1)
              if [[ -n "$LATEST" ]]; then
                ln -sfn "$(basename "$LATEST")" "$ARTS_DIR/current"
                echo "Created $ARTS_DIR/current → $(basename "$LATEST")"
              fi
            fi

            write_service() {
              local name="$1" desc="$2" bin="$3"
              cat >"$UNIT/$name.service" <<EOF
[Unit]
Description=$desc
After=network.target

[Service]
Type=oneshot
Environment=BLOODBENDER_SOURCE=$SRC
Environment=BLOODBENDER_ROOT=$DATA
Environment=BLOODTWIN_ARTIFACTS=$ARTIFACTS_REL
ExecStart=$bin
StandardOutput=journal
StandardError=journal
EOF
            }

            write_service "bloodbender-sync"      "bloodBender incremental pump sync"              "${syncBin}"
            write_service "bloodbender-inference"  "bloodBender inference — refresh dashboard JSON" "${infBin}"
            write_service "bloodbender-calibrate"  "bloodBender monthly conformal recalibration"    "${calBin}"
            write_service "bloodbender-fit-isf"    "bloodBender weekly ISF re-estimation"           "${isfBin}"

            # Dashboard: persistent service (not oneshot), restarts on failure.
            # Binds to 0.0.0.0 so it's reachable from other devices on the LAN.
            # On a headless Pi run: loginctl enable-linger <user>
            cat >"$UNIT/bloodbender-dashboard.service" <<EOF
[Unit]
Description=bloodBender dashboard server
After=network.target

[Service]
Type=simple
Environment=BLOODBENDER_SOURCE=$SRC
Environment=BLOODBENDER_ROOT=$DATA
Environment=BLOODTWIN_ARTIFACTS=$ARTIFACTS_REL
Environment=DASHBOARD_HOST=0.0.0.0
ExecStart=${dashBin}
Restart=on-failure
RestartSec=10

[Install]
WantedBy=default.target
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
            systemctl --user enable --now bloodbender-sync.timer bloodbender-inference.timer bloodbender-calibrate.timer bloodbender-fit-isf.timer bloodbender-dashboard.service
            echo "Services installed and enabled."
            systemctl --user list-timers bloodbender-sync.timer bloodbender-inference.timer bloodbender-calibrate.timer bloodbender-fit-isf.timer --no-pager
            echo ""
            systemctl --user status bloodbender-dashboard.service --no-pager
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
