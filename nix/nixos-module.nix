# bloodBender NixOS system-level service module.
#
# Declares system-level systemd services (not HM user services) so they start
# at boot without a login session, survive a reflash, and sit beside system
# services like AdGuard/Unbound/Tailscale.
#
# Resource caps (MemoryMax / CPUWeight / IOWeight) ensure bloodBender is
# best-effort and can NEVER starve or OOM the DNS stack on a shared host.
#
# Usage in a NixOS host (e.g. nixbox host "hole"):
#
#   inputs.bloodbender.url = "github:Bolt259/bloodBender";
#   imports = [ inputs.bloodbender.nixosModules.bloodbender ];
#
#   services.bloodbender = {
#     enable          = true;
#     sourceDir       = "/home/lkshply/bloodBender";
#     dataDir         = "/mnt/ssd/bloodbender-data";  # USB-SSD strongly recommended
#     user            = "lkshply";
#     credentialsFile = "/home/lkshply/.env";          # Tandem creds — never committed
#   };
#
self:
{ config, lib, pkgs, ... }:
let
  cfg = config.services.bloodbender;
  sys = pkgs.system;

  syncPkg  = self.packages.${sys}.bloodbender-sync;
  infPkg   = self.packages.${sys}.bloodbender-inference-cron;
  calPkg   = self.packages.${sys}.bloodbender-calibrate;
  isfPkg   = self.packages.${sys}.bloodbender-fit-isf;
  dashPkg  = self.packages.${sys}.bloodbender-dashboard;

  # Base environment for all services (DR-11 split: source ≠ data).
  baseEnv = [
    "BLOODBENDER_SOURCE=${cfg.sourceDir}"
    "BLOODBENDER_ROOT=${cfg.dataDir}"
    "BLOODTWIN_ARTIFACTS=bloodTwin/artifacts/current"
    "PUMP=${cfg.pump}"
  ];

  # Shared service hardening + resource caps.
  # bloodBender is BEST-EFFORT; DNS stack is PRIORITY TENANT on this box.
  # On a 4 GB Pi 5 shared with AdGuard + Unbound + Tailscale:
  #   DNS stack peak: ~600-800 MB
  #   bloodBender services (non-concurrent): ~200-500 MB peak each
  #   Dashboard (persistent): ~200 MB
  #   Headroom for kernel + other: ~1 GB+
  commonCaps = {
    MemoryHigh      = "400M";   # kernel starts reclaiming; service stays alive
    MemoryMax       = "512M";   # hard cap — OOM kill fires above this
    CPUWeight       = 20;       # systemd default=100; DNS stack has priority
    IOWeight        = 10;       # heavily deprioritised to protect SD/SSD + DNS
    OOMScoreAdjust  = 500;      # kernel prefers to kill bloodBender over AdGuard
  };

  # Full base service config — merge commonCaps then override per-service.
  baseService = {
    User             = cfg.user;
    Group            = cfg.group;
    WorkingDirectory = cfg.sourceDir;
    Environment      = baseEnv;
    StandardOutput   = "journal";
    StandardError    = "journal";
  } // commonCaps
    // lib.optionalAttrs (cfg.credentialsFile != null) {
      EnvironmentFile = cfg.credentialsFile;
    };
in
{
  options.services.bloodbender = {
    enable = lib.mkEnableOption "bloodBender glucose sync + ONNX inference + dashboard";

    sourceDir = lib.mkOption {
      type        = lib.types.str;
      description = "Absolute path to the bloodBender source repo on this host.";
    };

    dataDir = lib.mkOption {
      type        = lib.types.str;
      description = ''
        Runtime data root: merged CSVs, logs, model artifacts.
        USB-SSD strongly recommended — the 5-min inference cron writes JSON on
        every cycle; SD-card churn will wear and slow a microSD over time.
      '';
    };

    user = lib.mkOption {
      type    = lib.types.str;
      default = "lkshply";
    };

    group = lib.mkOption {
      type    = lib.types.str;
      default = "users";
    };

    pump = lib.mkOption {
      type        = lib.types.str;
      default     = "1539514";
      description = "Tandem pump serial number to sync and run inference against.";
    };

    credentialsFile = lib.mkOption {
      type        = lib.types.nullOr lib.types.str;
      default     = null;
      description = ''
        Path to a .env / EnvironmentFile containing Tandem API credentials.
        This file must NEVER be committed to version control.
        Example: /home/lkshply/.env  or  /run/secrets/bloodbender
      '';
    };

    dashboard = {
      host = lib.mkOption {
        type    = lib.types.str;
        default = "0.0.0.0";
        description = "Dashboard listen address. Use 0.0.0.0 for LAN/Tailscale access.";
      };
      port = lib.mkOption {
        type    = lib.types.int;
        default = 8765;
      };
    };

    syncInterval = lib.mkOption {
      type    = lib.types.str;
      default = "15min";
    };

    inferenceInterval = lib.mkOption {
      type    = lib.types.str;
      default = "5min";
    };

    enableCalibrationTimer = lib.mkOption {
      type        = lib.types.bool;
      default     = false;
      description = "Monthly conformal recalibration. Disable until 30+ days of Pi-local data.";
    };

    enableIsfTimer = lib.mkOption {
      type        = lib.types.bool;
      default     = false;
      description = "Weekly ISF re-estimation. Disable until sufficient Pi-local data.";
    };
  };

  config = lib.mkIf cfg.enable (lib.mkMerge [
    {
      # ── Sync ──────────────────────────────────────────────────────────────
      systemd.services.bloodbender-sync = {
        description = "bloodBender incremental pump sync";
        after  = [ "network-online.target" ];
        wants  = [ "network-online.target" ];
        serviceConfig = baseService // {
          Type       = "oneshot";
          Restart    = "on-failure";
          RestartSec = "60";
          ExecStart  = "${syncPkg}/bin/bloodbender-sync";
        };
      };

      systemd.timers.bloodbender-sync = {
        description = "bloodBender pump sync every ${cfg.syncInterval}";
        wantedBy    = [ "timers.target" ];
        timerConfig = {
          OnBootSec       = "5min";
          OnUnitActiveSec = cfg.syncInterval;
          AccuracySec     = "60";
        };
      };

      # ── Inference ─────────────────────────────────────────────────────────
      # Lightweight: single ONNX forward pass (~100 ms on Pi 5). No GPU needed.
      systemd.services.bloodbender-inference = {
        description = "bloodBender ONNX inference — refresh dashboard JSON";
        serviceConfig = baseService // {
          Type       = "oneshot";
          Restart    = "on-failure";
          RestartSec = "30";
          ExecStart  = "${infPkg}/bin/bloodbender-inference-cron";
        };
      };

      systemd.timers.bloodbender-inference = {
        description = "bloodBender inference every ${cfg.inferenceInterval}";
        wantedBy    = [ "timers.target" ];
        timerConfig = {
          OnBootSec       = "90";
          OnUnitActiveSec = cfg.inferenceInterval;
          AccuracySec     = "30";
        };
      };

      # ── Dashboard (persistent) ────────────────────────────────────────────
      systemd.services.bloodbender-dashboard = {
        description = "bloodBender dashboard HTTP server";
        after    = [ "network.target" ];
        wantedBy = [ "multi-user.target" ];
        serviceConfig = baseService // {
          Type       = "simple";
          Restart    = "on-failure";
          RestartSec = "10";
          # Dashboard gets its own env list to include DASHBOARD_* vars
          Environment = baseEnv ++ [
            "DASHBOARD_HOST=${cfg.dashboard.host}"
            "DASHBOARD_PORT=${toString cfg.dashboard.port}"
          ];
          ExecStart  = "${dashPkg}/bin/bloodbender-dashboard";
          # Dashboard is lighter than inference — tighter cap
          MemoryHigh = "300M";
          MemoryMax  = "384M";
        };
      };
    }

    (lib.mkIf cfg.enableCalibrationTimer {
      systemd.services.bloodbender-calibrate = {
        description = "bloodBender monthly conformal recalibration";
        serviceConfig = baseService // {
          Type      = "oneshot";
          Restart   = "on-failure";
          ExecStart = "${calPkg}/bin/bloodbender-calibrate";
        };
      };
      systemd.timers.bloodbender-calibrate = {
        description = "bloodBender conformal recalibration (monthly)";
        wantedBy    = [ "timers.target" ];
        timerConfig = {
          OnCalendar  = "monthly";
          Persistent  = true;
          AccuracySec = "1h";
        };
      };
    })

    (lib.mkIf cfg.enableIsfTimer {
      systemd.services.bloodbender-fit-isf = {
        description = "bloodBender weekly ISF re-estimation";
        serviceConfig = baseService // {
          Type      = "oneshot";
          Restart   = "on-failure";
          ExecStart = "${isfPkg}/bin/bloodbender-fit-isf";
        };
      };
      systemd.timers.bloodbender-fit-isf = {
        description = "bloodBender ISF re-estimation (weekly)";
        wantedBy    = [ "timers.target" ];
        timerConfig = {
          OnCalendar  = "weekly";
          Persistent  = true;
          AccuracySec = "1h";
        };
      };
    })
  ]);
}
