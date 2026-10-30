# bloodBender home-manager module.
#
# Declares systemd user timers for sync (15 min) and inference (5 min) using
# the Nix-built wrapper scripts — no `nix develop` at runtime.
#
# Usage in your home-manager config:
#
#   inputs.bloodbender.url = "path:/home/lkshply/dev/bloodBender-data-sync";
#
#   imports = [ inputs.bloodbender.homeManagerModules.bloodbender ];
#
#   services.bloodbender = {
#     enable = true;
#     repoPath = "/home/lkshply/dev/bloodBender-data-sync";
#     # pumpSerial defaults to "1539514"
#   };
#
self:
{ config, lib, pkgs, ... }:
let
  cfg = config.services.bloodbender;
  sys = pkgs.system;
  syncPkg = self.packages.${sys}.bloodbender-sync;
  infPkg  = self.packages.${sys}.bloodbender-inference-cron;
in
{
  options.services.bloodbender = {
    enable = lib.mkEnableOption "bloodBender sync + inference systemd user services";

    repoPath = lib.mkOption {
      type = lib.types.str;
      description = "Absolute path to the bloodBender-data-sync repository root.";
    };

    pumpSerial = lib.mkOption {
      type = lib.types.str;
      default = "1539514";
      description = "Tandem pump serial number to sync and run inference against.";
    };

    syncInterval = lib.mkOption {
      type = lib.types.str;
      default = "15min";
      description = "OnUnitActiveSec for the sync timer (e.g. \"15min\", \"30min\").";
    };

    inferenceInterval = lib.mkOption {
      type = lib.types.str;
      default = "5min";
      description = "OnUnitActiveSec for the inference timer.";
    };
  };

  config = lib.mkIf cfg.enable {
    systemd.user.services.bloodbender-sync = {
      Unit = {
        Description = "bloodBender incremental pump sync";
        After = [ "network-online.target" ];
      };
      Service = {
        Type = "oneshot";
        Environment = [
          "BLOODBENDER_ROOT=${cfg.repoPath}"
          "PUMP=${cfg.pumpSerial}"
        ];
        ExecStart = "${syncPkg}/bin/bloodbender-sync";
        StandardOutput = "journal";
        StandardError = "journal";
      };
    };

    systemd.user.timers.bloodbender-sync = {
      Unit.Description = "bloodBender sync every ${cfg.syncInterval}";
      Timer = {
        OnBootSec = "2min";
        OnUnitActiveSec = cfg.syncInterval;
        AccuracySec = "60";
      };
      Install.WantedBy = [ "timers.target" ];
    };

    systemd.user.services.bloodbender-inference = {
      Unit.Description = "bloodBender inference — refresh dashboard JSON";
      Service = {
        Type = "oneshot";
        Environment = [
          "BLOODBENDER_ROOT=${cfg.repoPath}"
          "PUMP=${cfg.pumpSerial}"
        ];
        ExecStart = "${infPkg}/bin/bloodbender-inference-cron";
        StandardOutput = "journal";
        StandardError = "journal";
      };
    };

    systemd.user.timers.bloodbender-inference = {
      Unit.Description = "bloodBender inference every ${cfg.inferenceInterval}";
      Timer = {
        OnBootSec = "60";
        OnUnitActiveSec = cfg.inferenceInterval;
        AccuracySec = "30";
      };
      Install.WantedBy = [ "timers.target" ];
    };
  };
}
