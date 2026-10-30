# Runnable apps (`nix run .#<name>`), backed by the wrappers in nix/packages.nix.
{ ... }:
{
  perSystem = { config, ... }:
    let
      app = name: {
        type = "app";
        program = "${config.packages.${name}}/bin/${name}";
      };
    in
    {
      apps = {
        sync-data = app "sync-data";
        normal-full-sync = app "normal-full-sync";
        available-data = app "available-data";
        run-inference = app "run-inference";
        bloodbender-sync = app "bloodbender-sync";
        bloodbender-inference-cron = app "bloodbender-inference-cron";
        bloodbender-calibrate = app "bloodbender-calibrate";
        bloodbender-fit-isf = app "bloodbender-fit-isf";
        install-services = app "install-services";

        # Task B — retraining (#12).
        # Full v1.3 retrain takes 20+ min and needs a CUDA GPU, so it is
        # intentionally NOT exposed as a scheduled `nix run .#retrain` app:
        # the CPU python env can't train on GPU, and the CUDA env
        # (packages.bloodbender-cuda-env) only exists on x86_64-linux and is
        # not wired in as a module arg here. Retrain manually with:
        #
        #   nix develop .#cuda
        #   PYTHONPATH="$PWD" python bloodTwin/pipelines/train_lstm.py \
        #     --config bloodTwin/configs/lstm_v1_3.yaml
        #
        # (BLOODTWIN_CONFIG overrides the config path.)

        default = app "sync-data";
      };
    };
}
