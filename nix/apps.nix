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
        default = app "sync-data";
      };
    };
}
