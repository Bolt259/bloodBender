{
  description = "bloodBender - Tandem insulin pump data processing & ML glucose prediction";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-25.05";
    flake-parts.url = "github:hercules-ci/flake-parts";
  };

  # The cuda devShell can pull prebuilt CUDA/torch from this cache instead of
  # compiling from source -- but the Nix daemon only honors it for trusted
  # users. If `trusted-users = root` (the default), run as root once:
  #   echo "trusted-users = root $USER" >> /etc/nix/nix.conf && systemctl restart nix-daemon
  # Otherwise the cuda env builds torch from source (~hours). CPU shells are
  # unaffected: they hit the default cache.nixos.org.
  nixConfig = {
    extra-substituters = [ "https://cuda-maintainers.cachix.org" ];
    extra-trusted-public-keys = [
      "cuda-maintainers.cachix.org-1:0dq3bujKpuEPMCX6U4WylrUDZ9JyUG0VpVZa7CNfq5E="
    ];
  };

  # Dendritic-lite: flake-parts with an explicit per-concern module tree under
  # nix/. CPU is the default everywhere (reliable binary cache); GPU lives in
  # its own nix/cuda.nix module for local accelerated training (e.g. RTX 2070).
  outputs = inputs@{ flake-parts, ... }:
    flake-parts.lib.mkFlake { inherit inputs; } {
      systems = [ "x86_64-linux" "aarch64-linux" ];

      imports = [
        ./nix/pkgs.nix
        ./nix/python.nix
        ./nix/packages.nix
        ./nix/apps.nix
        ./nix/devshells.nix
        ./nix/cuda.nix
      ];

      perSystem = { pkgs, ... }: {
        formatter = pkgs.nixpkgs-fmt;
      };
    };
}
