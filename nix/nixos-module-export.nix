# flake-parts module: exports nixosModules.bloodbender from the flake.
{ self, ... }:
{
  flake.nixosModules.bloodbender = import ./nixos-module.nix self;
}
