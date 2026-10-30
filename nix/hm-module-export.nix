# flake-parts module: exports homeManagerModules.bloodbender from the flake.
# `self` here is the flake's final output set, passed by flake-parts.
{ self, ... }:
{
  flake.homeManagerModules.bloodbender = import ./hm-module.nix self;
}
