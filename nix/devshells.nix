# CPU development shells. GPU shell is defined separately in nix/cuda.nix.
{ ... }:
{
  perSystem = { pkgs, pythonSyncEnv, pythonInferenceEnv, ... }:
    let
      pyHook = ''export PYTHONPATH="$PWD''${PYTHONPATH:+:$PYTHONPATH}"'';
    in
    {
      devShells.default = pkgs.mkShell {
        packages = [ pythonSyncEnv ];
        shellHook = pyHook + ''
          echo "bloodBender sync shell (CPU)"
        '';
      };

      devShells.inference = pkgs.mkShell {
        packages = [ pythonInferenceEnv ];
        shellHook = pyHook + ''
          echo "bloodBender inference shell (CPU torch)"
        '';
      };

      devShells.cpp = pkgs.mkShell {
        packages = with pkgs; [ gcc gnumake cmake ];
        shellHook = ''echo "bareMetalBender C++ shell -- make -C bareMetalBender"'';
      };
    };
}
