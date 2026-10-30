# CPU development shells. GPU shell is defined separately in nix/cuda.nix.
{ ... }:
{
  perSystem = { pkgs, pythonSyncEnv, pythonInferenceEnv, ... }:
    let
      pyHook = ''export PYTHONPATH="$PWD''${PYTHONPATH:+:$PYTHONPATH}"'';
    in
    {
      devShells.default = pkgs.mkShell {
        packages = [ pythonSyncEnv pkgs.uv ];
        # uv must use the Nix-provided Python on NixOS (can't run bundled glibc binaries)
        UV_PYTHON_DOWNLOADS = "never";
        UV_PYTHON = "${pythonSyncEnv}/bin/python3";
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
