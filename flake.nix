{
  description = "bloodBender - Advanced Tandem Insulin Pump Data Processing & ML Prediction System";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-23.11";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs {
          inherit system;
          config = {
            allowUnfree = true; # Required for CUDA packages
            cudaSupport = true;
          };
        };

        # Python version
        python = pkgs.python310;

        # Python packages
        pythonPackages = python.pkgs;

        # Core Python dependencies
        pythonEnv = python.withPackages (ps: with ps; [
          # Core data processing
          numpy
          pandas
          arrow

          # PyTorch and ML (with CUDA support)
          pytorch
          torchvision
          torchaudio
          
          # Lightning and ML tools
          pytorch-lightning
          torchmetrics
          tensorboard
          
          # ONNX
          onnx
          
          # Machine Learning
          scikit-learn
          scipy
          
          # Utilities
          certifi
          requests
          urllib3
          python-dotenv
          
          # Logging and formatting
          colorama
          coloredlogs
          
          # Development tools
          pytest
          pytest-cov
          black
          flake8
          mypy
          ipython
          jupyter
          
          # Timezone handling
          pytz
          python-dateutil
          
          # JSON/YAML
          pyyaml
          
          # Type hints
          types-pyyaml
          types-requests
        ]);

        # C++ build environment for bareMetalBender
        cppBuildInputs = with pkgs; [
          gcc
          gnumake
          cmake
        ];

        # CUDA packages (optional, for GPU support)
        cudaPackages = with pkgs; [
          cudatoolkit
          cudnn
          linuxPackages.nvidia_x11
        ];

        # Development tools
        devTools = with pkgs; [
          git
          direnv
          nix-direnv
          
          # Editors/LSP
          nil # Nix LSP
          
          # Utilities
          tree
          htop
          tmux
        ];

        # Full development shell
        devShell = pkgs.mkShell {
          name = "bloodBender-dev";
          
          buildInputs = [
            pythonEnv
          ] ++ cppBuildInputs ++ devTools;

          shellHook = ''
            echo "🩸 bloodBender Development Environment"
            echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
            echo ""
            echo "Modules available:"
            echo "  • bloodBath   - Data synchronization & processing"
            echo "  • bloodTwin   - LSTM ML prediction model"
            echo "  • bareMetalBender - C++ glucose dynamics solver"
            echo ""
            echo "Quick start:"
            echo "  python -m bloodBath status"
            echo "  python bloodTwin/pipelines/train_lstm.py"
            echo "  cd bareMetalBender && make"
            echo ""
            echo "Python: $(python --version)"
            echo "PyTorch: $(python -c 'import torch; print(f"v{torch.__version__}")')"
            echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"
            echo ""
            
            # Set up Python path to include project root
            export PYTHONPATH="${toString ./.}:$PYTHONPATH"
            
            # Project-specific environment variables
            export BLOODBATH_ROOT="${toString ./.}/bloodBath"
            export BLOODTWIN_ROOT="${toString ./.}/bloodTwin"
            export BAREMETALBENDER_ROOT="${toString ./.}/bareMetalBender"
            
            # Load .env if it exists
            if [ -f .env ]; then
              echo "Loading environment from .env file..."
              set -a
              source .env
              set +a
            else
              echo "⚠️  No .env file found. Copy .env.example to .env and configure."
            fi
            
            echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
          '';

          # Environment variables
          PYTHONPATH = ".";
          
          # Prevent Python from writing bytecode
          PYTHONDONTWRITEBYTECODE = "1";
          
          # Force colored output
          FORCE_COLOR = "1";
        };

        # Python-only shell (no CUDA, lighter)
        pythonOnlyShell = pkgs.mkShell {
          name = "bloodBender-python";
          buildInputs = [ pythonEnv ] ++ devTools;
          
          shellHook = ''
            echo "🐍 bloodBender Python Environment (CPU-only)"
            export PYTHONPATH="${toString ./.}:$PYTHONPATH"
          '';
        };

        # C++ only shell for bareMetalBender development
        cppOnlyShell = pkgs.mkShell {
          name = "bareMetalBender-dev";
          buildInputs = cppBuildInputs ++ devTools;
          
          shellHook = ''
            echo "⚙️  bareMetalBender C++ Development"
            echo "Run: cd bareMetalBender && make"
          '';
        };

      in
      {
        # Development shells
        devShells = {
          default = devShell;
          python = pythonOnlyShell;
          cpp = cppOnlyShell;
        };

        # Packages
        packages = {
          # bloodBath Python package
          bloodBath = pythonPackages.buildPythonPackage {
            pname = "bloodBath";
            version = "2.0.0";
            src = ./bloodBath;
            
            propagatedBuildInputs = with pythonPackages; [
              numpy
              pandas
              arrow
              requests
              pytz
            ];
            
            # Skip tests during build (run separately)
            doCheck = false;
            
            meta = {
              description = "Tandem t:connect pump data synchronization and processing";
              license = pkgs.lib.licenses.mit;
            };
          };

          # bareMetalBender C++ executable
          bareMetalBender = pkgs.stdenv.mkDerivation {
            pname = "bareMetalBender";
            version = "1.0.0";
            src = ./bareMetalBender;
            
            buildInputs = cppBuildInputs;
            
            buildPhase = ''
              make
            '';
            
            installPhase = ''
              mkdir -p $out/bin
              cp ivp $out/bin/bareMetalBender
            '';
            
            meta = {
              description = "C++ glucose dynamics IVP solver";
              license = pkgs.lib.licenses.mit;
            };
          };
        };

        # Apps (runnable commands)
        apps = {
          bloodBath = {
            type = "app";
            program = toString (pkgs.writeShellScript "bloodBath" ''
              cd ${toString ./.}
              ${pythonEnv}/bin/python -m bloodBath "$@"
            '');
          };

          sync = {
            type = "app";
            program = toString (pkgs.writeShellScript "sync" ''
              cd ${toString ./.}
              ${pythonEnv}/bin/python -m bloodBath.cli.main sync "$@"
            '');
          };
          
          trainLSTM = {
            type = "app";
            program = toString (pkgs.writeShellScript "train-lstm" ''
              cd ${toString ./.}
              ${pythonEnv}/bin/python bloodTwin/pipelines/train_lstm.py "$@"
            '');
          };
        };

        # Formatter for `nix fmt`
        formatter = pkgs.nixpkgs-fmt;
      }
    );
}
