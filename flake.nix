{
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";

    flake-parts.url = "github:hercules-ci/flake-parts";
    flake-parts.inputs.nixpkgs-lib.follows = "nixpkgs";

    dream2nix.url = "github:nix-community/dream2nix";
    dream2nix.inputs.nixpkgs.follows = "nixpkgs";
  };

  outputs = { self, nixpkgs, flake-parts, dream2nix } @ inputs:
    flake-parts.lib.mkFlake { inherit inputs; } ({ ... }: {
      systems = [ "x86_64-linux" "aarch64-linux" "x86_64-darwin" "aarch64-darwin" ];

      perSystem = { config, system, lib, pkgs', ... }:
        let
          cudaLibs = lib.optionals pkgs'.stdenv.hostPlatform.isLinux
            (with pkgs'.cudaPackages; [ cudnn_9_3 ]);
        in
        rec {
          _module.args.pkgs' = import nixpkgs {
            inherit system;
            config = { allowUnfree = true; };
          };

          packages = {
            simple-alpha-zero = dream2nix.lib.evalModules {
              packageSets.nixpkgs = pkgs';
              modules = [
                ./nix/simple-alpha-zero.nix
                {
                  paths.projectRoot = ./.;
                  paths.projectRootFile = "flake.nix";
                  paths.package = ./.;
                }
              ];
            };
          };

          devShells.default = pkgs'.mkShell {
            name = "simple-alpha-zero-dev";

            inputsFrom = [ packages.simple-alpha-zero.devShell ];
            buildInputs = cudaLibs;

            shellHook = ''
              export MPLCONFIGDIR=$(pwd)/.matplotlib
              export LD_LIBRARY_PATH="${lib.makeSearchPath "lib"
                (lib.map (p: p.lib) cudaLibs)}";

              if [ -n "$WSLPATH" ]; then
                export LD_LIBRARY_PATH=/usr/lib/wsl/lib:$LD_LIBRARY_PATH
              fi
            '';
          };
        };
      # end of perSystem
    });
}
