{
  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs";
  };

  outputs = inputs @ { self, nixpkgs, flake-utils, ... }:
    flake-utils.lib.eachDefaultSystem (system: let
      pkgs = import nixpkgs {
        inherit system;
        allowUnfree = true;
        # cudaSupport = system == "x86_64-linux";
        cudaSupport = true;
      };
      python = import ./default.nix { inherit pkgs; };
      buildApptainer = import ./apptainer.nix;
    in {
      packages.default = python;
      packages.apptainer = buildApptainer {
        inherit pkgs;
        contents = [
          pkgs.coreutils
          pkgs.bashInteractive
          pkgs.stdenv.cc.libc
          python
        ];
      };
      devShells.default = import ./shell.nix { inherit pkgs; };
    });
}
