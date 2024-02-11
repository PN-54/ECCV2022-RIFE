{ pkgs ? import <nixpkgs> {
    config = {
      allowUnfree = true;
      cudaSupport = builtins.currentSystem == "x86_64-linux";
    };
  }
}:
pkgs.mkShell {
  packages = [
    (import ./default.nix { inherit pkgs; })
  ];
}
