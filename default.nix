{ pkgs ? import <nixpkgs> {
    config = {
      allowUnfree = true;
      cudaSupport = builtins.currentSystem == "x86_64-linux";
    };
  }
}:

let
  libheif-new = pkgs.libheif.overrideAttrs (old: rec {
    version = "1.17.6";
    src = pkgs.fetchFromGitHub {
      owner = "strukturag";
      repo = "libheif";
      rev = "v${version}";
      sha256 = "sha256-pp+PjV/pfExLqzFE61mxliOtVAYOePh1+i1pwZxDLAM=";
    };
    nativeBuildInputs = [ pkgs.cmake pkgs.pkg-config ];
  });

  libjxl-new = pkgs.libjxl.overrideAttrs (old: rec {
    version = "0.9.2";
    src = pkgs.fetchFromGitHub {
      owner = "libjxl";
      repo = "libjxl";
      rev = "v${version}";
      sha256 = "sha256-t5cuuvqGC3os49lhV8hixCn6xqCoDppPC7nwf3f8Rkw=";
      fetchSubmodules = true;
    };
    patches = [ ];
  });


  python = pkgs.python3.override {
    packageOverrides = self: super: {
      imagecodecs =
        super.buildPythonPackage rec {
          pname = "imagecodecs";
          version = "2024.1.1";
          format = "setuptools";
          src = super.fetchPypi {
            inherit pname version;
            hash = "sha256-/eRr1pjQCCVd7vVBHFmzXA6HUpXoNb9geffiqyLyFus=";
          };
          patches = [ ./imagecodecs-deps-nix.patch ];
          patchFlags = [ "--binary" ];
          doCheck = false;
          nativeBuildInputs = [
            pkgs.pkg-config
            super.cython_3
            super.pip
          ];
          buildInputs = with pkgs; [
            brotli
            brunsli
            bzip2
            c-blosc
            charls
            giflib
            jxrlib
            lcms
            libaec
            libdeflate
            libheif-new
            libjxl-new
            libpng
            libtiff
            libwebp
            lz4
            snappy
            xz
            zlib
            zopfli
            zstd
          ];
          propagatedBuildInputs = with super; [
            py
            setuptools
            numcodecs
          ];
        };

      orbax-checkpoint =
        super.buildPythonPackage rec {
          pname = "orbax_checkpoint";
          version = "0.5.3";
          pyproject = true;
          src = super.fetchPypi {
            inherit pname version;
            hash = "sha256-FXKQTLv+hROSfg2A+AtzDg7y9oAzLTwoENhENTKTi0U=";
          };
          nativeBuildInputs = with super; [
            flit
          ];
          propagatedBuildInputs = with super; [
            absl-py
            etils
            jax
            jaxlib
            msgpack
            nest-asyncio
            numpy
            pyyaml
            protobuf
            py
            tensorstore
            typing-extensions
          ];
        };

      flax = super.flax.overridePythonAttrs (old: {
        doCheck = false;
        propagatedBuildInputs =
          old.propagatedBuildInputs ++ [ self.orbax-checkpoint ];
      });
    };
  };

in
python.withPackages (p: with p; [
  imagecodecs
  lmdb
  numcodecs
  numpy
  opencv4
  zarr

  # Torch
  torch-bin
  torchvision-bin

  # JAX
  flax
  jax
  jaxlib
  jaxtyping
  optax
  orbax-checkpoint
])
