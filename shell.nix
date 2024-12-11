let
  pkgs = import <nixpkgs> { };
in
pkgs.mkShell rec {
  packages = [
    pkgs.python312
    pkgs.pypy310
    pkgs.poetry
    pkgs.zlib
  ];
  shellHook = ''
    export LD_LIBRARY_PATH="${pkgs.lib.makeLibraryPath packages}:$LD_LIBRARY_PATH"
    export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:${pkgs.stdenv.cc.cc.lib}/lib:$LD_LIBRARY_PATH"
  '';
}
