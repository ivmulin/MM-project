{ pkgs ? import <nixpkgs> {} }:

pkgs.mkShell {
  buildInputs = with pkgs; [

    # keep this line if you use bash
    bashInteractive
  ];
shellHook = ''
    # Make sure the GR package is installed in the current project
    julia -e 'using Pkg; Pkg.activate("./"); Pkg.add("GR")'

  # Patch the GKS binary for GR
     sudo patchelf \
     --set-interpreter ${with pkgs; pkgs.glibc}/lib/ld-linux-x86-64.so.2 \
     --set-rpath "${with pkgs; lib.makeLibraryPath [
    gcc9 
    stdenv.cc.cc.lib
  ]}" \
     /home/horhik/.julia/artifacts/13488323454a8a92411d5d627bb0f85b9b4c7006/bin/gksqt
'';
}
