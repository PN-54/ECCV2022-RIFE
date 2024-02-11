{ pkgs
, contents
, runscript ? "#!/bin/sh\nexec ${pkgs.hello}/bin/hello"
, startscript ? "#!/bin/sh\nexec ${pkgs.hello}/bin/hello"
}:
pkgs.runCommand "make-container" {} ''
  set -o pipefail
  closureInfo=${pkgs.closureInfo { rootPaths=contents ++ [pkgs.bashInteractive]; }}
  mkdir -p $out/r/{bin,lib,etc,dev,proc,sys,usr,.singularity.d/{actions,env,libs}}
  cd $out/r
  cp -na --parents $(cat $closureInfo/store-paths) .
  touch etc/{passwd,group}
  ln -s /bin /lib usr/
  ln -s ${pkgs.bashInteractive}/bin/bash bin/sh
  for p in ${pkgs.lib.concatStringsSep " " contents}; do
    ln -sn $p/bin/* bin/ || true
    ln -sn $p/lib/* lib/ || true
  done
  echo "${runscript}" >.singularity.d/runscript
  echo "${startscript}" >.singularity.d/startscript
  chmod +x .singularity.d/{runscript,startscript}
  cd $out
  ${pkgs.squashfsTools}/bin/mksquashfs r container.sqfs -no-hardlinks -all-root
''
