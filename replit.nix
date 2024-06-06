{pkgs}: {
  deps = [
    pkgs.ffmpeg-full
    pkgs.libxcrypt
    pkgs.pkg-config
    pkgs.openssl
    pkgs.rustc
    pkgs.libiconv
    pkgs.cargo
    pkgs.glibcLocales
    pkgs.cacert
  ];
}
