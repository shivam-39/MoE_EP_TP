#!/usr/bin/env bash
# Download OpenMPI with curl and install it under .deps/openmpi (no sudo, no brew).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
VERSION="${OPENMPI_VERSION:-4.1.8}"
PREFIX="$ROOT/.deps/openmpi"
SRC_DIR="$ROOT/.deps/src"
TARBALL="openmpi-${VERSION}.tar.gz"
# v4.1.8 -> v4.1
MAJOR_MINOR="${VERSION%.*}"
URL="${OPENMPI_URL:-https://download.open-mpi.org/release/open-mpi/v${MAJOR_MINOR}/${TARBALL}}"
JOBS="$(getconf _NPROCESSORS_ONLN 2>/dev/null || echo 4)"

mkdir -p "$SRC_DIR"
cd "$SRC_DIR"

if [[ ! -f "$TARBALL" ]]; then
  echo "Downloading $URL"
  curl -fL --retry 3 --retry-delay 2 -o "$TARBALL" "$URL"
else
  echo "Using existing $SRC_DIR/$TARBALL"
fi

rm -rf "openmpi-${VERSION}"
tar xzf "$TARBALL"
cd "openmpi-${VERSION}"

./configure \
  --prefix="$PREFIX" \
  --disable-mpi-fortran \
  --enable-orterun-prefix-by-default

make -j"$JOBS"
make install

echo
echo "OpenMPI ${VERSION} installed to ${PREFIX}"
echo "Activate it with:  source scripts/env.sh"
