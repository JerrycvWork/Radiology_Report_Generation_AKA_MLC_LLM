#!/bin/bash

pushd data
for f in *xz; do
  echo "Recompressing ${f}"
  xz -dc $f | gzip > "${f/.xz/.gz}"
done
popd