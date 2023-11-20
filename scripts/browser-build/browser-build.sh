#!/bin/bash

rm -rf temp/*

# make sure dependency wheels are available
pushd wheels
bash get-wheels.sh
popd

npm install
npm run build