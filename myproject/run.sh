#!/usr/bin/env bash
cd ~/Desktop
echo "$PWD"
cd builds
echo "$PWD"
make
cd bin
unset GTK_PATH
./ipaproject
