#!/usr/bin/env bash
cd ~/Desktop
echo "$PWD"
cd Builds
echo "$PWD"
make
cd bin
unset GTK_PATH   
./ipaproject