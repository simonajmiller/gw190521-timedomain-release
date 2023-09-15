#!/bin/bash

# Download and unzip 
curl https://zenodo.org/record/8349403/files/gw190521_TD_PE.zip --output "gw190521_TD_PE.zip"
unzip gw190521_TD_PE.zip

# Remove zip and Mac OSX files
rm gw190521_TD_PE.zip
rmdir __MACOSX/
