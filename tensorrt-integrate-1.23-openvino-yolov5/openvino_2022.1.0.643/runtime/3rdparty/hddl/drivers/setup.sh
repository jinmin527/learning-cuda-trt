#! /bin/bash
# Copyright (C) 2019 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

DRIVER_DIR=`dirname "$(readlink -f "$0")"`
COMPONENTS="
drv_vsc
"
MAKEFLAGS=""

if [ $# -eq 2 ] && [ "$2" = "USE_SYSTEM_CONTIG_HEAP" ]; then
  MAKEFLAGS=$2"=ON"
fi

if [ "$1" = "install" ]; then
  TARGET=$1
elif [ "$1" = "uninstall" ]; then
  TARGET=$1
else
  echo "Unrecognezed argements. Please use"
  echo "    bash setup.sh install|uninstall"
  exit
fi

function install_component {
  local component=$1
  local target=$2
  local makeflags=$3

  echo "Running $target for component $component"
  cd $DRIVER_DIR/$component
  make $target $makeflags
}

vercomp () {
  if [[ $1 == $2 ]]
  then
    return 1
  fi
  # Internal Field Separator: split string to words.
  local IFS=.
  local i ver1=($1) ver2=($2)
  # fill empty fields in ver1 with zeros
  for ((i=${#ver1[@]}; i<${#ver2[@]}; i++))
  do
    ver1[i]=0
  done
  for ((i=0; i<${#ver1[@]}; i++))
  do
    if [[ -z ${ver2[i]} ]]
    then
      # fill empty fields in ver2 with zeros
      ver2[i]=0
    fi
    if ((10#${ver1[i]} > 10#${ver2[i]}))
    then
      return 2
    fi
    if ((10#${ver1[i]} < 10#${ver2[i]}))
    then
      return 0
    fi
  done
  return 1
}

# Judge whether to install drv_ion
if [[ $(lsb_release -si) == "Ubuntu" ]];
then
  cmp_ver="5.11.1"
  cur_ver=$(uname -r)
  cur_ver=${cur_ver%%-*} # Remove string after '-'
  vercomp $(uname -r) $cmp_ver	
  if [[ $? == 0 ]]; then
    install_component "drv_ion" $TARGET $MAKEFLAGS
  fi
fi

# Install others
for component in $COMPONENTS
do
  install_component $component $TARGET $MAKEFLAGS
done
