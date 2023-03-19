#!/usr/bin/env bash

#
# Setup ASDF and Plugins
#
# @example
#
#    ASDF_PLUGIN_URL_nodejs=https://someurl \
#      ./setup.bash
#

set -e

[ $(command -v git) ] || {
  echo "==> 📛📦 Missing git"
  exit 1
}
[ $(command -v curl) ] || {
  echo "==> 📛📦 Missing curl"
  exit 1
}

# Local vars
ASDF_VERSION=${ASDF_VERSION:-v0.9.0}
ASDF_HOME=$HOME/.asdf
ASDF_BIN=$ASDF_HOME/asdf.sh

##
# Helper to idempotently add strings to target files
#
append_uniquely() {
  # this is a more correct way to test for presence of strings in files
  # and then act on the outcome.
  cat "${1}" | grep -zqv "${2}" && {
    echo "==> ✍ Writing \"$2\" into \"$1\" "
    echo "${2}" >>$1
  } || {
    echo "==> ✅ '${2}' already exists in ${1}"
  }
}

get_processor() {
  [[ $(sysctl -e -n machdep.cpu.brand_string) =~ "Apple" ]] && {
    echo 'M1'
  } || uname -m
}
get_platform() {
  uname -s
}

case "${SHELL}" in
/bin/bash)
  SHELL_PROFILE=~/.bashrc
  SHELL_NAME="bash"
  ;;
/bin/zsh)
  SHELL_PROFILE=~/.zshrc
  SHELL_NAME="zsh"
  ;;
esac

echo "==> 💁 [ASDF] install with plugins"

if [ ! -f "$ASDF_BIN" ]; then
  echo "===> ⤵️ ASDF not detected ... installing"
  git clone https://github.com/asdf-vm/asdf.git "$ASDF_HOME" --branch $ASDF_VERSION
  [ ! command -v asdf ] &>/dev/null && {
    echo "====> ⚕️ adding to shell profile"
    append_uniquely "$SHELL_PROFILE" ". $ASDF_HOME/asdf.sh"
    append_uniquely "$SHELL_PROFILE" ". $ASDF_HOME/completions/asdf.bash"
  }
fi

source "$ASDF_BIN"

for plugin in $(cut -d' ' -f1 ./.tool-versions); do
  echo "==> 💁 [ASDF] Ensure ${plugin} plugin"
  if [ -d "$ASDF_HOME/plugins/${plugin}" ]; then
    echo "===> 📦 attempting upgrade"
    asdf plugin-update "${plugin}"
  else
    echo "===> ⤵️ installing"
    plugin_url_var=ASDF_PLUGIN_URL_${plugin//-/_}
    plugin_url="${!plugin_url_var}"

    if [ ${!plugin_url_var+x} ]; then
      echo "====> 💁 [${plugin}] installed from ${plugin_url}"
    fi

    asdf plugin-add "${plugin}" "${plugin_url}"
  fi
done

PROCESSOR=$(get_processor)

echo "==> 💁 [ASDF] install tools"
case $PROCESSOR in
M1)
  echo "===> 💁 [ASDF] install for M1"
  RUBY_CFLAGS=-DUSE_FFI_CLOSURE_ALLOC \
    arch -arm64 \
    asdf install
  ;;

x86_64)
  echo "===> 💁 [ASDF] install for x86_64"
  asdf install
  ;;
esac

echo "==> 💁 [ASDF] reshim globals"
asdf reshim

echo "==> 💁 [ASDF] Done ✅"
