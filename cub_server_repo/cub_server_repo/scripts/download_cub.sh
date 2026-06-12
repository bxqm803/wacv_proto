#!/usr/bin/env bash
set -euo pipefail

DATA_PARENT="${1:-./data}"
ARCHIVE="${DATA_PARENT}/CUB_200_2011.tgz"
TARGET="${DATA_PARENT}/CUB_200_2011"
URL="https://data.caltech.edu/records/65de6-vp158/files/CUB_200_2011.tgz?download=1"
MD5="97eceeb196236b17998738112f37df78"

mkdir -p "${DATA_PARENT}"

if [[ -f "${TARGET}/images.txt" && -d "${TARGET}/images" ]]; then
  echo "CUB already exists: ${TARGET}"
  exit 0
fi

if [[ ! -f "${ARCHIVE}" ]]; then
  echo "Downloading CUB-200-2011..."
  if command -v curl >/dev/null 2>&1; then
    curl -L --fail --retry 5 --retry-delay 3 -o "${ARCHIVE}" "${URL}"
  elif command -v wget >/dev/null 2>&1; then
    wget --tries=5 -O "${ARCHIVE}" "${URL}"
  else
    echo "Neither curl nor wget is available." >&2
    exit 1
  fi
fi

echo "${MD5}  ${ARCHIVE}" | md5sum -c -
tar -xzf "${ARCHIVE}" -C "${DATA_PARENT}"

test -f "${TARGET}/images.txt"
test -d "${TARGET}/images"
echo "CUB ready: ${TARGET}"
