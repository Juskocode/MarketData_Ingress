#!/usr/bin/env bash
set -euo pipefail

TAG="${1:-}"
if [[ -z "${TAG}" ]]; then
  echo "Usage: $0 <vMAJOR.MINOR.PATCH>" >&2
  exit 1
fi

VERSION="$(sed -nE 's/^project\(MarketData_Ingress VERSION ([0-9]+\.[0-9]+\.[0-9]+).*$/\1/p' CMakeLists.txt)"
if [[ -z "${VERSION}" ]]; then
  echo "Could not read project version from CMakeLists.txt" >&2
  exit 1
fi
if [[ "${TAG}" != "v${VERSION}" ]]; then
  echo "Release tag ${TAG} does not match project version v${VERSION}" >&2
  exit 1
fi

echo "[release-tag] ${TAG} matches project version"
