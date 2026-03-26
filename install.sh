#!/usr/bin/env bash
set -euo pipefail

umask 077

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${ROOT_DIR}/.venv"
ENV_FILE="${ROOT_DIR}/.env"
ENV_EXAMPLE="${ROOT_DIR}/.env.example"
PROXIES_EXAMPLE="${ROOT_DIR}/proxies.example.txt"
PROXIES_FILE="${ROOT_DIR}/proxies.txt"
PYTHON_BIN=""

info() {
  printf '[install] %s\n' "$*"
}

warn() {
  printf '[install] warning: %s\n' "$*" >&2
}

die() {
  printf '[install] error: %s\n' "$*" >&2
  exit 1
}

require_linux() {
  if [[ "$(uname -s)" != "Linux" ]]; then
    die "This installer is supported on Linux only."
  fi
}

find_python() {
  if command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="$(command -v python3)"
    return
  fi
  if command -v python >/dev/null 2>&1; then
    PYTHON_BIN="$(command -v python)"
    return
  fi
  die "python3 is required but was not found."
}

ensure_venv() {
  if [[ ! -d "${VENV_DIR}" ]]; then
    info "Creating virtual environment at ${VENV_DIR}"
    "${PYTHON_BIN}" -m venv "${VENV_DIR}"
  fi
}

install_requirements() {
  info "Installing runtime dependencies"
  "${VENV_DIR}/bin/python" -m pip install --upgrade pip
  "${VENV_DIR}/bin/python" -m pip install -r "${ROOT_DIR}/requirements.txt"
}

prompt_choice() {
  local prompt_text="$1"
  shift
  local -a choices=("$@")
  local selection=""

  while [[ -z "${selection}" ]]; do
    printf '\n%s\n' "${prompt_text}"
    local i=1
    for choice in "${choices[@]}"; do
      printf '  %s) %s\n' "${i}" "${choice}"
      i=$((i + 1))
    done
    read -r -p "> " selection
    if [[ "${selection}" =~ ^[0-9]+$ ]] && [[ "${selection}" -ge 1 ]] && [[ "${selection}" -le ${#choices[@]} ]]; then
      printf '%s\n' "${choices[$((selection - 1))]}"
      return
    fi
    selection=""
    warn "Invalid choice, try again."
  done
}

prompt_yes_no() {
  local prompt_text="$1"
  local default_value="${2:-y}"
  local suffix="[Y/n]"
  if [[ "${default_value}" == "n" ]]; then
    suffix="[y/N]"
  fi

  while true; do
    read -r -p "${prompt_text} ${suffix} " answer
    answer="${answer:-${default_value}}"
    case "${answer}" in
      y|Y|yes|YES) printf 'yes\n'; return ;;
      n|N|no|NO) printf 'no\n'; return ;;
    esac
    warn "Please answer yes or no."
  done
}

prompt_text() {
  local prompt_text="$1"
  local default_value="${2:-}"
  local answer=""
  if [[ -n "${default_value}" ]]; then
    read -r -p "${prompt_text} [${default_value}] " answer
    printf '%s\n' "${answer:-${default_value}}"
    return
  fi
  read -r -p "${prompt_text} " answer
  printf '%s\n' "${answer}"
}

write_env() {
  local env_file_path="$1"
  shift

  if [[ ! -f "${ENV_EXAMPLE}" ]]; then
    die ".env.example was not found."
  fi

  if [[ ! -f "${env_file_path}" ]]; then
    cp "${ENV_EXAMPLE}" "${env_file_path}"
  fi

  "${PYTHON_BIN}" - "$env_file_path" "$@" <<'PY'
import re
import sys
from pathlib import Path

path = Path(sys.argv[1])
updates = {}
for item in sys.argv[2:]:
    key, value = item.split("=", 1)
    updates[key] = value

lines = path.read_text().splitlines()
used = set()
output = []
pattern_cache = {}

for line in lines:
    replaced = False
    for key, value in updates.items():
        pattern = pattern_cache.get(key)
        if pattern is None:
            pattern = re.compile(rf"^\s*#?\s*{re.escape(key)}=.*$")
            pattern_cache[key] = pattern
        if pattern.match(line):
            output.append(f"{key}={value}")
            used.add(key)
            replaced = True
            break
    if not replaced:
        output.append(line)

for key, value in updates.items():
    if key not in used:
        output.append(f"{key}={value}")

path.write_text("\n".join(output) + "\n")
PY
}

ensure_proxies() {
  if [[ -f "${PROXIES_FILE}" ]]; then
    return
  fi
  if [[ -f "${PROXIES_EXAMPLE}" ]]; then
    cp "${PROXIES_EXAMPLE}" "${PROXIES_FILE}"
  fi
}

cuda_available() {
  if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi >/dev/null 2>&1 && return 0
  fi
  return 1
}

normalize_model_path() {
  local candidate="$1"
  if [[ -n "${candidate}" && "${candidate}" != /* ]]; then
    printf '%s\n' "${ROOT_DIR}/${candidate}"
    return
  fi
  printf '%s\n' "${candidate}"
}

require_linux
find_python

info "Select system profile"
PROFILE="$(prompt_choice "Choose a profile:" "Low" "Medium" "High")"
info "Selected profile: ${PROFILE}"

info "Select accelerator"
ACCELERATOR="$(prompt_choice "Choose an accelerator:" "CUDA" "CPU")"
info "Selected accelerator: ${ACCELERATOR}"

if [[ "${ACCELERATOR}" == "CUDA" ]] && ! cuda_available; then
  warn "CUDA was selected, but no NVIDIA runtime was detected."
  answer="$(prompt_yes_no "Continue in CPU mode instead?" "y")"
  if [[ "${answer}" == "yes" ]]; then
    ACCELERATOR="CPU"
  else
    die "CUDA setup is required for the selected mode."
  fi
fi

DEEPSEEK_API_KEY="$(prompt_text "Enter your DeepSeek API key:")"
if [[ -z "${DEEPSEEK_API_KEY}" ]]; then
  die "DEEPSEEK_API_KEY cannot be empty."
fi

RAG_ENABLED_VALUE="false"
VISION_ENABLED_VALUE="false"
BGE_BATCH_SIZE="8"
BGE_DEVICE="cpu"
BGE_PRELOAD="false"
QWEN_MODEL_PATH=""
QWEN_PRELOAD="false"
QWEN_LOAD_IN_4BIT="false"
QWEN_DTYPE="float16"

case "${PROFILE}" in
  Low)
    RAG_ENABLED_VALUE="false"
    VISION_ENABLED_VALUE="false"
    BGE_BATCH_SIZE="8"
    ;;
  Medium)
    RAG_ENABLED_VALUE="true"
    VISION_ENABLED_VALUE="false"
    BGE_BATCH_SIZE="16"
    ;;
  High)
    RAG_ENABLED_VALUE="true"
    VISION_ENABLED_VALUE="true"
    BGE_BATCH_SIZE="32"
    ;;
  *)
    die "Unknown profile: ${PROFILE}"
    ;;
esac

if [[ "${ACCELERATOR}" == "CUDA" ]]; then
  BGE_DEVICE="cuda"
  BGE_PRELOAD="true"
  if [[ "${RAG_ENABLED_VALUE}" == "false" ]]; then
    BGE_PRELOAD="false"
  fi
  if [[ "${VISION_ENABLED_VALUE}" == "true" ]]; then
    QWEN_PRELOAD="true"
    QWEN_LOAD_IN_4BIT="true"
    QWEN_DTYPE="float16"
    DEFAULT_QWEN_PATH="${ROOT_DIR}/models/vl"
    if [[ -d "${DEFAULT_QWEN_PATH}" ]]; then
      QWEN_MODEL_PATH="${DEFAULT_QWEN_PATH}"
    else
      QWEN_MODEL_PATH="$(prompt_text "Enter the local Qwen2.5-VL model directory:")"
      QWEN_MODEL_PATH="$(normalize_model_path "${QWEN_MODEL_PATH}")"
    fi
    if [[ -z "${QWEN_MODEL_PATH}" ]]; then
      die "QWEN_VL_MODEL_PATH is required when Vision is enabled."
    fi
  fi
else
  BGE_DEVICE="cpu"
  BGE_PRELOAD="false"
  RAG_ENABLED_VALUE="false"
  VISION_ENABLED_VALUE="false"
fi

if [[ "${VISION_ENABLED_VALUE}" == "false" ]]; then
  QWEN_MODEL_PATH=""
  QWEN_PRELOAD="false"
  QWEN_LOAD_IN_4BIT="false"
fi

write_env "${ENV_FILE}" \
  "DEEPSEEK_API_KEY=${DEEPSEEK_API_KEY}" \
  "RAG_ENABLED=${RAG_ENABLED_VALUE}" \
  "VISION_ENABLED=${VISION_ENABLED_VALUE}" \
  "BGE_M3_MODEL_PATH=BAAI/bge-m3" \
  "BGE_M3_DEVICE=${BGE_DEVICE}" \
  "BGE_M3_BATCH_SIZE=${BGE_BATCH_SIZE}" \
  "BGE_M3_PRELOAD=${BGE_PRELOAD}" \
  "QWEN_VL_MODEL_PATH=${QWEN_MODEL_PATH}" \
  "QWEN_VL_LOAD_IN_4BIT=${QWEN_LOAD_IN_4BIT}" \
  "QWEN_VL_TORCH_DTYPE=${QWEN_DTYPE}" \
  "QWEN_VL_PRELOAD=${QWEN_PRELOAD}"

ensure_proxies
ensure_venv
install_requirements

info "Installation summary"
printf '  profile: %s\n' "${PROFILE}"
printf '  accelerator: %s\n' "${ACCELERATOR}"
printf '  RAG_ENABLED: %s\n' "${RAG_ENABLED_VALUE}"
printf '  VISION_ENABLED: %s\n' "${VISION_ENABLED_VALUE}"
printf '  .env: %s\n' "${ENV_FILE}"
if [[ -n "${QWEN_MODEL_PATH}" ]]; then
  printf '  QWEN_VL_MODEL_PATH: %s\n' "${QWEN_MODEL_PATH}"
fi
printf '  virtualenv: %s\n' "${VENV_DIR}"

info "Next step: activate the virtual environment and run the app"
printf '  source "%s/bin/activate"\n' "${VENV_DIR}"
printf '  python app.py\n'
