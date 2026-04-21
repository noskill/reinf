#!/usr/bin/env bash
set -euo pipefail

XVNC_SCRIPT="${XVNC_SCRIPT:-../run-xvnc.sh}"
PASSWORD="${PASSWORD:-mypassword123}"
DISPLAY_NUM="${DISPLAY_NUM:-15}"
RFBPORT="${RFBPORT:-5900}"
TIMEOUT="${TIMEOUT:-0}"
DEVICE="${DEVICE:-cpu}"
PRINT_EVERY="${PRINT_EVERY:-20}"
SLEEP="${SLEEP:-0.03}"
EPISODES="${EPISODES:-0}"
CHECKPOINT="${CHECKPOINT:-}"
SOFTWARE_RENDER=1
DRY_RUN=0
EXTRA_ARGS=()
WM="${WM:-metacity-nocomposite}"
WM_SHIM_DIR=""

usage() {
  cat <<EOF
Usage: $0 --checkpoint <path> [options] [-- <extra wm_maze_play.py args>]

Options:
  --checkpoint PATH    Joint checkpoint path (required)
  --password VALUE     VNC password (default: ${PASSWORD})
  --display NUM        VNC display number (default: ${DISPLAY_NUM})
  --port NUM           VNC rfb port (default: ${RFBPORT})
  --timeout SEC        Command timeout for run-xvnc.sh (default: ${TIMEOUT})
  --device DEV         wm_maze_play.py device (default: ${DEVICE})
  --episodes N         Completed episodes target (0 = endless, default: ${EPISODES})
  --print-every N      Status print period (default: ${PRINT_EVERY})
  --sleep SEC          Sleep between env steps (default: ${SLEEP})
  --no-software-render Do not set SDL/LIBGL software-render env vars
  --wm NAME            VNC WM mode: metacity|metacity-nocomposite|none (default: ${WM})
  --dry-run            Print resolved command and exit
  -h, --help           Show this help

Example:
  $0 --checkpoint logs/runs/<run>/checkpoints/checkpoint_episode_400.pt --episodes 50
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --checkpoint) CHECKPOINT="$2"; shift 2 ;;
    --password) PASSWORD="$2"; shift 2 ;;
    --display) DISPLAY_NUM="$2"; shift 2 ;;
    --port) RFBPORT="$2"; shift 2 ;;
    --timeout) TIMEOUT="$2"; shift 2 ;;
    --device) DEVICE="$2"; shift 2 ;;
    --episodes) EPISODES="$2"; shift 2 ;;
    --print-every) PRINT_EVERY="$2"; shift 2 ;;
    --sleep) SLEEP="$2"; shift 2 ;;
    --no-software-render) SOFTWARE_RENDER=0; shift ;;
    --wm) WM="$2"; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help) usage; exit 0 ;;
    --) shift; EXTRA_ARGS+=("$@"); break ;;
    *) EXTRA_ARGS+=("$1"); shift ;;
  esac
done

if [[ -z "${CHECKPOINT}" ]]; then
  echo "Error: --checkpoint is required." >&2
  usage
  exit 1
fi

if [[ ! -f "${XVNC_SCRIPT}" ]]; then
  echo "Error: xvnc launcher not found at ${XVNC_SCRIPT}" >&2
  exit 1
fi

case "${WM}" in
  metacity|metacity-nocomposite|none) ;;
  *) echo "Error: --wm must be one of: metacity|metacity-nocomposite|none" >&2; exit 1 ;;
esac

PLAY_CMD=(
  python3 wm_maze_play.py
  --checkpoint "${CHECKPOINT}"
  --render
  --num-envs 1
  --device "${DEVICE}"
  --print-every "${PRINT_EVERY}"
  --sleep "${SLEEP}"
)

if [[ "${SOFTWARE_RENDER}" == "1" ]]; then
  PLAY_CMD=(
    env
    SDL_AUDIODRIVER=dummy
    SDL_RENDER_DRIVER=software
    LIBGL_ALWAYS_SOFTWARE=1
    __GLX_VENDOR_LIBRARY_NAME=mesa
    MESA_LOADER_DRIVER_OVERRIDE=llvmpipe
    SDL_FRAMEBUFFER_ACCELERATION=0
    "${PLAY_CMD[@]}"
  )
fi

if [[ "${EPISODES}" != "0" ]]; then
  PLAY_CMD+=(--episodes "${EPISODES}")
fi

PLAY_CMD+=("${EXTRA_ARGS[@]}")

PLAY_CMD_STR=""
for arg in "${PLAY_CMD[@]}"; do
  PLAY_CMD_STR+=" $(printf '%q' "${arg}")"
done
PLAY_CMD_STR="${PLAY_CMD_STR# }"

XVNC_CMD=(
  bash "${XVNC_SCRIPT}"
  -p "${PASSWORD}"
  -d "${DISPLAY_NUM}"
  -r "${RFBPORT}"
  -t "${TIMEOUT}"
  -- "${PLAY_CMD_STR}"
)

# ../run-xvnc.sh always prefers metacity first. In no-composite mode, override
# metacity with a shim that disables compositor (avoids GLX context path).
if [[ "${WM}" == "metacity-nocomposite" || "${WM}" == "none" ]]; then
  WM_SHIM_DIR="$(mktemp -d /tmp/wm_vnc_wmshim.XXXXXX)"
fi

if [[ "${WM}" == "metacity-nocomposite" ]]; then
  REAL_METACITY="$(command -v metacity || true)"
  if [[ -z "${REAL_METACITY}" ]]; then
    echo "Error: metacity not found in PATH." >&2
    exit 1
  fi
  cat > "${WM_SHIM_DIR}/metacity" <<EOF
#!/usr/bin/env bash
exec "${REAL_METACITY}" --no-composite "\$@"
EOF
  chmod +x "${WM_SHIM_DIR}/metacity"
  XVNC_CMD=(env PATH="${WM_SHIM_DIR}:$PATH" "${XVNC_CMD[@]}")
fi

if [[ "${WM}" == "none" ]]; then
  cat > "${WM_SHIM_DIR}/metacity" <<'EOF'
#!/usr/bin/env bash
exec sleep infinity
EOF
  chmod +x "${WM_SHIM_DIR}/metacity"
  XVNC_CMD=(env PATH="${WM_SHIM_DIR}:$PATH" "${XVNC_CMD[@]}")
fi

if [[ "${DRY_RUN}" == "1" ]]; then
  printf 'XVNC command:'
  printf ' %q' "${XVNC_CMD[@]}"
  printf '\n'
  exit 0
fi

cleanup() {
  if [[ -n "${WM_SHIM_DIR}" && -d "${WM_SHIM_DIR}" ]]; then
    rm -rf "${WM_SHIM_DIR}"
  fi
}
trap cleanup EXIT

printf 'Running:'
printf ' %q' "${XVNC_CMD[@]}"
printf '\n'
exec "${XVNC_CMD[@]}"
