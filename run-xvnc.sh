#!/bin/bash
# --------------------------------------------------------------------
# Defaults
DEFAULT_PASSWORD="mypassword123"
DEFAULT_DISPLAY_NUM=15
DEFAULT_RFBPORT=5900
DEFAULT_COMMAND="./isaaclab.sh -p ./my/issaclab/create_scene.py"
# --------------------------------------------------------------------
usage() {
  echo "Usage: $0 [-p password] [-d display_number] [-r rfbport] [-c command] [-- command...]"
  echo "  -p : VNC password           (default: $DEFAULT_PASSWORD)"
  echo "  -d : Display number         (default: $DEFAULT_DISPLAY_NUM)"
  echo "  -r : RFB port               (default: $DEFAULT_RFBPORT)"
  echo "  -c : Command to run         (default: $DEFAULT_COMMAND)"
  exit 1
}
# --------------------------------------------------------------------
# Parse options
# Parse short options first. Everything after "--" (if present) is treated
# as the command to run so the user doesn't have to quote it.

while getopts "p:d:r:c:h" opt; do
  case "$opt" in
    p) PASSWORD="$OPTARG" ;;
    d) DISPLAY_NUM="$OPTARG" ;;
    r) RFBPORT="$OPTARG" ;;
    c) COMMAND="$OPTARG" ;;
    h|?) usage ;;
  esac
done
# Shift off the processed options so "$@" now contains the remaining args.
shift $((OPTIND - 1))

# If there is anything left, treat it as the command (allows unquoted usage
# like:  run-xvnc.sh -p mypass -- python3 visualize.py arg1).
if [ "$#" -gt 0 ]; then
  COMMAND="$*"
fi
# --------------------------------------------------------------------
# Fill in defaults
PASSWORD=${PASSWORD:-$DEFAULT_PASSWORD}
DISPLAY_NUM=${DISPLAY_NUM:-$DEFAULT_DISPLAY_NUM}
RFBPORT=${RFBPORT:-$DEFAULT_RFBPORT}
COMMAND=${COMMAND:-$DEFAULT_COMMAND}
# Make the command visible to the xstartup script that will be
# executed by the VNC session.
export COMMAND

# --------------------------------------------------------------------
# If the user supplied just a Python script path (e.g. foo.py) we helpfully
# prefix it with the interpreter so they don't have to type "python3 foo.py".

if [[ "$COMMAND" =~ ^[^[:space:]]+\.py( .*)?$ ]]; then
  # Extract script name (first word).
  _script=${COMMAND%% *}
  if [ -f "$_script" ] && [ ! -x "$_script" ]; then
    COMMAND="python3 $COMMAND"
  fi
fi
# --------------------------------------------------------------------
# Validation
[[ "$DISPLAY_NUM" =~ ^[0-9]+$ ]] || { echo "Display number must be integer"; exit 1; }
if ! [[ "$RFBPORT" =~ ^[0-9]+$ ]] || (( RFBPORT < 1024 || RFBPORT > 65535 )); then
  echo "RFB port must be 1024-65535"; exit 1
fi
# --------------------------------------------------------------------
RESOLUTION="1920x1080"
VNC_DIR="$HOME/.vnc_isaaclab"
mkdir -p "$VNC_DIR"
# --------------------------------------------------------------------
echo "Setting up VNC password..."
vncpasswd -f > "$VNC_DIR/passwd" <<EOF
$PASSWORD
$PASSWORD
EOF
chmod 600 "$VNC_DIR/passwd"
# --------------------------------------------------------------------
# xstartup with COMMAND in foreground
# Create an xstartup script that:
#   1. starts a lightweight window-manager in the background
#   2. runs the user-supplied command in the foreground
#   3. stops the window-manager when the command finishes so that
#      the X session – and therefore the VNC server – terminates.
# This guarantees that the exit status of the original command is
# returned from the top-level `vncserver -fg` process.

cat > "$VNC_DIR/xstartup" <<'EOF'
#!/bin/sh

# Start a simple window manager (or fall back to twm/xterm if not available).
# Launch a lightweight window-manager if available.
if command -v metacity >/dev/null 2>&1; then
  metacity --replace 2>/dev/null &
elif command -v openbox >/dev/null 2>&1; then
  openbox 2>/dev/null &
else
  twm 2>/dev/null &
fi
WM_PID=$!

# Nothing else to do inside the VNC session for now.
wait "$WM_PID" 2>/dev/null || true
exit 0
EOF
chmod +x "$VNC_DIR/xstartup"
# --------------------------------------------------------------------
echo "Cleaning up old VNC sessions..."
vncserver -kill ":$DISPLAY_NUM" 2>/dev/null || true
# --------------------------------------------------------------------
echo "Starting VNC server..."
# Start VNC server (daemonises).
vncserver ":$DISPLAY_NUM" -geometry "$RESOLUTION" -depth 24 \
         -rfbport "$RFBPORT" \
         -SecurityTypes VncAuth \
         -PasswordFile "$VNC_DIR/passwd" \
         -xstartup "$VNC_DIR/xstartup"
# --------------------------------------------------------------------
# --------------------------------------------------------------------
# If the server started successfully, run the user command in the
# current shell so that it stays in the foreground for whoever
# invoked `run-xvnc.sh`.
# When the command exits we ensure that the VNC session is cleaned
# up before propagating the command's exit status.

if [ $? -eq 0 ]; then
  echo "VNC server started on display :$DISPLAY_NUM ($RFBPORT)."
  echo "Connect with a VNC client (password: $PASSWORD)."

  # Export DISPLAY so the child command can open windows.
  export DISPLAY=":$DISPLAY_NUM"

  # Run the user command in the foreground.
  echo "Running: $COMMAND"
  eval "$COMMAND"
  CMD_STATUS=$?

  # Tear down the VNC session.
  echo "Command finished (status=$CMD_STATUS).  Stopping VNC server..."
  vncserver -kill ":$DISPLAY_NUM" 2>/dev/null || true

  exit "$CMD_STATUS"
else
  echo "Failed to start VNC server. Check \$HOME/.vnc/*.log"
  exit 1
fi
