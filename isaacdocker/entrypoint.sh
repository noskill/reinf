#!/bin/bash
Xorg -noreset +extension GLX +extension RANDR +extension RENDER -logfile /tmp/xorg.log -config /etc/X11/xorg.conf :1 &
sleep 1
openbox &
export DISPLAY=:1
cd /workspace/isaac-lab
conda activate isaac
exec "$@"
