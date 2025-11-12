#!/usr/bin/env zsh
PIDS=()
sleep 3 & PIDS+=($!)
sleep 5 & PIDS+=($!)
typeset -i TOTAL=${#PIDS}
status_tick() {
  typeset -i running=0 done=0
  for p in $PIDS; do
    if kill -0 $p 2>/dev/null; then ((running++)) else ((done++)); fi
  done
  print -r -- "$(date +'%H:%M:%S') running:${running} done:${done}/${TOTAL}"
}
while true; do
  status_tick
  (( done == TOTAL )) && break
  sleep 1
done
status_tick

