ps -A -ostat,ppid,pid,cmd | grep -e '^[zZ]'

# kill -HUP ppid
