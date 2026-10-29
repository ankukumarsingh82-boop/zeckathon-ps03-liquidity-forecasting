#!/bin/bash
# Scan official RBI press releases for "Money Market Operations as on <date>" (Sep 18 - Sep 30, 2026 gap fill)
cd /workspace/zeckathon-ps03/data/raw/rbi_mmo
for p in $(seq 63599 63760); do
  curl -s -m 25 -A "Mozilla/5.0" "https://www.rbi.org.in/scripts/BS_PressReleaseDisplay.aspx?prid=$p" -o tmp_$p.html
  t=$(grep -oE 'Money Market Operations as on [A-Za-z]+ [0-9]+, 20[0-9]+' tmp_$p.html | head -1)
  if [ -n "$t" ]; then mv tmp_$p.html prid_$p.html; echo "$p|$t|$(date -u +%FT%TZ)" >> index.txt; else rm -f tmp_$p.html; fi
done
echo DONE >> index.txt
