# log

- touch preprocessing.sh which run all commands
- create channel_merge.py as CLI
  - add multi CPU support ("-n")
  - add merge channel option ("-m") so user can decide when to merge red and yellow channel
- create cellpose_seg.py as CLI
  - change channel back to [1, 3]
- create crop_img.py as CLI
  - add multi CPU support ("-n")
  - make mask-encoding an option ("--opcodedir")
  - make max cell count output an option ("--max_cell_count")
  - remove min frame size limit