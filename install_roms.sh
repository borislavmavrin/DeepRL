#!/bin/bash
#wget http://www.atarimania.com/roms/Roms.rar
unrar x Roms.rar
python -m atari_py.import_roms ROMS