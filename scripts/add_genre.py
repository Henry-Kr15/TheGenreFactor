#!/usr/bin/env python3
from get_genre_prototype import get_band_genre, clear_csv

filename = "missing_p136p101.csv"
clear_csv(filename)
get_band_genre("Gorillaz")
