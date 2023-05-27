#!/usr/bin/env python3

import pywikibot
from pywikibot import pagegenerators

def get_band_genre(band_name):
    site = pywikibot.Site('en', 'wikipedia')  # English Wikipedia
    page = pywikibot.Page(site, band_name)
    item = pywikibot.ItemPage.fromPage(page)  # this can be used for any page object

    # 'P136' is the property for 'Genre'
    item_dict = item.get()  # Get the item dictionary
    genres = item_dict['claims']['P136']

    for genre in genres:
        target = genre.getTarget()
        print(target.labels['en'])

print("Gorillaz:")
get_band_genre("Gorillaz")
print("Metallica:")
get_band_genre("Metallica")
