#!/usr/bin/env python3

import pywikibot
from pywikibot import pagegenerators
import csv

def get_band_genre(band_name):
    site = pywikibot.Site('en', 'wikipedia')  # English Wikipedia
    page = pywikibot.Page(site, band_name)
    item = pywikibot.ItemPage.fromPage(page)  # this can be used for any page object


    try:
        # 'P136' is the property for 'Genre'
        item_dict = item.get()  # Get the item dictionary
        genres = item_dict['claims']['P136']
    except KeyError:
        print(f"Für {band_name} ist kein Genre-Tag unter P136 vorhanden")
        # csv für Künstler, bei denen P136 nicht vorhanden ist
        with open("missing_p136.csv", "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([band_name])
            genres = []  # Kein Genre-Tag vorhanden

    for genre in genres:
        target = genre.getTarget()
        print(target.labels['en'])

print("Prince:")
get_band_genre("Prince")
print("Metallica:")
get_band_genre("Metallica")
