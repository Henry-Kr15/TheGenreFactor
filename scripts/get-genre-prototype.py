#!/usr/bin/env python3

import pywikibot
from pywikibot import pagegenerators
import csv

def get_band_genre(band_name):
    site = pywikibot.Site('en', 'wikipedia')  # English Wikipedia
    pages = site.search(band_name)  # Search for the band name and get a list of pages

    # page = pywikibot.Page(site, band_name)
    # item = pywikibot.ItemPage.fromPage(page)  # this can be used for any page object

    for page in pages:
        item = pywikibot.ItemPage.fromPage(page)  # Get the associated Wikidata item
        item_dict = item.get()  # Get the item dictionary

        # Check if the item is an instance of human (Q5) and has occupation (P106) as musician (Q639669)
        if ('P31' in item_dict['claims'] and any(claim.getTarget().getID() == 'Q5' for claim in item_dict['claims']['P31'])) and \
            ('P106' in item_dict['claims'] and any(claim.getTarget().getID() == 'Q639669' for claim in item_dict['claims']['P106'])):
            # The item is an instance of human and musician, so it's likely the band we're looking for
            try:
                # 'P136' is the property for 'Genre'
                genres = item_dict['claims']['P136']
            except KeyError:
                print(f"Für {band_name} ist kein Genre-Tag unter P136 vorhanden, probiere P101")
                try:
                    # "P101" ist die Eigenschaft "artists field of work"
                    genres = item_dict["claims"]["P101"]
                    continue
                except KeyError:
                    print(f"Für {band_name} ist auch kein Genre-Tag unter P101 vorhanden")
                    # csv für Künstler, bei denen P136 und P101 nicht vorhanden ist
                    with open("missing_p136p101.csv", "a", newline="") as f:
                        writer = csv.writer(f)
                        writer.writerow([band_name])
                        genres = []  # Kein Genre-Tag vorhanden
        else:
            print(f"{page.title()} ist keine Instanz eines Menschen und Künstlers")
            genres = []  # Kein Genre-Tag vorhanden

        for genre in genres:
            target = genre.getTarget()
            print(target.labels['en'])
        break


print("Prince:")
get_band_genre("Prince")
print("Metallica:")
get_band_genre("Metallica")
