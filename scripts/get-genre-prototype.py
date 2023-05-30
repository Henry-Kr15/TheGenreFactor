#!/usr/bin/env python3

import pywikibot
from pywikibot import pagegenerators
import csv

def get_band_genre(band_name):
    site = pywikibot.Site("en", "wikipedia") # Suche auf der englischen Wikipedia-Seite

    # Zuerst simpler Suchansatz auf der ersten Treffer-Seite
    page = pywikibot.Page(site, band_name)
    item = pywikibot.ItemPage.fromPage(page)

    # Erster Suchversuch nach dem Tag "P136"
    # TODO Detaillierterer Fehlerabfang (z.B. mit logging) könnte hier aufschlussreich sein
    try:
       item_dict = item.get() # Holt das item dictionary
       genres = item_dict["claims"]["P136"] # dictionary besitzt tiefe Struktur
    except KeyError:
        # Zweiter Suchversuch nach dem Tag "P136"
        pages = site.search(band_name)
        print(f"Erste Suche nach P136 auf der Topseite zu {band_name} fehlgeschlagen")

        for result in pages:
            item = pywikibot.ItemPage.fromPage(result) # Wikidata-Item von
            # der jew. Seite ziehen
            item_dict = item.get() # Holt wieder das item dictionary
            try:
                genres = item_dict["claims"]["P136"] # dictionary besitzt tiefe Struktur
            except KeyError:
                # Dritter Suchversuch, diesmal nach dem Tag "P106"
                # Es muss eine Instanz eines Menschen + Beruf des Künstlers gegeben
                # sein, sonst bedeutet "P106" etwas anderes
                # Instanz eines Menschen: Q5, Beruf des Künstlers: Q639669
                if ('P31' in item_dict['claims'] and any(claim.getTarget().getID() == 'Q5' for claim in item_dict['claims']['P31'])) and \
                   ('P106' in item_dict['claims'] and any(claim.getTarget().getID() == 'Q639669' for claim in item_dict['claims']['P106'])):
                   try:
                       genres = item_dict["claims"]["P101"]
                   except KeyError:
                       print(f"Für {band_name} ist auch kein Genre-Tag unter P101 vorhanden")
                       # csv für Künstler, bei denen P136 und P101 nicht vorhanden ist
                       # TODO in der fertigen Version sollte die Datei nicht jedes Mal neu geöffnet und geschlossen werden
                       with open("missing_p136p101.csv", "a", newline="") as f:
                           writer = csv.writer(f)
                           writer.writerow([band_name])

                       genres = []  # Kein Genre-Tag vorhanden
                       break
                else:
                    print(f"{page.title()} ist keine Instanz eines Menschen und Künstlers")
                    genres = []  # Kein Genre-Tag vorhanden

    for genre in genres:
        target = genre.getTarget()
        print(target.labels['en'])
        # TODO hinterher natürlich Rückgabe als String, anstatt es einfach nur zu printen

print("Prince:")
get_band_genre("Prince")
print("Metallica:")
get_band_genre("Metallica")
print("Gorillaz:")
get_band_genre("Gorillaz")
