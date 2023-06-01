#!/usr/bin/env python3

import pywikibot
from pywikibot import pagegenerators
import csv
from pywikibot.exceptions import NoPageError


def get_band_genre(band_name):
#    name = [album_name, band_name]
    site = pywikibot.Site("en", "wikipedia")  # Suche auf der englischen Wikipedia-Seite
    i = 0
    genres = None

    # Schleife zur Suche in verschiedenen Artikeln
    print("erster Versuch")

    search_name = band_name
    page = pywikibot.Page(site, search_name)
    while genres is None:
        if i == 6:
            print(f"Nach {i} Durchläufen nichts gefunden")
            break
        
        if i == 1:
            search_name = band_name + " (album)"
            page = pywikibot.Page(site, search_name)
        if i == 2:
            search_name = band_name + " (music)"
            page = pywikibot.Page(site, search_name)
        if i == 3:
            search_name = band_name + " (band)"
            page = pywikibot.Page(site, search_name)

        i += 1
        # Erster Suchversuch nach dem Tag "P136"
        # TODO Detaillierterer Fehlerabfang (z.B. mit logging) könnte hier aufschlussreich sein
        try:
            item = pywikibot.ItemPage.fromPage(page)
            item_dict = item.get()  # Holt das item dictionary
            genres = item_dict["claims"]["P136"]  # dictionary besitzt tiefe Struktur
            print(
                f"Erste Suche nach P136 auf der {i}ten Seite zu {band_name}: {page} erfolgt"
            )
        except (KeyError, NoPageError):
            # Kein Genre-Tag gefunden oder Seite nicht vorhanden
            # Nächste Seite ausprobieren
            print(
                f"Erste Suche nach P136 auf der {i}ten Seite zu {band_name}: {page} fehlgeschlagen"
            )
            try:
                page = next(
                    pywikibot.pagegenerators.SearchPageGenerator(search_name, site=site)
                )
                item = pywikibot.ItemPage.fromPage(page)
            except StopIteration:
                print(f"Kein Genre-Tag gefunden für {search_name}")
                break  # Schleife beenden, wenn alle Artikel durchsucht wurden

    if genres is None:
        for result in pywikibot.pagegenerators.SearchPageGenerator(
            band_name, site=site
        ):
            item = pywikibot.ItemPage.fromPage(result)  # Wikidata-Item von
            # der jew. Seite ziehen
            item_dict = item.get()  # Holt wieder das item dictionary
            page = pywikibot.Page(site, band_name)
            try:
                print("zweiter Versuch")
                if "P136" in item_dict["claims"]:
                    genres = item_dict["claims"][
                        "P136"
                    ]  # dictionary besitzt tiefe Struktur
                    print(f"2.1: Suche P136 auf Seite zu {band_name}: {page} erfolgt")
                    break
                elif "P495" in item_dict["claims"]:
                    genres = item_dict["claims"][
                        "P495"
                    ]  # TODO das sollte entfernt werden, P495 ist country of origin
                    print("2.2")
                    break
            except KeyError:
                print("dritter Versuch")
                # Dritter Suchversuch, diesmal nach dem Tag "P106"
                # Es muss eine Instanz eines Menschen + Beruf des Künstlers gegeben
                # sein, sonst bedeutet "P106" etwas anderes
                # Instanz eines Menschen: Q5, Beruf des Künstlers: Q639669
                if (
                    "P31" in item_dict["claims"]
                    and any(
                        claim.getTarget().getID() == "Q5"
                        for claim in item_dict["claims"]["P31"]
                    )
                ) and (
                    "P106" in item_dict["claims"]
                    and any(
                        claim.getTarget().getID() == "Q639669"
                        for claim in item_dict["claims"]["P106"]
                    )
                ):
                    try:
                        if "P101" in item_dict["claims"]:
                            genres = item_dict["claims"]["P101"]
                            break
                    except KeyError:
                        print(
                            f"Für {band_name} ist auch kein Genre-Tag unter P101 vorhanden"
                        )
                        # csv für Künstler, bei denen P136 und P101 nicht vorhanden ist
                        # TODO in der fertigen Version sollte die Datei nicht jedes Mal neu geöffnet und geschlossen werden
                        with open("missing_p136p101.csv", "a", newline="") as f:
                            writer = csv.writer(f)
                            writer.writerow([band_name])
                        break
                else:
                    print(
                        f"{page.title()} ist keine Instanz eines Menschen und Künstlers"
                    )

    #    return genres
    for genre in genres:
        target = genre.getTarget()
        print(target.labels["en"])
        # TODO hinterher natürlich Rückgabe als String, anstatt es einfach nur zu printen


print("Tool, Ænema:")
get_band_genre("Ænema")
print("Metallic: ...And Justice for All")
get_band_genre("...And Justice for All")
print("\nHerbert Grönemeyer:")
get_band_genre("Herbert Grönemeyer")

print("pearl jam: Ten")
get_band_genre("Ten")

def clear_csv(filename):
    with open(filename, "w") as file:
        pass
