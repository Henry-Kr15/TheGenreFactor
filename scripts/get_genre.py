#!/usr/bin/env python3

import pywikibot
from pywikibot import pagegenerators
from pywikibot.exceptions import NoPageError
from typing import Dict


def get_genre(band_name: str, album_name: str, album_type: str) -> str:
    """
    Versucht, das Genre auf Wikidata zu finden. Dafür werden verschiedene Kombinationen von
    Suchbegriffen verwendet.

    Args:

    band_name: str, Name der Band

    album_name: str, Name des Albums des Musiktitels

    album_type: str, single oder album

    return: str, Gibt das zuerst gefundene Genre zurück
    """

    genre_dict = {}

    # Gültigkeit der Eingabeparameter
    if sanity_checks(band_name, album_name, album_type):
        # 1. Suche nach Single/Album + band_name
        try:
            search_name = album_name + " (" + album_type + ") " + band_name
            genre_dict = site_search(search_name)
        except (KeyError, NoPageError):
            try:
                # 2. Suche nach Single/Album + (album_type)
                search_name = album_name + " (" + album_type + ")"
                genre_dict = site_search(search_name)
            except (KeyError, NoPageError):
                try:
                    # 3. Suche nach Signle/Album
                    search_name = album_name
                    genre_dict = site_search(search_name)
                except (KeyError, NoPageError):
                    try:
                        # 4. Nur Suche nach dem Bandnamen + (band)
                        search_name = band_name + " (band)"
                        genre_dict = site_search(search_name)
                    except:
                        try:
                            # 5. Nur Suche nach dem Bandnamen
                            search_name = band_name
                            genre_dict = site_search(search_name)

                        except (KeyError, NoPageError):
                            print(
                                f"Suche nach {album_name} mit allen Zusätzen fehlgeschlagen; zu den Suchbegriffen gab es keine Ergebnisse"
                            )
                            genre_dict = {0: "Not Found"}
        except:
            print(
                f"Suche nach {album_name} mit allen Zusätzen fehlgeschlagen; unbekannter Fehler"
            )
            genre_dict = {0: "Not Found"}

    # Erster Schlüssel des Dictionarys
    top_key = next(iter(genre_dict))
    # Wert des Schlüssels holen
    top_genre= genre_dict[top_key]
    return top_genre


def site_search(search_name) -> Dict:
    """
    Holt das Tag P136 aus einer Liste der Trefferseiten.

    Args:

    search_name: str, Sucheingabe
    """
    genres = None
    site = pywikibot.Site("en", "wikipedia")
    page = pywikibot.Page(site, search_name)

    while genres is None:
        item = pywikibot.ItemPage.fromPage(page)
        item_dict = item.get()
        genres = item_dict["claims"]["P136"]
        if genres is None:
            page = next(
                pywikibot.pagegenerators.SearchPageGenerator(search_name, site=site)
            )

    print(page)

    # Jetzt noch Genres in Dict schreiben
    genre_dict = {}

    for i, genre in enumerate(genres):
        target = genre.getTarget()
        genre_dict[i] = target.labels["en"]

    return genre_dict


def sanity_checks(band_name: str, album_name: str, album_type: str) -> bool:
    all_test_passed = True

    # Das hier kann gar nicht passieren, Funktion lässt sich ohne Parameter eh nicht aufrufen
    # Wirft zusätzlich komischen Fehler wenn Dataframe-Spalten eingegeben werden
    # if not band_name or not album_name:
    #     print("Fehler: Band- oder Albumname fehlt.")
    #     all_test_passed = False

    valid_types = ["single", "album"]
    if album_type not in valid_types:
        print(f"Fehler: Ungültiger album_type. Erlaubte Werte sind: {valid_types}")
        all_test_passed = False

    invalid_chars = ["#", "$", "%"]  # Liste der ungültigen Zeichen
    for char in invalid_chars:
        if char in band_name or char in album_name:
            print(f"Fehler: Ungültiges Zeichen '{char}' im Band- oder Albumnamen.")
            all_test_passed = False

    return all_test_passed
