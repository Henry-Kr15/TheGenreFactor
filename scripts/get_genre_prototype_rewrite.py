#!/usr/bin/env python3

import pywikibot
from pywikibot import pagegenerators
from pywikibot.exceptions import NoPageError
from typing import Dict


def get_genre(band_name: str, album_name: str, album_type: str) -> Dict:
    genre_dict = {}

    # Zuerst Überprüfung, ob sich um Single oder Album handelt
    if album_type == "single" or album_type == "album":
        # 1. Suche nach Single/Album + band_name
        try:
            search_name = album_name + " " + band_name + f" ({album_type})"
            genre_dict = site_search(search_name)
        except (KeyError, NoPageError):
            try:
                # 2. Suche nach Single/Album
                search_name = album_name + f" ({album_type})"
                genre_dict = site_search(search_name)
            except (KeyError, NoPageError):
                try:
                    # 3. Nur Suche nach dem Namen des Liedes
                    search_name = album_name
                    genre_dict = site_search(search_name)
                except (KeyError, NoPageError):
                    try:
                        # 4. Nur Suche nach dem Namen der Band
                        search_name = band_name
                        genre_dict = site_search(search_name)
                    except (KeyError, NoPageError):
                        print(
                            f"Suche nach {album_name} mit allen Zusätzen fehlgeschlagen; zu den Suchbegriffen gab es keine Ergebnisse"
                        )
                        genre_dict = {}
                    except:
                        print(
                            f"Suche nach {album_name} mit allen Zusätzen fehlgeschlagen; unbekannter Fehler"
                        )
                        genre_dict = {}
        #         except:
        #             print(f"Suche nach {album_name} nach Schritt 3 fehlgeschlagen")
        #     except:
        #         print(f"Suche nach {album_name} nach Schritt 2 fehlgeschlagen")
        # except:
        #     print(f"Suche nach {album_name} nach Schritt 1 fehlgeschlagen")
    else:
        print(f"Fehler: album_type={album_type}")

    return genre_dict


def site_search(search_name) -> Dict:
    genres = None
    site = pywikibot.Site("en", "wikipedia")
    page = pywikibot.Page(site, search_name)

    while genres is None:
        item = pywikibot.ItemPage.fromPage(page)
        item_dict = item.get()
        genres = item_dict["claims"]["P136"]
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
