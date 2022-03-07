#!/usr/bin/env python3
"""Can I join?"""


import requests


def availableShips(passengerCount):
    """Get a list of habitable ships"""
    if passengerCount < 1:
        return []
    ships = []
    total = []
    url = 'https://swapi-api.hbtn.io/api/starships/?format=json'

    while 1:
        r = requests.get(url).json()
        total += r.get('results')
        next = r.get('next')

        if next:
            url = next
        else:
            break
    for ship in total:
        try:
            if int(ship.get('passengers').replace(',', '')) >= passengerCount:
                ships.append(ship.get('name'))
        except Exception as e:
            continue
    return ships
