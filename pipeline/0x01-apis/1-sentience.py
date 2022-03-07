#!/usr/bin/env python3
"""Where I am?"""


import requests


def sentientPlanets():
    """Return planets with sentient life"""
    planets = []
    lifeforms = []
    url = 'https://swapi-api.hbtn.io/api/species/?format=json'

    while 1:
        r = requests.get(url).json()
        lifeforms += r.get('results')
        next = r.get('next')

        if next:
            url = next
        else:
            break
    for types in lifeforms:
        try:
            if types.get('designation') == 'sentient' or\
                    types.get('classification') == 'sentient':
                origin = types.get('homeworld')
                if origin:
                    name = requests.get(origin).json().get('name')
                    planets.append(name)
        except Exception as e:
            continue
    return planets
