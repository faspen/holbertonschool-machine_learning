#!/usr/bin/env python3
"""How many by rocket?"""


import re
import requests


if __name__ == "__main__":
    """Display launches per rocket"""
    url = "https://api.spacexdata.com/v4/launches"
    results = requests.get(url).json()
    rockets = {}

    for la in results:
        rocket = la.get('rocket')
        url = "https://api.spacexdata.com/v4/rockets/{}".format(rocket)
        results = requests.get(url).json()
        rocket = results.get('name')

        if rockets.get(rocket) is None:
            rockets[rocket] = 1
        else:
            rockets[rocket] += 1
    rList = sorted(rockets.items(), key=lambda v: v[0])
    rList = sorted(rList, key=lambda v: v[1], reverse=True)

    for rocket in rList:
        print("{}: {}".format(rocket[0], rocket[1]))
