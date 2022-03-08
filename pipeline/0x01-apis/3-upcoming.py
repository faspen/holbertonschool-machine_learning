#!/usr/bin/env python3
"""What will be next?"""


from unittest import result
import requests


if __name__ == "__main__":
    """Get SpaceX launch information"""
    url = "https://api.spacexdata.com/v4/launches/upcoming"
    results = requests.get(url).json()
    date = float('inf')
    launch = None
    rocket = None
    pad = None
    location = None

    for la in results:
        launchTime = la.get('date_unix')

        if launchTime < date:
            date = launchTime
            local = la.get('date_local')
            launch = la.get('name')
            rocket = la.get('rocket')
            pad = la.get('launchpad')

    if rocket:
        rocket = requests.get("https://api.spacexdata.com/v4/rockets/{}".
                              format(rocket)).json().get('name')
    if pad:
        pad_tmp = requests.get("https://api.spacexdata.com/v4/launchpads/{}".
                               format(pad)).json()
        pad = pad_tmp.get('name')
        location = pad_tmp.get('locality')

    print("{} ({}) {} - {} ({})".format(launch, local, rocket, pad, location))
