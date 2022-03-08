#!/usr/bin/env python3
"""Find Github user"""


import requests
import time
import sys


if __name__ == "__main__":
    """Print location of Github user"""
    url = sys.argv[1]
    r = requests.get(url)

    if r.status_code != 200:
        if r.status_code == 403:
            reset = int(r.headers.get('X-Ratelimit-Reset'))
            min = int(reset) - time.time()
            min = round(min / 60)

            print("Resest in {} min".format(min))
            exit()
    user = r.json()
    loc = user.get('location')

    if loc:
        print(user.get('location'))
    else:
        print("Not found")
