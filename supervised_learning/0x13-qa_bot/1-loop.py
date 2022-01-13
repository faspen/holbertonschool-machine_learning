#!/usr/bin/env python3
"""Create the loop"""


while True:
    question = input("Q: ")
    texts = ["exit", "quit", "goodbye", "bye"]

    if question.lower().strip() in texts:
        print("A: Goodybe")
        exit(0)
    else:
        print("A: ")
