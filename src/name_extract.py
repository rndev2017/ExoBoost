import os 


def extract_star_names():
    file = "\\path\\to\\names"

    names = open(file, 'r')
    star_names = []

    with names:
        for line in names.readlines():
            star = line[:-1]
            star_names.append(star)
        
    return star_names
