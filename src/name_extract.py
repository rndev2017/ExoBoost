import os


def extract_star_names(file):
    """Extracts star names from a text file

       Arguments:
           file {str} -- the path to the text file

       Returns:
          star_names {list} -- a list of star_names from text file
    """
    names = open(file, 'r')
    star_names = [line[:-1] for line in names.readlines()]

    return star_names
