import os


def get_correct_path(files):

    return os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),  files)
