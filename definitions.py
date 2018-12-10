from os.path import dirname, abspath, join

"""Can be modified."""
# The output directory's name.
OUT_DIR_NAME = 'out'

"""Do not modify."""
# The root folder.
ROOT = dirname(abspath(__file__))
# The output folder's path.
OUT_PATH = join(ROOT, OUT_DIR_NAME)
