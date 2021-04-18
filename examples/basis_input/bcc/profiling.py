import cProfile
import pstats
from pstats import SortKey

from pathlib import Path

from pscfFieldGen.generation import generate_field_file, read_input_file

pwd = Path.cwd()
fpath = pwd / "model"

p, c, o = read_input_file(fpath)

cProfile.run('generate_field_file(p,c,o)','partstats')

