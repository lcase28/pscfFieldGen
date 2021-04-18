import cProfile
import pstats
from pstats import SortKey

from pathlib import Path

from pscfFieldGen.generation import generate_field_file, read_input_file

pwd = Path.cwd()
fpath = pwd / "model"

p, c, o = read_input_file(fpath)

cProfile.run('generate_field_file(p,c,o)','partstats')

stat = pstats.Stats('partstats')
stat = stat.strip_dirs()

# Print list sorted by total time
stat.sort_stats(SortKey.TIME).print_stats()

# Print list sorted by cumulative time
stat.sort_stats(SortKey.CUMULATIVE).print_stats()

