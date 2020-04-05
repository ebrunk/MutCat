from os.path import abspath

VERSION = "0.9.2"
print("CCAL version {} @ {}".format(VERSION, abspath(__file__)))


from .make_match_panel import make_match_panel
from .normalize_nd_array import normalize_nd_array
from .infer import infer
from .compute_empirical_p_value import compute_empirical_p_value
from .plot_points import plot_points

