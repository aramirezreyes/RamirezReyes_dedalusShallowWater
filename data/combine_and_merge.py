# This file merges files from different processes and times prduced from dedalus v2
# The first argument must be the prefix of the output for example, 'analysis' if the file structure is:
# analysis
# analysis/analysis_s5.h5
# analysis/analysis_s1.h5
# analysis/analysis_s4.h5
# analysis/analysis_s3.h5
# analysis/analysis_s2.h5

from dedalus.tools import post
import sys
import pathlib

file_prefix = sys.argv[1]
post.merge_process_files(file_prefix, cleanup=True)
set_paths = list(pathlib.Path(file_prefix).glob(file_prefix+"_s*.h5"))
post.merge_sets(file_prefix+"/"+file_prefix+".h5", set_paths, cleanup=True)
