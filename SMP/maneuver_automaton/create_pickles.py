"""
Script to create pickle files of automata for faster loading
"""
import os
import fnmatch
from tqdm import tqdm

from SMP.maneuver_automaton.maneuver_automaton import ManeuverAutomaton

path_file = os.path.abspath(__file__)
dir_file = os.path.dirname(path_file)
root_motion_primitives = os.path.join(dir_file, "primitives")
root_automata = os.path.join(dir_file, "automata")

# create automaton pickle file with the given motion primitive file
for path, directories, files in os.walk(root_motion_primitives):
    for file_motion_primitive in tqdm(fnmatch.filter(files, "*.xml")):
        ManeuverAutomaton.create_pickle(file_motion_primitive=file_motion_primitive, dir_save=root_automata)

automata = []
# test whether it was successful
for path, directories, files in os.walk(root_automata):
    for file_motion_primitive in tqdm(fnmatch.filter(files, "*" + os.path.extsep + ManeuverAutomaton.extension)):
        automaton_file = os.path.join(path, file_motion_primitive)
        automata.append(ManeuverAutomaton.load_automaton(automaton_file))

print("Automata pickles created.")
