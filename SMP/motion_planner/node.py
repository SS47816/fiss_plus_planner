from typing import List
from SMP.maneuver_automaton.motion_primitive import MotionPrimitive
from commonroad.scenario.state import KSState


class Node:
    """
    Class for nodes used in the motion planners.
    """

    def __init__(self, list_paths: List[List[KSState]], list_primitives: List[MotionPrimitive], depth_tree: int):
        """
        Initialization of class Node.
        """
        # list of paths of motion primitives
        self.list_paths = list_paths
        # list of motion primitives
        self.list_primitives = list_primitives
        # depth of the node
        self.depth_tree = depth_tree

    def get_successors(self) -> List[MotionPrimitive]:
        """
        Returns all possible successor primitives of the current primitive (node).
        """
        return self.list_primitives[-1].list_successors


class PriorityNode(Node):
    """
    Class for nodes with priorities used in the motion planners.
    """

    def __init__(self, list_paths: List[List[KSState]], list_primitives: List[MotionPrimitive], depth_tree: int,
                 priority: float):
        """
        Initialization of class PriorityNode.
        """
        super().__init__(list_paths, list_primitives, depth_tree)
        # priority/cost of the node
        self.priority = priority
