import heapq
from abc import ABC, abstractmethod


class Queue(ABC):
    """
    Abstract class for queues.
    """
    def __init__(self):
        self.list_elements = []

    def empty(self) -> bool:
        """
        Returns true if the queue is empty.
        """
        return len(self.list_elements) == 0

    def insert(self, item) -> None:
        """
        Inserts the item into the queue.
        """
        self.list_elements.append(item)

    @abstractmethod
    def pop(self):
        """
        Pops an element from the queue.
        """
        pass


class FIFOQueue(Queue):
    """
    Implementation of First-In First-Out queues.
    """
    def __init__(self):
        super(FIFOQueue, self).__init__()

    def pop(self):
        """
        Pops the first element in the queue.
        """
        if self.empty():
            return None
        return self.list_elements.pop(0)


class LIFOQueue(Queue):
    """
    Implementation of Last-In First-Out queues.
    """
    def __init__(self):
        super(LIFOQueue, self).__init__()

    def pop(self):
        """
        Pops the last element in the queue.
        """
        if self.empty():
            return None
        return self.list_elements.pop()


class PriorityQueue:
    """
    Implementation of queues of items with priorities.
    """
    def __init__(self):
        self.list_elements = []
        self.count = 0

    def empty(self) -> bool:
        """
        Returns true if the queue is empty.
        """
        return len(self.list_elements) == 0

    def insert(self, item, priority) -> None:
        """
        Inserts an item into the queue with the given priority.

        :param item: the element to be put in the queue
        :param priority: the priority used to sort the queue. It's often the value of some cost function.
        """
        self.count += 1
        heapq.heappush(self.list_elements, (priority * 10000, self.count, item))

    def pop(self):
        """
        Pops the item with the least priority off the heap (Priority queue) if the queue is not empty.
        """
        if self.empty():
            return None

        return heapq.heappop(self.list_elements)[2]
