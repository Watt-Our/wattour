from __future__ import annotations

import collections
import uuid
from abc import ABC, abstractmethod
from typing import Generator, Generic, Optional, Self, TypeVar

T = TypeVar("T")
U = TypeVar("U")


class BaseNode(ABC, Generic[T, U]):
    def __init__(self):
        self.id = uuid.uuid4()
        self.next: list[U] = []

    @property
    @abstractmethod
    def dummy(self) -> bool: ...

    def add(self, node: U):
        self.next.append(node)


#


# TODO: broken
class Node(BaseNode[T, Self], ABC):
    # type of node that has value, validates input, and enriches input

    def __init__(self, is_dummy: bool = False):
        super().__init__()
        self.is_dummy = is_dummy

    @property
    def dummy(self) -> bool:
        return self.is_dummy

    @abstractmethod
    def validate(self, prev: Self) -> None: ...

    @abstractmethod
    def enrich(self, prev: Self) -> None: ...


#

V = TypeVar("V", bound=Node)


# this is a little wrong bc i want it to work for different types of nodes
class Idk(Generic[V]):
    def __init__(self):
        self.head: Optional[V] = None
        self.tail: Optional[V] = None

        self.size = 0
        self.branches = 0
        self.dummy_nodes = 0

    # ^^ i think append (or a prelude) will just become polymorphic and V will be bound to different node types
    def append(self, data: V):
        new_node = data

        if self.tail:
            new_node.validate(self.tail)
            new_node.enrich(self.tail)
            self.tail.add(new_node)
        else:
            self.head = new_node

        self.tail = new_node
        self.size += 1
        self.branches += 1

    def append_dummy(self, data: V):
        self.append(data)
        self.dummy_nodes += 1

    def iter_nodes(self, show_dummy: bool = True) -> Generator[V]:
        # bfs
        if not self.head:
            raise ValueError("Timeseries is empty")

        q = collections.deque([self.head])
        while q:
            cur = q.popleft()
            if not show_dummy and cur.dummy:
                continue

            yield cur
            if cur.next:
                q.extend(cur.next)

    def add_branch(self, node: V, branch: Self):
        """Add a branch to a node."""
        if node.dummy:
            raise ValueError("Cannot add a branch to a dummy node.")

        if not node.next:
            self.branches += 1
        self.branches += branch.branches - 1
        self.size += branch.size
        self.dummy_nodes += branch.dummy_nodes

        node.next.append(branch.head)
