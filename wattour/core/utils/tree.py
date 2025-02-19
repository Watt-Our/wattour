from __future__ import annotations

import collections
import uuid
from abc import ABC, abstractmethod
from typing import Generator, Generic, Optional, Self, TypeVar

U = TypeVar("U", bound="BaseNode")


class BaseNode(Generic[U], ABC):
    def __init__(self):
        self.id = uuid.uuid4()
        self.next: list[U] = []

    @property
    @abstractmethod
    def dummy(self) -> bool: ...

    def add(self, node: U):
        self.next.append(node)


#

T = TypeVar("T", bound="Node")


class Node(BaseNode[T], ABC):
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
class Tree(Generic[V]):
    def __init__(self):
        self.head: Optional[V] = None
        self.size = 0  # excludes dummies
        self.branches = 0
        self.dummy_nodes = 0

    # ^^ i think append (or a prelude) will just become polymorphic and V will be bound to different node types
    def append(self, existing_node: V | None, new_node: V):
        if not existing_node:
            if self.head:
                raise ValueError("Tree already has a head")
            self.head = new_node
            self.branches += 1
        else:
            new_node.validate(existing_node)
            new_node.enrich(existing_node)
            existing_node.add(new_node)

            if len(existing_node.next) > 1:
                self.branches += 1

        self.size += 1

    def copy(self) -> Self:
        new_tree = type(self)()

        def copy_helper(old_node: V) -> V:
            new_children = []
            for child in old_node.next:
                new_children.append(copy_helper(child))
            old_node_attrs = vars(old_node).copy()
            old_node_attrs.pop("next", None)
            old_node_attrs.pop("id", None)
            new_node = type(old_node)(**old_node_attrs)
            new_node.next = new_children
            return new_node

        new_head = None
        if self.head is not None:
            new_head = copy_helper(self.head)
        new_tree.head = new_head
        new_tree.size = self.size
        new_tree.branches = self.branches
        new_tree.dummy_nodes = self.dummy_nodes
        return new_tree

    def append_dummy(self, existing_node: V, dummy_node: V):
        if not dummy_node.is_dummy:
            raise ValueError("new_node must have is_dummy=True")

        self.append(existing_node, dummy_node)
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

    # mutating
    @staticmethod
    def merge_trees(first_tree: Tree[V], second_tree: Tree[V]) -> Tree[V]:
        """Merge two trees into one (combining the heads). Head of first tree will be the head of the merged tree."""
        if not first_tree.head:
            return second_tree
        if not second_tree.head:
            return first_tree

        if second_tree.head.next:
            if first_tree.head.next:
                first_tree.branches += 1
            first_tree.branches += second_tree.branches - 1
            first_tree.size += second_tree.size - 1
            first_tree.dummy_nodes += second_tree.dummy_nodes

        for node in second_tree.head.next:
            first_tree.head.add(node)
        return first_tree

    def __str__(self):
        """Return a str representation of Tree instance."""
        return f"Tree: {self.size} nodes, {self.branches} branches, {self.dummy_nodes} dummy nodes"
