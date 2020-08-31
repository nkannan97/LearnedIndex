from collections import namedtuple

import numpy as np

TreeNodeItem = namedtuple('TreeNodeItem', ['key', 'child_node'])
# Special case: for a leaf node, every child_node in tree_node.children is None, then this is a leaf node
# Special case: for a NON-leaf node, the last TreeNodeItem in tree_node.children has a key of float('inf')


class TreeNode:
    def __init__(self, capacity):
        self.children = [TreeNodeItem(float('inf'), None)]
        self.num_descendants = 0
        self.capacity = capacity

    def is_full(self):
        return len(self.children) - 1 >= self.capacity
        # -1 for the sentinel inf at the end.

    def is_leaf(self):
        return self.children[0].child_node is None

    def search(self, key):
        # Returns position of key WITHIN SUBTREE
        i = 0
        cumulative_descendants = 0
        while self.children[i].key < key:
            if not self.is_leaf():  # non-leaf node
                cumulative_descendants += self.children[i].child_node.num_descendants
            i += 1
        # because the last child always has a key of float('inf')
        assert i < len(self.children)
        if key == self.children[i].key:
            return i + cumulative_descendants + (self.children[i].child_node.num_descendants if not self.is_leaf() else 0)
        # key != self.children[counter] (i.e. no exact match)
        if not self.is_leaf():  # non-leaf node; search children
            child_search_result = self.children[i].child_node.search(key)
            if child_search_result is not None:
                return i + cumulative_descendants + child_search_result
        return None

    def update_num_descendants(self):
        # -1 for the sentinel inf at the end.
        self.num_descendants = sum(
            [1 + (child.child_node.num_descendants if not self.is_leaf() else 0) for child in self.children]) - 1

    def split_ith_child(self, i):
        assert not self.is_full()
        assert not self.is_leaf()

        ith_child = self.children[i].child_node
        iplus1th_child = TreeNode(self.capacity)
        # partition children
        promoted_index = len(ith_child.children) // 2
        iplus1th_child.children = ith_child.children[promoted_index + 1:]
        promoted_child = ith_child.children[promoted_index]
        ith_child.children = ith_child.children[:promoted_index]
        ith_child.children.append(TreeNodeItem(
            float('inf'), promoted_child.child_node))
        # update self.children
        self.children.insert(
            i + 1, TreeNodeItem(self.children[i].key, iplus1th_child))
        self.children[i] = TreeNodeItem(promoted_child.key, ith_child)
        # update num_descendants
        self.children[i].child_node.update_num_descendants()
        if i + 1 < len(self.children):
            self.children[i + 1].child_node.update_num_descendants()
        self.update_num_descendants()

    def insert(self, key):
        assert not self.is_full()
        i = 0
        while self.children[i].key < key:
            i += 1
        # because the last child always has a key of float('inf')
        assert i < len(self.children)
        if self.is_leaf():  # this is a leaf node
            self.children.insert(i, TreeNodeItem(key, None))
        else:
            if self.children[i].child_node.is_full():
                self.split_ith_child(i)
                if key >= self.children[i].key:
                    i += 1
            self.children[i].child_node.insert(key)
            self.children[i].child_node.update_num_descendants()
            if i + 1 < len(self.children):
                self.children[i + 1].child_node.update_num_descendants()
        self.update_num_descendants()

    def as_indented_string(self, level):
        return '(' + str(self.num_descendants) + ')' + ''.join([
            '\t'*level
            + str(child.key)
            + '\n'
            + (child.child_node.as_indented_string(level + 1)
               if child.child_node is not None else '')
            for child in self.children
        ])

    def size_if_full(self):
        return self.capacity + (sum([child.child_node.size_if_full() for child in self.children]) if not self.is_leaf() else 0)

    def depth(self):
        return 1 + (max([child.child_node.depth() for child in self.children]) if not self.is_leaf() else 0)


class BTree:
    def __init__(self, node_capacity):
        self.root = TreeNode(node_capacity)
        self.node_capacity = node_capacity

    def search(self, key):
        return self.root.search(key)

    def lookup(self, key, verbose=False):
        position = self.search(key)
        if verbose:
            print('lookup: ' + str(key) + ', ' + str(position))
        return position

    def insert(self, key, verbose=False):
        if verbose:
            print('train: ' + str(key))
        if self.root.is_full():
            root = TreeNode(self.node_capacity)
            root.children[0] = TreeNodeItem(float('inf'), self.root)
            self.root = root
            self.root.split_ith_child(0)
        self.root.insert(key)

    def size_if_full(self):
        return self.root.size_if_full()

    def depth(self):
        return self.root.depth()

    def __str__(self):
        return self.root.as_indented_string(0)


def main():
    data = np.random.default_rng().random(1000000)
    argsort_data = np.argsort(data)

    btree = BTree(20)

    for i in data:
        btree.insert(i)

    print(btree.size_if_full())
    for p, i in enumerate(argsort_data):
        l = btree.lookup(data[i])
        if p != l:
            print(str(data[i]) + ': ' + str(l) + '. True: ' + str(p))


if __name__ == '__main__':
    main()
