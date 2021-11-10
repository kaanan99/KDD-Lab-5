class Node:
    def __init__(self, is_leaf, value, other=None):
        self.leaf = is_leaf
        if self.leaf:
            self.decision = value
            self.prob = other
        else:
            self.var = value
            self.edges = []
    
    def add_edge(self, edge):
        if not self.leaf:
            self.edges.append(edge)
        else:
            print("ERROR: node is a leaf. No edge added.")

    def get_dict(self):
        if self.leaf:
            return {'decision': self.decision, 'p': self.prob}

        return {'var': self.var,
                'edges': [e.get_dict() for e in self.edges]}

class Edge:
    def __init__(self, value):
        self.val = value
        self.child_node = None
        self.direction = None

    def set_node(self, node):
        self.child_node = node

    def set_numer_direction(self, direction):
        self.direction = direction

    def get_dict(self):
        if self.child_node.leaf:
            name = "leaf"
        else:
            name = "node"
        edge_dict = {'value': self.val if isinstance(self.val, str) else float(self.val),
                     name: self.child_node.get_dict()
                    }

        # add the direction attribute only if there is one (i.e. if numerical edge)
        if self.direction is not None:
            edge_dict["direction"] = self.direction

        return {'edge': edge_dict}
