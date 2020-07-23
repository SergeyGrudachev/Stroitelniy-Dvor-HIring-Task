import itertools
import pandas as pd


class FPNode(object):
    """
    A node in the FP tree.
    """

    def __init__(self, value, count, parent):
        """
        Create the node.
        """
        self.value = value
        self.count = count
        self.parent = parent
        self.link = None
        self.children = []

    def has_child(self, value):
        """
        Check if node has a particular child node.
        """
        for node in self.children:
            if node.value == value:
                return True

        return False

    def get_child(self, value):
        """
        Return a child node with a particular value.
        """
        for node in self.children:
            if node.value == value:
                return node

        return None

    def add_child(self, value):
        """
        Add a node as a child node.
        """
        child = FPNode(value, 1, self)
        self.children.append(child)
        return child


class FPTree(object):
    """
    A frequent pattern tree.
    """

    def __init__(self, transactions, threshold, root_value, root_count):
        """
        Initialize the tree.
        """
        self.frequent = self.find_frequent_items(transactions, threshold)
        self.headers = dict.fromkeys(list(self.frequent.keys()))
        self.root = self.build_fptree(transactions, root_value, root_count, self.frequent, self.headers)

    @staticmethod
    def find_frequent_items(transactions, threshold):
        """
        Create a dictionary of items with occurrences above the threshold.
        """
        items = {}

        for transaction in transactions:
            for item in transaction:
                if item in items:
                    items[item] += 1
                else:
                    items[item] = 1

        for key in list(items.keys()):
            if items[key] < threshold:
                del items[key]

        return items

    def build_fptree(self, transactions, root_value, root_count, frequent, headers):
        """
        Build the FP tree and return the root node.
        """
        root = FPNode(root_value, root_count, None)

        for transaction in transactions:
            sorted_items = [x for x in transaction if x in frequent]
            sorted_items.sort(key=lambda x: frequent[x], reverse=True)
            if len(sorted_items) > 0:
                self.insert_tree(sorted_items, root, headers)

        return root

    def insert_tree(self, items, node, headers):
        """
        Recursively grow FP tree.
        """
        first = items[0]
        child = node.get_child(first)
        if child is not None:
            child.count += 1
        else:
            # Add new child.
            child = node.add_child(first)

            # Link it to header structure.
            if headers[first] is None:
                headers[first] = child
            else:
                current = headers[first]
                while current.link is not None:
                    current = current.link
                current.link = child

        # Call function recursively.
        remaining_items = items[1:]
        if len(remaining_items) > 0:
            self.insert_tree(remaining_items, child, headers)

    def tree_has_single_path(self, node):
        """
        If there is a single path in the tree,
        return True, else return False.
        """
        num_children = len(node.children)
        if num_children > 1:
            return False
        elif num_children == 0:
            return True
        else:
            return True and self.tree_has_single_path(node.children[0])

    def mine_patterns(self, threshold):
        """
        Mine the constructed FP tree for frequent patterns.
        """
        if self.tree_has_single_path(self.root):
            return self.generate_pattern_list()
        else:
            return self.zip_patterns(self.mine_sub_trees(threshold))

    def zip_patterns(self, patterns):
        """
        Append suffix to patterns in dictionary if
        we are in a conditional FP tree.
        """
        suffix = self.root.value

        if suffix is not None:
            # We are in a conditional tree.
            new_patterns = {}
            for key in patterns.keys():
                new_patterns[tuple(sorted(list(key) + [suffix]))] = patterns[key]

            return new_patterns

        return patterns

    def generate_pattern_list(self):
        """
        Generate a list of patterns with support counts.
        """
        patterns = {}
        items = self.frequent.keys()

        # If we are in a conditional tree,
        # the suffix is a pattern on its own.
        if self.root.value is None:
            suffix_value = []
        else:
            suffix_value = [self.root.value]
            patterns[tuple(suffix_value)] = self.root.count

        for i in range(1, len(items) + 1):
            for subset in itertools.combinations(items, i):
                pattern = tuple(sorted(list(subset) + suffix_value))
                patterns[pattern] = \
                    min([self.frequent[x] for x in subset])

        return patterns

    def mine_sub_trees(self, threshold):
        """
        Generate subtrees and mine them for patterns.
        """
        patterns = {}
        mining_order = sorted(self.frequent.keys(),
                              key=lambda x: self.frequent[x])

        # Get items in tree in reverse order of occurrences.
        for item in mining_order:
            suffixes = []
            conditional_tree_input = []
            node = self.headers[item]

            # Follow node links to get a list of
            # all occurrences of a certain item.
            while node is not None:
                suffixes.append(node)
                node = node.link

            # For each occurrence of the item,
            # trace the path back to the root node.
            for suffix in suffixes:
                frequency = suffix.count
                path = []
                parent = suffix.parent

                while parent.parent is not None:
                    path.append(parent.value)
                    parent = parent.parent

                for i in range(frequency):
                    conditional_tree_input.append(path)

            # Now we have the input for a subtree,
            # so construct it and grab the patterns.
            subtree = FPTree(conditional_tree_input, threshold,
                             item, self.frequent[item])
            subtree_patterns = subtree.mine_patterns(threshold)

            # Insert subtree patterns into main patterns dictionary.
            for pattern in subtree_patterns.keys():
                if pattern in patterns:
                    patterns[pattern] += subtree_patterns[pattern]
                else:
                    patterns[pattern] = subtree_patterns[pattern]

        return patterns


def generate_association_rules(transactions, min_supp, min_conf):
    """
    Given a transactions list of lists with items,
    min_supp integer that is the smallest number of
    occurrences of item in transactions and min_conf
    float that is the smallest level of confidence for generated
    rules, return a dict of association rules in the form
    {(premise): ((consequence), confidence, lift)}
    """
    tree = FPTree(transactions, min_supp, None, None)
    patterns = tree.mine_patterns(min_supp)
    rules = {}
    for itemset in patterns:
        pattern_supp = patterns[itemset] / len(transactions)

        for i in range(1, len(itemset)):
            for antecedent in itertools.combinations(itemset, i):
                antecedent = tuple(sorted(antecedent))
                consequent = tuple(sorted(set(itemset) - set(antecedent)))

                if antecedent in patterns and consequent in patterns:
                    antecedent_supp = patterns[antecedent] / len(transactions)
                    cosequent_support = patterns[consequent] / len(transactions)

                    confidence = pattern_supp / antecedent_supp
                    lift = pattern_supp / (antecedent_supp * cosequent_support)

                    if confidence >= min_conf:
                        rules[antecedent] = (consequent, confidence, lift)

    return rules


def make_transactions_list(orders_data):
    transactions = {}
    for order in orders_data.itertuples():
        if order.order_id in transactions:
            if order.material not in transactions[order.order_id]:
                transactions[order.order_id].append(order.material)
        else:
            transactions[order.order_id] = [order.material]
    transactions = list(transactions.values())
    return transactions


def make_rules_df(transactions_list, min_supp_percent=0.001, min_conf=0.6):
    min_supp = round(len(transactions_list) * min_supp_percent)
    rules = generate_association_rules(transactions_list, min_supp, min_conf)
    result = {'X': [], 'Y': [], 'Confidence': [], 'Lift': []}
    for rule in rules:
        result['X'].append(list(rule))
        result['Y'].append(list(rules[rule][0]))
        result['Confidence'].append(rules[rule][1])
        result['Lift'].append(rules[rule][2])
    return pd.DataFrame(data=result).sort_values(by='Confidence', ascending=False, ignore_index=True)


data = pd.read_csv('2_5226447128707991493.csv', sep=';')
data = data.drop(data[(data['inv_qty'] <= 0)].index)

rules_table = {'ALL': make_rules_df(make_transactions_list(data), min_supp_percent=0.0013),
               1000: make_rules_df(make_transactions_list(data[data['org'] == 1000]), min_supp_percent=0.0013),
               2000: make_rules_df(make_transactions_list(data[data['org'] == 2000]), min_supp_percent=0.0017),
               3000: make_rules_df(make_transactions_list(data[data['org'] == 3000]), min_supp_percent=0.0013),
               4001: make_rules_df(make_transactions_list(data[data['org'] == 4001])),
               7000: make_rules_df(make_transactions_list(data[data['org'] == 7000]), min_supp_percent=0.017)}
