import numpy as np
import pandas as pd
import pickle
import concurrent.futures
from scipy.stats import mode


class Node:
    def __init__(self, parent = None):
        self.parent_node = parent
        self.left_node = None
        self.right_node = None
        self.threshold = None
        self.clss = None
        self.level = None
        self.entropy = None
        self.isLeaf = False

    def getParent(self):
        return(self.parent_node)
    
    def getLeft(self):
        return(self.left_node)
    
    def getRight(self):
        return(self.right_node)

    def getInfo(self, depth):
        if self.getLeft() != None:
            print(depth,":")
            print("LEFT:", self.getLeft().threshold, " ", self.getLeft().clss)
            self.getLeft().getInfo(depth+1)
        if self.getRight() != None:
            print(depth,":")
            print("RIGHT:", self.getRight().threshold, " ", self.getRight().clss)
            self.getRight().getInfo(depth+1)

    def getNodeClassByThreshold(self, value, level):
        if value[self.level] <= self.threshold:
            if self.left_node != None:
                return(self.left_node.getNodeClassByThreshold(value, level + 1))
            else:
                return(self.clss)
                
        else:
            if self.right_node != None:
                return(self.right_node.getNodeClassByThreshold(value, level + 1))
            else:
                return(self.clss)
    
    def setLeft(self, node):
        self.left_node = node
    
    def setRight(self, node):
        self.right_node = node
    
    def setThreshold(self, threshold):
        self.threshold = threshold

    def setEntropy(self, entropy):
        self.entropy = entropy


class Tree:
    def __init__(self, depth, mode = 'classification'):
        self.depth = depth
        self.mode = mode
        self.classes = None
        self.root = Node()
        self.structure = [self.root]
        
    def export_tree(self, file = "tree"):
        data = {"depth": self.depth, "classes": self.classes, "root": self.root, "structure": self.structure, 'mode': self.mode}
        pickle.dump(data, open(file+".p", "wb"))
    
    def import_tree(self, file = "tree"):
        parameters = pickle.load(open(file+".p", "rb")) 
        self.depth = parameters["depth"]
        self.classes = parameters["classes"]
        self.root = parameters["root"]
        self.structure = parameters["structure"]
        self.mode = parameters["mode"]

    def get_tree(self):
        print("ROOT NODE:", self.root.threshold)
        self.root.getInfo(0)

    def test(self, data):
        i = 0
        for idx, x in data.iterrows():
            clss = self.root.getNodeClassByThreshold(x, 0)
            if clss == x[-1]:
                i = i + 1
        return(i)

    def classify(self, data):
        classifications = []
        for idx, x in data.iterrows():
            clss = self.root.getNodeClassByThreshold(x, 0)
            classifications.append(clss)
        return(classifications)

    def entropyf(self, data):
        val = 0
        if self.mode == 'classification':
            for clss in set(data.iloc[:,-1]):
                prob = data[data.iloc[:,-1] == clss].shape[0]/data.shape[0]
                val = val - prob*np.log(prob)
        else:
            val = np.var(data.iloc[:,-1])
        return(val)

    def best_split(self, data, attribute, parent):
        best_left_data = None
        best_right_data = None
        best_entropy_left = None
        best_entropy_right = None
        best_IG = 0
        for attribute in range(0, data.shape[1]-1, 1):
            if len(set(data.iloc[:,attribute])) <= 50:
                values = set(data.iloc[:,attribute])
            else:
                values = np.quantile(list(set(data.iloc[:,attribute])), [i/50 for i in range(51)])
            for value in values:
                left_data = data[data.iloc[:,attribute] <= value]
                right_data = data[data.iloc[:,attribute] > value]
                if left_data.shape[0] == 0:
                    entropy_left = 0
                else:
                    entropy_left = self.entropyf(left_data)

                if right_data.shape[0] == 0:
                    entropy_right = 0
                else:
                    entropy_right = self.entropyf(right_data)

                IG = parent.entropy - entropy_left*(left_data.shape[0]/data.shape[0]) - entropy_right*(right_data.shape[0]/data.shape[0])
                if IG >= best_IG:
                    best_IG = IG
                    best_left_data = left_data
                    best_right_data = right_data
                    best_entropy_left = entropy_left
                    best_entropy_right = entropy_right
                    parent.threshold = value
                    parent.level = attribute
        return([best_left_data, best_right_data, best_entropy_left, best_entropy_right])

    def set_classes(self, data):
        self.classes = set(data.iloc[:,-1])

    def set_node_class(self, node, data):
        if self.mode == 'classification':
            best_clss = None
            best_clss_elements = 0
            for clss in data.iloc[:,-1]:
                clss_elements = data[data.iloc[:,-1] == clss].shape[0]
                if clss_elements > best_clss_elements:
                    best_clss = clss
                    best_clss_elements = clss_elements
            if best_clss_elements == data.shape[0]:
                node.isLeaf = True
            node.clss = best_clss
        else:
            node.clss = np.mean(data.iloc[:,-1])
            if data.shape[0] == data[data.iloc[:,0] == data.iloc[0,0]].shape[0]:
                node.isLeaf = True

    def append(self, data, attribute, parent, depth, progress = 0):
        self.set_node_class(parent, data)
        left_data, right_data, left_entropy, right_entropy = self.best_split(data, attribute, parent)
        if (left_data.shape[0] > 5 and self.depth > depth) and not parent.isLeaf:
            self.structure.append(Node(parent))
            parent.setLeft(self.structure[-1])
            parent.getLeft().entropy = left_entropy
            progress = self.append(left_data, attribute + 1, parent.getLeft(), depth + 1, progress)
        else:
            progress = progress + 1/(2**(depth+1))
            if progress in [i/8 for i in range(9)]: 
                print(progress)

        if (right_data.shape[0] > 5 and self.depth > depth) and not parent.isLeaf:
            self.structure.append(Node(parent))
            parent.setRight(self.structure[-1])
            parent.getRight().entropy = right_entropy
            progress = self.append(right_data, attribute + 1, parent.getRight(), depth + 1, progress)   
        else:
            progress = progress + 1/(2**(depth+1))
            if progress in [i/8 for i in range(9)]: 
                print(progress)

        return(progress)

    def build_tree(self, data):
        self.set_classes(data)
        self.root.entropy = self.entropyf(data)
        self.append(data, 0, self.root, 0)
        return(self)

class Forest:
    def __init__(self, mode):
        if mode != 'classification' and mode != 'regression':
            print('Wrong mode!\n Only \'classification\' and \'regression\' modes are supported')
            return
        self.trees = [] 
        self.mode = mode

    def export_forest(self, file = "forest"):
        data = {"trees": self.trees, "mode": self.mode}
        pickle.dump(data, open(file+".p", "wb"))
    
    def import_forest(self, file = "forest"):
        parameters = pickle.load(open(file+".p", "rb")) 
        self.trees = parameters["trees"]
        self.mode = parameters["mode"]

    def build_forest(self, num_of_trees, data, trees_size, data_split_size, replace_data = False):
        results = []
        with concurrent.futures.ProcessPoolExecutor() as executor:
            for _ in range(num_of_trees):
                sample = data.sample(data_split_size, replace = replace_data)
                results.append(executor.submit(Tree(trees_size, self.mode).build_tree, sample))
            for process in concurrent.futures.as_completed(results):
                self.trees.append(process.result())

    def test(self, data):
        decisions = pd.DataFrame()
        for i, tree in enumerate(self.trees):
            decisions.insert(i,i, tree.classify(data))

        if self.mode == 'classification':
            correct = 0
            final_decision = []
            for idx, row in decisions.iterrows():
                final_decision.append(mode(row)[0][0])
                if final_decision[idx] == data.iloc[idx, data.shape[1]-1]:
                    correct = correct + 1
            print("accuracy: ",correct/len(data))
        else:
            final_decision = []
            for _, row in decisions.iterrows():
                final_decision.append(np.mean(row))
            print("SSE: ", np.sum((data.iloc[:,-1] - final_decision)**2))






