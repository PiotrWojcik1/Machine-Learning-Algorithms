import multiprocessing
from unicodedata import digit
import numpy as np
import pandas as pd
import pickle
import time

class Node:
    def __init__(self, id, parent = None):
        self.id = id
        self.parent_node = parent
        self.left_node = None
        self.right_node = None
        self.threshold = None
        self.clss = None
        self.level = None
        self.variable = None
        self.gini = None
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

    def getNodeByThreshold(self, value, level):
        #if value[level] <= self.threshold:
        if value[self.level] <= self.threshold:
            if self.left_node != None:
                return(self.left_node.getNodeByThreshold(value, level + 1))
            else:
                return(self.clss)
                
        else:
            if self.right_node != None:
                return(self.right_node.getNodeByThreshold(value, level + 1))
            else:
                return(self.clss)
    
    def setLeft(self, node):
        self.left_node = node
    
    def setRight(self, node):
        self.right_node = node
    
    def setThreshold(self, threshold):
        self.threshold = threshold

    def setGini(self, gini):
        self.gini = gini


class Tree:
    def __init__(self, depth):
        self.depth = depth
        self.classes = None
        self.root = Node(0)
        self.structure = [self.root]

    #def isLeaf(self, node):
    #    if node.getLeft == None and node.getRight == None:
    #        return(False)
    #    else:
    #        return(True)
        
    def export_tree(self, file = "tree"):
        data = {"depth": self.depth, "classes": self.classes, "root": self.root, "structure": self.structure}
        pickle.dump(data, open(file+".p", "wb"))
    
    def import_tree(self, file = "tree"):
        parameters = pickle.load(open(file+".p", "rb")) 
        self.depth = parameters["depth"]
        self.classes = parameters["classes"]
        self.root = parameters["root"]
        self.structure = parameters["structure"]

    def get_tree(self):
        print("ROOT NODE:", self.structure[0].threshold)
        self.root.getInfo(0)

    def getNodeByID(self, id):
        for node in self.structure:
            if node.id == id:
                return(node)

    def classify(self, data):
        i = 0
        for idx, x in data.iterrows():
            clss = self.root.getNodeByThreshold(x, 0)
            if clss == x[-1]:
                i = i + 1
        return(i)

    def entropyf(self, data):
        val = 0
        for clss in self.classes:
            prob = data[data.iloc[:,-1] == clss].shape[0]/data.shape[0]
            if prob != 0:
                val = val - prob*np.log(prob)
        return(val)

    def best_split(self, data, attribute, parent):
        best_left_data = None
        best_right_data = None
        best_gini_left = None
        best_gini_right = None
        best_IG = 0
        for attribute in range(0, data.shape[1]-1, 5): #TO DO: CHANGE ATTRIBUTE LOOPING
            if len(set(data.iloc[:,attribute])) <= 20:
                values = set(data.iloc[:,attribute])
            else:
                values = np.quantile(list(set(data.iloc[:,attribute])), [i/20 for i in range(21)])
            for value in values:
                left_data = data[data.iloc[:,attribute] <= value]
                right_data = data[data.iloc[:,attribute] > value]
                if left_data.shape[0] == 0:
                    gini_left = 0
                else:
                    gini_left = self.entropyf(left_data)

                if right_data.shape[0] == 0:
                    gini_right = 0
                else:
                    gini_right = self.entropyf(right_data)

                IG = parent.gini - gini_left*(left_data.shape[0]/data.shape[0]) - gini_right*(right_data.shape[0]/data.shape[0])
                if IG >= best_IG:
                    best_IG = IG
                    best_left_data = left_data
                    best_right_data = right_data
                    best_gini_left = gini_left
                    best_gini_right = gini_right
                    parent.threshold = value
                    parent.level = attribute
        print(best_left_data.shape)
        return([best_left_data, best_right_data, best_gini_left, best_gini_right])

    def set_classes(self, data):
        self.classes = set(data.iloc[:,-1])

    def set_node_class(self, node, data):
        best_clss = None
        best_clss_elements = 0
        for clss in self.classes:
            clss_elements = data[data.iloc[:,-1] == clss].shape[0]
            if clss_elements > best_clss_elements:
                best_clss = clss
                best_clss_elements = clss_elements
        if best_clss_elements == data.shape[0]:
            node.isLeaf = True
        node.clss = best_clss

    def append(self, data, attribute, parent, depth, queue = multiprocessing.Queue()):
        self.set_node_class(parent, data)
        left_data, right_data, left_gini, right_gini = self.best_split(data, attribute, parent)
        if depth == 0:
            queue.put(0)
            queue.put(parent)
        
        if (left_data.shape[0] > 20 and self.depth > depth) and (attribute < data.shape[1] - 2 and not parent.isLeaf):
            self.structure.append(Node(len(self.structure), parent))
            parent.setLeft(self.structure[-1])
            parent.getLeft().gini = left_gini
            queue.put(1)
            queue.put(parent.getLeft())
            print(parent.getLeft().)
            queue.put(parent.id)
                #parent.setLeft(queue.get())

            #if depth == 0:               
            #    process_left = multiprocessing.Process(target = self.append, args = (left_data, attribute + 1, parent.getLeft(), depth + 1, queue))
            #    process_left.start()
            #else:
            self.append(left_data, attribute + 1, parent.getLeft(), depth + 1, queue)

        if (right_data.shape[0] > 20 and self.depth > depth) and (attribute < data.shape[1] - 2 and not parent.isLeaf):
            self.structure.append(Node(len(self.structure), parent))
            parent.setRight(self.structure[-1])
            parent.getRight().gini = right_gini
            queue.put(2)
            queue.put(parent.getRight())
            queue.put(parent.id)
            #if depth == 0:                
            #    process_right = multiprocessing.Process(target = self.append, args = (right_data, attribute + 1, parent.getRight(), depth + 1))
            #    process_right.start()
            #else:
            self.append(right_data, attribute + 1, parent.getRight(), depth + 1, queue)
        
        if depth == 0:
            print("fdsfds")
            print(self.structure)
            #print(self.root.getLeft())
            #process_left.join()
            #self.root.setLeft(parent.getLeft())
        return()

    def build_tree(self, data):
        self.set_classes(data)
        self.root.gini = self.entropyf(data)
        queue = multiprocessing.Queue()
        append_process = multiprocessing.Process(target = self.append, args = (data, 0, self.root, 0, queue))
        append_process.start()
        while append_process.exitcode == None:
            #while not queue.empty():
                #self.structure.append(queue.get())
                #print("aaaaaaa")
            while not queue.empty():
                state = queue.get()
                print(state)
                if state == 1:
                    self.structure.append(queue.get())
                    self.getNodeByID(queue.get()).setLeft(self.structure[-1])
                elif state == 2:
                    self.structure.append(queue.get())
                    self.getNodeByID(queue.get()).setRight(self.structure[-1])
                else:
                    self.structure[0] = queue.get()
                    self.root = self.structure[0]
        #self.structure.append(queue.get())
        append_process.join()
        print(self.structure)
            #self.structure.append(queue.get())


        #self.append(data, 0, self.root, 0)


if __name__ =='__main__':
    decision_tree = Tree(700)
    csv_url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
    col_names = ['Sepal_Length','Sepal_Width','Petal_Length','Petal_Width','Class']
    iris =  pd.read_csv(csv_url, names = col_names)

    column1 = [-3.2,2,-1,-3.8,-6.7,0.9,-1.8,-0.4,-1.8,2.5]
    column2 = [2.8,1.8,-6.4,-8.1,6.8,8.9,11.9,0,-6.8,7.5]
    column3 = [1,0,1,1,1,0,0,1,1,0]
    data = pd.DataFrame({1:column1, 2:column2, "label":column3})

    digits = pd.read_csv("train.csv")
    test_data = pd.read_csv("test.csv")
    test_data_results = pd.read_csv("digit_classifier.csv")

    test_data['class'] = test_data_results['Label']

    idx = list([i for i in range(1,785)])
    idx.append(0)
    digits = digits.iloc[:,idx]
    #iris = iris.sample(150)
    #print(iris.iloc[100:150,])
    #decision_tree.import_tree()
    start = time.time()
    decision_tree.build_tree(digits.iloc[0:100,:])
    end = time.time()
    print("czas: ",end-start)
    print(decision_tree.root.getLeft())
    decision_tree.get_tree()
    #decision_tree.export_tree()
    print(decision_tree.classify(test_data))