import math
import csv
import copy
import xml.etree.cElementTree as ET
import argparse
import  os
import os.path
from xml.etree.ElementTree import Element, SubElement, Comment, tostring


class tree_node:
    def __init__(self):
        self.parent = None
        self.child = []
        self.class_label = None
        self.entropy = None
        self.feature = None
        self.attribute = None


class training_data:

    attribute_names = []

    log_base = None

    def __init__(self, row):
        self.attributes = row
        self.classification = row[len(row) - 1]
        del self.attributes[-1]


class classification_4_attribute_value:
    def __init__(self):
        self.name = None
        self.count = 0

class attribute_value:
    def __init__(self, name, classification, t_data):
        self.name = name
        self.classifications = []
        self.e_sub = []
        self.increment(classification, t_data)


    def increment(self, classification, t_data):
        #check the available classifications
        #if not exist then add one
        #else increment the existing count

        self.e_sub.append(t_data)
        found = False
        for i in range(len(self.classifications)):
            if self.classifications[i].name == classification:
                found = True
                self.classifications[i].count = self.classifications[i].count +1
                break
        if not found:
            n_classification = classification_4_attribute_value()
            n_classification.name = classification
            n_classification.count = n_classification.count + 1
            self.classifications.append(n_classification)

    def entropy(self):
        return Entropy.calc_entropy(self.e_sub)


class attribute:
    def __init__(self, Examples, attribute_name):
        self.Examples = Examples
        self.name = attribute_name
        self.attribute_values = []
        self.classify(Examples, attribute_name)

    def classify(self, Examples, attribute_name):
        index = 0
        for i in range(len(training_data.attribute_names)):
            if attribute_name == training_data.attribute_names[i]:
                index = i
                break
        for i in range(len(Examples)):
            atr_val = Examples[i].attributes[index]

            found = False
            for j in range(len(self.attribute_values)):
                if self.attribute_values[j].name == atr_val:
                    found = True
                    self.attribute_values[j].increment(Examples[i].classification, Examples[i])
                    break

            if not found:
                n_attr_val = attribute_value(atr_val, Examples[i].classification, Examples[i])
                self.attribute_values.append(n_attr_val)

    def gain(self):
        gain = 0
        total = len(self.Examples)
        for i in range(len(self.attribute_values)):
            prob = float(len(self.attribute_values[i].e_sub))/total
            gain = gain + (prob*self.attribute_values[i].entropy())
        return gain







class Entropy:

    @classmethod
    def calc_entropy(cls, Examples):
        classification =[]#class, even occurance
        for i in range(len(Examples)):
            t_classificcation = Examples[i].classification

            found = False
            for j in range(0, len(classification), 2):
                if classification[j] == t_classificcation:
                    classification[j+1] = classification[j+1]+1
                    found = True
                    break
            if not found:
                classification.append(t_classificcation)
                classification.append(1)

        if training_data.log_base is None:
            training_data.log_base = len(classification)/2

        entropy = 0
        total_elements = len(Examples)
        for i in range(0, len(classification), 2):
            probability = float(classification[i+1])/total_elements
            entropy = entropy + (probability*math.log(probability, training_data.log_base))#len(training_data.attribute_names)))
        return entropy*(-1)



class DecisionTreeLearner:

    def __init__(self):
        self.root = None

    #examples are list of type training_data
    def ID3(self, Examples, TargetAttribute, Attribute, branch, parent):

        root = tree_node()

        if self.root is None:
            self.root = root
        else:
            root.parent = parent
            parent.child.append(root)
            root.feature = branch
            root.attribute = TargetAttribute

        f_attr = None
        for i in range((len(Examples))):
            if i==0:
                f_attr = Examples[i].classification
            elif Examples[i].classification != f_attr:
                f_attr = None
                break
        if f_attr is not None:
            root.entropy = 0
            root.class_label = f_attr
            return root
        classify=[]
        entropy = Entropy.calc_entropy(Examples)
        for i in range(len(Attribute)):
            classify.append(attribute(Examples, Attribute[i]))
        gain = []
        for i in range(len(Attribute)):
            gain.append(entropy - classify[i].gain())

        max_gain = 0
        index = 0
        for i in range(len(gain)):
            if i == 0:
                max_gain=gain[i]
                index = i
            if gain[i]>max_gain:
                max_gain = gain[i]
                index = i

        #new chosen attribute
        selected_attr = classify[index]
        n_attr = copy.deepcopy(Attribute)
        n_attr.remove(selected_attr.name)

        root.entropy = entropy

        for i in range(len(selected_attr.attribute_values)):
            self.ID3(selected_attr.attribute_values[i].e_sub, selected_attr.name, n_attr,
                     selected_attr.attribute_values[i].name, root)


def create_tree(xml, node):
    current = ET.SubElement(xml, 'node')
    if node.entropy != 0:
        current.set('entropy', str(node.entropy))
    else:
        current.set('entropy', '0.0')
    current.set('feature', str(node.attribute))
    current.set('value', str(node.feature))

    if node.class_label is not None:
        current.text = node.class_label
    else:
        for i in range(len(node.child)):
            create_tree(current, node.child[i])


def is_valid_file(parser, arg):
    if not os.path.exists(arg):
        parser.error("The file %s does not exist!" % arg)
    else:
        return arg


def is_not_valid_file(parser, arg):
    if os.path.exists(arg):
        parser.error("The file %s already exist!" % arg)
    else:
        return arg




if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('--data', required=True, type=lambda x: is_valid_file(parser, x))
    parser.add_argument('--output', required=True, type=lambda x: is_not_valid_file(parser, x))
    args = parser.parse_args()

    rows = []

    with open(args.data) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            if len(row) == 0:
                continue
            rows.append(training_data(row))

    '''training_data.attribute_names.append('Outlook')
    training_data.attribute_names.append('temp')
    training_data.attribute_names.append('humidity')
    training_data.attribute_names.append('wind')
'''
    num_attr = len(rows[0].attributes)
    for i in range(num_attr):
        st = 'att' + str(i)
        training_data.attribute_names.append(st)

    dtl = DecisionTreeLearner()
    dtl.ID3(rows, None, training_data.attribute_names, None, None)

    tree = ET.Element("tree", {})
    tree.set('entropy', str(dtl.root.entropy))
    for i in range(len(dtl.root.child)):
        create_tree(tree, dtl.root.child[i])
    xml_tree = ET.ElementTree(tree)
    xml_tree.write(args.output)

    print('Execution finished\n')
