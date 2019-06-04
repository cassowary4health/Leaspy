import xml.etree.ElementTree as ET

def assert_existence_of_children_nodes(parent_node, name_list, path):
    """
    Checks that the Element parent_node does have all the elements in name_list as children
    :param parent_node: The parent node to check
    :param name_list: The children nodes to check
    :param path: The path of the xml file containing the elements, for the error message
    :return: An error if one of the elements is not in the children node
    """
    for elem in name_list:
        assert parent_node.find(elem) is not None, \
            "%r is not in %r, child of %r" % (elem, path, parent_node.text)
