# ---------------------------------------------------------------------------- #
#                           Structural Element Class                           #
# ---------------------------------------------------------------------------- #
class Element:
    """A class of a structural element, containing an element id, type, all its features and all connections to other structural elements.
    
    Class Attributes
    ----------------
        ``featuresDict (dict)``: A dictionary with keys of element types and values of number of features.
    
    Instance Attributes
    -------------------
        ``id (str)``: The id of the structural element.
        ``type (str)``: The type of the structural element.
        ``features (list)``: All features of the structural element.
        ``connections (list)``: All connections to other structural elements
    
    Class Methods
    -------------
        ``homoFeatureCount()``: Returns the smallest number of features of all types, for a uniform number of features.
        ``totalCount()``: Returns the number of existing elements overall.
    
    Instance Methods
    ----------------
        ``typeCount()``: Returns the number of existing elements of this type.
    
    Examples
    --------
    >>> # Creating structural elements:
    >>> # (If connections or features are not specified, empty lists will be created).
    >>> element1 = Element('B1')
    >>> element2 = Element('B2', [1, 2, 3])
    >>> element3 = Element('B3', [1, 2, 3, 4], ['B1', 'B2'])
    >>> # Printing a structural element:
    >>> print(element2)
    >>> # Output: '[B1],[1, 2, 3],[]'
    """
    featuresDict = {}
    _countDict = {'Total': 0}
    
    # ---------------------------- Constructor Method ---------------------------- #
    def __init__(self, id:str, features:list=None, connections:list[str]=None):
        self.id = id
        
        # Initialize element type with None in case of empty dictionary or a mismatch between the dictionary and the id:
        self.type = None
        # If the dictionary is not empty, get the element type from the first letter of its id attribute:
        if Element.featuresDict:
            for key in Element.featuresDict.keys():
                if self.id[0] == key[0]:
                    self.type = key
        
        # If 'features' list is not specified, define an empty list for the element instance:
        if features:
            self.features = features
        else:
            self.features = []
        
        # If 'connections' list is not specified, define an empty list for the element instance:
        if connections:
            self.connections = connections
        else:
            self.connections = []
        
        # If this is the first instance of this type, initialize the counter for this type:
        if not self.type in Element._countDict.keys():
            Element._countDict[self.type] = 0
        # Increase the total number of elements for each new element instance:
        Element._countDict['Total'] += 1
        Element._countDict[self.type] += 1
    
    # ----------------------------- Destructor Method ---------------------------- #
    def __del__(self):
        # Decrease the total number of elements for each deleted element instance:
        Element._countDict['Total'] -= 1
        Element._countDict[self.type] -= 1
    
    def __str__(self) -> str:
        return f'[{self.id}],{self.features},{self.connections}'
    
    # ---------------- Get Number of Features for Homogenous Graph --------------- #
    @classmethod
    def homoFeatureCount(cls) -> int:
        """Based on the class attribute ``featuresDict (dict)``, return the smallest number of features.
        
        Returns
        -------
        ``(int)``: The smallest number of features an element type can have.
        
        Examples
        --------
        >>> Element.featuresDict = {'A': 2, 'B': 2, 'C': 3}
        >>> print(Element.homoFeatureCount())
        >>> # Output: '2'
        """
        count = 0
        
        # If the dictionary is empty, there are no features:
        if not cls.featuresDict:
            return count
        # If the dictionary is not empty, initialize it with its first value:
        else:
            count = list(cls.featuresDict.values())[0]
        
        # Getting the smallest number of features from all element types:
        for value in cls.featuresDict.values():
            if count > value:
                count = value
        
        return count
    
    # ------------------- Get Total Number of Elements Existing ------------------ #
    @classmethod
    def totalCount(cls):
        return cls._countDict['Total']
    
    # ----------------- Get the Number of Elements from this Type ---------------- #
    def typeCount(self):
        return Element._countDict[self.type]

# ---------------------------------------------------------------------------- #
#                               Graph Node Class                               #
# ---------------------------------------------------------------------------- #
class Node:
    """A class of a graph nodes, containing a node id and a structural element.
    
    Instance Attributes
    -------------------
        ``nodeID (int)``: The id of the graph node.
        ``element (Element)``: A structural element class instance.
    
    Instance Methods
    ----------------
        ``getNodeAsList()``: Returns the node as a list for writing as a row in a CSV file.
    """
    # ---------------------------- Constructor Method ---------------------------- #
    def __init__(self, nodeID:int, element:Element):
        self.nodeID = nodeID
        self.element = element
    
    # ---------------- Converting Node to List for Nodes.csv File ---------------- #
    def getNodeAsList(self):
        featureCount = self.element.homoFeatureCount()
        return [self.nodeID] + [self.element.id] + self.element.features[:featureCount]

# ---------------------------------------------------------------------------- #
#                               Graph Edge Class                               #
# ---------------------------------------------------------------------------- #
class Edge:
    """A class of a graph edge, containing an edge id, a source node and a destination node.
    
    Instance Attributes
    -------------------
        ``edgeID (int)``: The id of the graph edge.
        ``src (Node)``: A graph node class instance as the source of the edge.
        ``dst (Node)``: A graph node class instance as the destination of the edge.
    
    Instance Methods
    ----------------
        ``getEdgeAsList()``: Returns the edge as a list for writing as a row in a CSV file.
    
    Examples
    --------
    >>> # Checking if two edges contain the same graph nodes in any direction:
    >>> edge1 = Edge(0, node1, node2)
    >>> edge2 = Edge(1, node2, node1)
    >>> print(e1 == e2)
    >>> # Output: 'True'
    >>> # This test can also be helpful in finding an edge in a list:
    >>> eList = [edge1]
    >>> print(edge2 in eList)
    >>> # Output: 'True'
    """
    # ---------------------------- Constructor Method ---------------------------- #
    def __init__(self, edgeID:int, src:Node, dst:Node):
        self.edgeID = edgeID
        self.src = src
        self.dst = dst
    
    # ----------------- Check if Two Edges Contain the same Nodes ---------------- #
    def __eq__(self, other) -> bool:
        return ((self.src.element.id == other.src.element.id) and (self.dst.element.id == other.dst.element.id)) \
            or ((self.src.element.id == other.dst.element.id) and (self.dst.element.id == other.src.element.id))
    
    # ---------------- Converting Edge to List for Edges.csv File ---------------- #
    def getEdgeAsList(self):
        return [self.src.nodeID] + [self.dst.nodeID]