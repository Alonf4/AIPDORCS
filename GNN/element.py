# ------------------------- Structural Element Class ------------------------- #
class Element:
    """A class of a structural element, containing an element id, type, all its features and all connections to other structural elements.
    
    Class Attributes
    ----------------
        ``featuresDict (dict)``: A dictionary with keys of element types and values of number of features.
        ``countDict (dict)``: A dictionary containing the number of elements for each type, and the total existing number of elements.
    
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
    countDict = {'Total': 0}
    
    def __init__(self, id:str, features:list=None, connections:list=None):
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
        if not self.type in Element.countDict.keys():
            Element.countDict[self.type] = 0
        # Increase the total number of elements for each new element instance:
        Element.countDict['Total'] += 1
        Element.countDict[self.type] += 1
    
    def __del__(self):
        # Decrease the total number of elements for each deleted element instance:
        Element.countDict['Total'] -= 1
        Element.countDict[self.type] -= 1
    
    def __str__(self) -> str:
        return f'[{self.id}],{self.features},{self.connections}'
    
    @classmethod
    def homoFeatureCount(cls):
        """Based on the class attribute ``featuresDict (dict)``, return the smallest number of features.
        
        Returns
        -------
        ``count (int)``: The smallest number of features an element type can have.
        
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
            count = list(cls.featuresDict.values()[0])
        
        # Getting the smallest number of features from all element types:
        for value in cls.featuresDict.values():
            if count < value:
                count = value
        
        return count
    
    @classmethod
    def totalCount(cls):
        return cls.countDict['Total']
    
    def typeCount(self):
        return Element.countDict[self.type]