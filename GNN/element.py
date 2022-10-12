# ------------------------- Structural Element Class ------------------------- #
class Element:
    """A class of a structural element, containing an element id, type, all its features and all connections to other structural elements.
    
    Class Attributes
    ----------------
        ``featuresDict (dict)``: A dictionary with keys of element types and values of number of features.
    
    Instance Attributes
    --------------------
        ``id (str)``: The id of the structural element.
        ``type (str)``: The type of the structural element.
        ``features (list)``: All features of the structural element.
        ``connections (list)``: All connections to other structural elements
    
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
    featuresDict = {'Beam': 4, 'Column': 4, 'Slab': 5, 'Wall': 4}
    
    def __init__(self, id:str, features:list=None, connections:list=None):
        self.id = id
        
        # Initialize element type with None in case of empty dictionary or a mismatch between the dictionary and the id:
        self.type = None
        # If the dictionary is not empty, get the element type from the first letter of its id attribute:
        if self.featuresDict:
            for key in self.featuresDict.keys():
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
    
    def __str__(self) -> str:
        return f'[{self.id}],{self.features},{self.connections}'
    
    def homoFeatureCount(self):
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
        if not self.featuresDict:
            return count
        # If the dictionary is not empty, initialize it with its first value:
        else:
            count = list(self.featuresDict.values()[0])
        
        # Getting the smallest number of features from all element types:
        for value in self.featuresDict.values():
            if count < value:
                count = value
        
        return count