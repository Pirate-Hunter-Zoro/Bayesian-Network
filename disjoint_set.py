# In order perform variable elimination, ultimately we'll need to use dijoint sets as certain factors will be joined into other factors
class DisjointSet:
    def __init__(self):
        self.__nodes = {}

    class __Node:
        def __init__(self, id: int):
            self.__id = id
            self.__parent = self

        def __compress(self):
            if self.__parent != self:
                self.__parent.__compress()
                self.__parent = self.__parent.__parent
        
        def join(self, other):
            self.__compress()
            other.__compress()
            if self.__parent != self:
                self.__parent.join(other)
            elif other.__parent != other:
                self.join(other.__parent)
            else:
                # both are parent nodes
                self.__parent = other
            self.__compress()
            other.__compress()

        def get_parent(self) -> int:
            return self.__parent.__id
    
    def add_element(self, id: int):
        if id not in self.__nodes.keys():
            self.__nodes[id] = DisjointSet.__Node(id=id)
    
    def join(self, first_id: int, second_id: int):
        if first_id in self.__nodes.keys() and second_id in self.__nodes.keys():
            self.__nodes[first_id].join(self.__nodes[second_id])
    
    def get(self, id: int) -> int:
        return self.__nodes[id].get_parent()