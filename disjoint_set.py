# In order perform variable elimination, ultimately we'll need to use dijoint sets as certain factors will be joined into other factors
class DisjointSetCollection:
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
            if self.__parent != other.__parent and self.__parent != self:
                self.__compress()
                self.__parent.__parent = other
            else:
                self.__parent = other
            self.__compress()

        def get_parent(self) -> int:
            self.__compress()
            return self.__parent.__id
    
    def add_element(self, id: int):
        if id not in self.__nodes.keys():
            self.__nodes[id] = DisjointSetCollection.__Node(id=id)
    
    def join(self, first_id: int, second_id: int):
        if first_id in self.__nodes.keys() and second_id in self.__nodes.keys():
            self.__nodes[first_id].join(self.__nodes[second_id])
    
    def get(self, id: int) -> int:
        return self.__nodes[id].get_parent()