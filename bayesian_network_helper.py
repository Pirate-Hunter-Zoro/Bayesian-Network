def get_ancestors(network: dict) -> dict[int,list[int]]:
    """Helper method to return a mapping of nodes to their ancestors

    Args:
        network (dict): underlying bayesian network

    Returns:
        dict[int,list[int]]: mapping of nodes to their ancestors
    """
    return {i:[p for p in network[str(i)]["parents"]] for i in range(len(network))}

def topologically_sort(network: dict) -> list[int]:
    """Helper method to topologically sort variables in a bayesian network - which must be acyclic

    Args:
        network (dict): underlying bayesian network

    Returns:
        list[int]: mapping of nodes to their ancestors
    """
    level = {} # mapping of variable to "depth" of said variable
    vars = [int(i) for i in network.keys()]
    # keep track of the immediate ancestors
    ancestors = get_ancestors(network=network)
    
    def determine_level(var: int) -> int:
        """Helper method to determine the maximum number of variables to any path towards a root node in the network

        Args:
            var (int): variable in question

        Returns:
            int: said length of maximum path
        """
        if var not in level.keys():
            # need to solve this problem
            if len(ancestors[var]) == 0:
                # base case
                level[var] = 0
            else:
                record = 0
                for ancestor in ancestors[var]:
                    record =  max(record, 1 + determine_level(ancestor))
                level[var] = record
        return level[var]
    
    for v in vars:
        determine_level(var=v)

    vars.sort(key=lambda x: level[x])
    return vars