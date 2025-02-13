import numpy as np
from disjoint_set import DisjointSet

def find_row_sum_pairs(var: int, all_vars: list[int]) -> list[tuple[int,int]]:
    """Given a variable that should be eliminated by summing over its possibilities, and the ordered list of all variables, find all of the pairs of rows that should be summed together to eliminate said variable

    Args:
        var (int): variable to eliminate
        all_vars (list[int]): ordered list of all variables

    Returns:
        list[tuple[int,int]]: list of pairs of rows that should be summed together in the corresponding array of values
    """
    idx = all_vars.index(var)
    binary_posn = len(all_vars) - 1 - idx
    row_pairs = []
    prev_second = 1
    i = 0
    for _ in range(2**(len(all_vars)-1)):
        # this is how many row pairs there will be - all other possible combinations of all other variables
        first = i if i < prev_second else prev_second+1
        second = first + 2**binary_posn
        prev_second = second
        i = first + 1
        row_pairs.append([first,second])
    return row_pairs

def find_corresponding_rows(bit_mask: list[int], common_vars: list[int], all_vars: list[int]) -> list[int]:
    """Given a list of common variables and which ones are set to true, find the rows in the distribution array that would correspond to all_vars which share the same truth value

    Args:
        bit_mask (list[int]): truth values for each of the common variables
        common_vars (list[int]): list of common variables
        all_vars (list[int]): list of all variables (common variables will be a subset)

    Returns:
        list[int]: list of rows corresponding with the truth values associated with the common variables
    """
    vars_to_posn = {v : all_vars.index(v) for v in common_vars}
    common_rows = set()
    # we will have a list of sets which we will take the intersection of
    row_sets = [set() for _ in common_vars]
    for i, v in enumerate(common_vars):
        idx = vars_to_posn[v]
        binary_posn = len(all_vars) - 1 - idx
        switch_every = 2 ** binary_posn
        # see if at this row, the given common variable's truth value is matched
        for row in range(2**len(all_vars)):
            negative_row = ((row // switch_every) % 2) == 0
            if not bit_mask[i] and negative_row:
                row_sets[i].add(row)
            elif bit_mask[i] and not negative_row:
                row_sets[i].add(row)
    # now take the intersection of all the row sets
    for row in row_sets[0]:
        missing = False
        for i in range(1, len(row_sets)):
            if row not in row_sets[i]:
                missing = True
                break
        if not missing:
            common_rows.add(row)

    result = list(common_rows)
    result.sort()
    return result

def is_subset(first: set[int], second: set[int]) -> bool:
    """Return if the first set is a subset of the second set

    Args:
        first (set[int]): supposed subset
        second (set[int]): supposed superset

    Returns:
        bool: whether the first set is a subset of the second set
    """
    for v in first:
        if v not in second:
            return False
    return True

def find_common_rows(prev_factor_vars: list[int], next_factor_vars: list[int]) -> tuple[dict[int,list[int]],list[int]]:
    """Given two (ordered) lists of variables, determine which rows of the second factor must be multiplied by each row of the first factor

    Args:
        prev_factor_vars (list[int]): variables corresponding to first factor/distribution
        next_factor_vars (list[int]): variables corresponding to second factor/distribution

    Returns:
        tuple[dict[int,list[int]],list[int]]: for each row of the first factor, which rows of the second factor should it multiply? Also, return the variable order for the joined array.
    """
    common_vars = [v for v in prev_factor_vars if v in next_factor_vars]
    common_vars_set = set(common_vars)
    
    # Each row in the previous factor corresponding to one combination of the values for the variables in common must multiply each row in the second factor 
    row_multiplications = {}
    # Calculate multiplication consistencies
    for i in range(2**len(common_vars)):
        # i represents the bit mask for all the binary values of the common variables
        # for each appearance of these shared variables, all such rows in the first factor multiply by all such rows in the second factor
        bit_mask = [0 for _ in range(len(common_vars))]
        for j in range(len(common_vars)):
            # see which variables are set to true by creating the bit mask
            if ((1 << j) & i) > 0:
                bit_mask[j] += 1
        # now we have a bit mask telling us which variables will be true - find the rows for each of the two factors where this combination of truth values for the shared variables occurs
        prev_factor_rows = find_corresponding_rows(bit_mask, common_vars, prev_factor_vars)
        next_factor_rows = find_corresponding_rows(bit_mask, common_vars, next_factor_vars)
        for prev_row in prev_factor_rows:
            row_multiplications[prev_row] = []
            for next_row in next_factor_rows:
                row_multiplications[prev_row].append(next_row)

    # TODO - fix this joined variable order...
    prev_vars = set(prev_factor_vars)
    next_vars = set(next_factor_vars)
    if is_subset(prev_vars, next_vars):
        joined_vars = next_factor_vars
    elif is_subset(next_vars, prev_vars):
        joined_vars = prev_factor_vars
    else:
        joined_vars = [v for v in prev_factor_vars] + [v for v in next_factor_vars if v not in common_vars_set]
    return (row_multiplications, joined_vars)

def join_factors(var: int, eliminate: bool, relevant_factors: list[tuple[list[int],np.array]]) -> tuple[list[int],np.array]:
    """Given which variable we want to eliminate, multiply its respective factors together and then sum out over the variable to eliminate

    Args:
        var (int): the variable (possibly to eliminate)
        eliminate (bool): whether said variable will be eliminated
        factors (list[tuple[list[int],np.array]]): list of distributions corresponding to this variable

    Returns:
        tuple[list[int],np.array]: new list of relevant variables to this array along with the array (a new distribution)
    """
    # We need to find out which rows multiply together
    # Break into pairs
    prev_factor = relevant_factors[0]
    for i in range(1, len(relevant_factors)):
        next_factor = relevant_factors[i]
        # Find the list of rows corresponding to each factor that must multiply together
        row_multiplications, joined_vars = find_common_rows(prev_factor[0], next_factor[0])
        # Create a new array to populate with products
        merged_array = np.zeros(shape=2**len(joined_vars))
        for prev_idx in row_multiplications:
            for next_idx in row_multiplications[prev_idx]:
                # TODO - calculate curr_idx based on prev_idx and next_idx
                merged_array[max(prev_idx, next_idx)] = prev_factor[1][prev_idx] * next_factor[1][next_idx]
        # Set up for the next two multiplications
        merged = (joined_vars, merged_array)
        prev_factor = merged

    # If eliminating, sum over the variable to eliminate it
    resulting_variables = prev_factor[0]
    resulting_array = prev_factor[1]
    summed_array = np.zeros(len(resulting_array)//2) if eliminate else resulting_array
    if eliminate:
        # Find the corresponding pairs of rows that must be summed together
        row_pairs = find_row_sum_pairs(var, resulting_variables)
        for i, row_pair in enumerate(row_pairs):
            summed_array[i] = resulting_array[row_pair[0]] + resulting_array[row_pair[1]]

    return [v for v in resulting_variables if (v != var if eliminate else True)], summed_array


def relevant(parents: list[int], parents_in_evidence: list[int], evidence: dict[int,bool], row: int) -> bool:
    """Based on the parents' order for a given probability array, which parents are in the evidence, and whether those parents are true or false, determine if the row is consistent with the evidence based on our probability array construction convention

    Args:
        parents (list[int]): list of parents in order for some probability array
        parents_in_evidence (list[int]): list of parents which are in the evidence
        evidence (dict[int,bool]): records which variables in the evidence are true or false
        row (int): row index to determine if consistent with evidence

    Returns:
        bool: whether row number is consistent with evidence
    """
    for parent in parents_in_evidence:
        parent_idx = parents.index(parent)
        binary_posn = len(parents) - 1 - parent_idx
        switch_every = 2 ** binary_posn
        negative_row = ((row // switch_every) % 2) == 0
        if (negative_row and evidence[parent]) or ((not negative_row) and (not evidence[parent])):
            return False
    return True

def handle_vars(vars: list[int], eliminate: bool, factor_index_to_factor: dict[int,tuple[list[int],np.array]], factor_tracker: DisjointSet, var_to_factor_indices: dict[int,list[int]]) -> np.array:
    """Join the list of factors, with a flag to determine if said variables are to be eliminated

    Args:
        vars (list[int]): list of variables
        eliminate (bool): whether said variables are to be eliminated
        factor_index_to_factor (dict[int,tuple[list[int],np.array]]): mapping factor indices to the factors themselves
        factor_tracker (DisjointSet): to handle joining of merged factors by index
        var_to_factor_indices (dict[int,list[int]]): mapping variables to list of relevant factor indices that include said variable

    Returns:
        np.array: final distribution of numbers
    """
    array = None
    for var in vars:
        unique_factor_indices = set([factor_tracker.get(f_idx) for f_idx in var_to_factor_indices[var]])
        unique_factors = [factor_index_to_factor[f_idx] for f_idx in unique_factor_indices]
        resulting_vars, array = join_factors(var=var, eliminate=eliminate, relevant_factors=unique_factors)
        join_base = None
        # Factors are now merged - so update factor tracker accordingly
        for f_idx in unique_factor_indices:
            if join_base == None:
                join_base = f_idx
            factor_index_to_factor[f_idx] = (resulting_vars, array)
            factor_tracker.join(f_idx, join_base)
    return array

def create_factors(network: dict, evidence: dict[int,bool]) -> tuple[dict[int, tuple[list[int],np.array]], dict[int,set[int]]]:
    """Create the two dictionaries that correspond with the factors created from this bayesian network

    Args:
        network (dict): the network in question
        evidece (dict[int,bool]): evidence variables with their values

    Returns:
        tuple[dict[int, tuple[list[int],np.array]], dict[int,set[int]]]: map of factor ids to the factors themselves as well a map of variable ids to the set of factors pertaining to said variable
    """
    factor_index_to_factor = {}
    var_to_factor_indices = {i: set() for i in range(len(network))}
    for i, info in network.items():
        i = int(i)

        parents = info["parents"]
        parents_in_evidence = [p for p in parents if p in evidence.keys()]
        parents_not_in_evidence = [p for p in parents if p not in evidence.keys()]
        # obtain the array of probabilities before considering evidence
        prob_array = np.array([pair[1] for pair in info["prob"]])

        # now filter the probability array so that it only contains entries consistent with the evidence
        relevant_rows = []
        for row in range(len(prob_array)):
            if relevant(parents, parents_in_evidence, evidence, row):
                relevant_rows.append(row)
        # quick sanity check
        assert len(relevant_rows) == 2 ** len(parents_not_in_evidence)

        # our base array size depends on the number of parents not in our evidence (and so take on both true and false values)
        new_prob_array = np.zeros(len(relevant_rows))
        # fill new_prob_array with relevant rows
        for j, row in enumerate(relevant_rows):
            new_prob_array[j] = prob_array[row]

        # double the size of our array so that it can contain the probabilities of 'i' being true and false
        # BUT only if we are not in the evidence
        final_prob_array = np.zeros((2 if i not in evidence.keys() else 1) * len(new_prob_array))
        for j in range(len(new_prob_array)):
            if i not in evidence.keys():
                final_prob_array[2*j] = 1 - new_prob_array[j]
                final_prob_array[2*j+1] = new_prob_array[j]
            elif evidence[i]:
                # i is set to true in evidence
                final_prob_array[j] = new_prob_array[j]
            else:
                # i is set to false in evidence
                final_prob_array[j] = 1 - new_prob_array[j]

        # finally store into a tuple - and only count 'i' as one of the variables if it is not in the evidence
        prob_tuple = ([parent for parent in parents_not_in_evidence] + ([i] if i not in evidence.keys() else []), final_prob_array)
        for var in prob_tuple[0]:
            var_to_factor_indices[var].add(len(factor_index_to_factor))
        factor_index_to_factor[len(factor_index_to_factor)] = prob_tuple

    return factor_index_to_factor, var_to_factor_indices

def calculate_probability(var: int, others: dict[int,bool], network: dict) -> float:
    """Given a variable and set values for all other variables in the above Bayesian Network, calculate the the probability of var being true

    Args:
        var (int): variable in question
        others (dict[int,bool]): all other variable values

    Returns:
        bool: probability that var is true given all other variable values
    """
    # join on the factors that matter to var
    factor_idx_to_factor, var_to_factor_indices = create_factors(network=network, evidence=others) 
    # the following result should be a 2D vector - var is False or var is True
    var_distribution = join_factors(var, eliminate=False, relevant_factors=[factor_idx_to_factor[i] for i in var_to_factor_indices[var]])[1]
    var_distribution = var_distribution / np.sum(var_distribution)
    return var_distribution[1] # probability of var being true