import math
import numpy as np
import random
import copy
from bayesian_network_helper import topologically_sort
from disjoint_set import DisjointSetCollection

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
    if len(common_vars) == 0:
        # then return all rows
        return [i for i in range(1 << len(all_vars))]
    else:
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

def find_common_rows(prev_factor_vars: list[int], next_factor_vars: list[int]) -> tuple[dict[int,list[int]],list[int]]:
    """Given two (ordered) lists of variables, determine which rows of the second factor must be multiplied by each row of the first factor

    Args:
        prev_factor_vars (list[int]): variables corresponding to first factor/distribution
        next_factor_vars (list[int]): variables corresponding to second factor/distribution

    Returns:
        tuple[dict[int,list[int]],list[int]]: for each row of the first factor, which rows of the second factor should it multiply? Also, return the variable order for the joined array.
    """
    common_vars = [v for v in prev_factor_vars if v in next_factor_vars]
    
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

    joined_vars = [v for v in prev_factor_vars if v not in common_vars] + [v for v in next_factor_vars]
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
        merged_array = np.zeros(shape=1<<len(joined_vars))
        idx = 0
        for prev_idx in row_multiplications:
            for next_idx in row_multiplications[prev_idx]:
                merged_array[idx] = prev_factor[1][prev_idx] * next_factor[1][next_idx]
                idx += 1
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

def handle_vars(vars: list[int], eliminate: bool, factor_index_to_factor: dict[int,tuple[list[int],np.array]], factor_tracker: DisjointSetCollection, var_to_factor_indices: dict[int,list[int]]) -> np.array:
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
    # It is possible some of the query variables are independent - ultimately take all of our remaining factors and join them together
    arrays_to_join = []
    factors_to_join = set()
    for var in vars:
        unique_factor_indices = list(set([factor_tracker.get(f_idx) for f_idx in var_to_factor_indices[var]]))
        # Sanity check
        assert len(unique_factor_indices) == 1
        next_factor_idx = unique_factor_indices[0]
        if next_factor_idx not in factors_to_join:
            factors_to_join.add(unique_factor_indices[0])
            arrays_to_join.append(factor_index_to_factor[next_factor_idx][1])

    return join_distributions(arrays_to_join)

def create_factors(network: dict, evidence: dict[int,bool]) -> tuple[dict[int, tuple[list[int],np.array]], dict[int,set[int]]]:
    """Create the two dictionaries that correspond with the factors created from this bayesian network

    Args:
        network (dict): the network in question
        evidece (dict[int,bool]): evidence variables with their values

    Returns:
        tuple[dict[int, tuple[list[int],np.array]], dict[int,set[int]]]: map of factor ids to the factors themselves as well a map of variable ids to the set of factors pertaining to said variable
    """
    factor_index_to_factor = {}
    var_to_factor_indices = {int(v): set() for v in network.keys()}
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

def perform_gibbs_sampling(non_evidence_variables: list[int], current_evidence: dict[int,bool], network: dict, queries: list[int]) -> int:
    """Perform one round of Gibbs sampling given a bayesian network - and return the row in the corresponding probability distribution whose count we should update

    Args:
        non_evidence_variables (list[int]): variables to determine probabilistically
        current_evidence (dict[int,bool]): current set of values assigned to each variable
        network (dict): underlying bayesian network
        queries (list[int]): list of query variables whose values we are interested in calculating the probability for

    Returns:
        int: corresponding row in the likelihood array
    """

    # perform the Gibbs algorithm this many times
    var_to_change = non_evidence_variables[int(random.random()*len(non_evidence_variables))]
    del current_evidence[var_to_change]
    # simulate the probability of var_to_change being true given everything else
    p = calculate_probability(var_to_change, current_evidence, network)
    current_evidence[var_to_change] = (random.random() < p)
    # see how our queries showed up
    query_values = [1 if current_evidence[q] else 0 for q in queries]
    # find the entry in our probability distribution that corresponds with this combination of query values
    posn = find_corresponding_rows(query_values, queries, queries)[0]
    # return the entry index so that the caller can update 
    return posn

def perform_likelihood_weighting(network: dict, sorted_vars: list[int], queries: list[int], evidence: dict[int,bool], current_evidence: dict[int, bool]) -> tuple[float, int]:
    """Perform one round of likelihood weighting given a bayesian network - assign variable values to the current_evidence dictionary and return the corresponding weight and position in a probability distribution that should be updated

    Args:       
        network (dict): underlying bayesian network
        sorted_vars (list[int]): topologically sorted list of variables (helpful to pass so that resorting need not be done when this function is called periodically)
        queries (list[int]): list of query variables
        evidence (dict[int,bool]): set of evidence variables with pre-determined values
        current_evidence (dict[int, bool]): currently observed evidence that can be changed - represents current state of our network

    Returns:
        tuple[float, int]: weight associated with the random variable assignment as well as the index in the probability distribution that would warrant an update
    """
    weight = 1.0

    for var in sorted_vars:
        probabilities = [pair[1] for pair in network[str(var)]["prob"]]
        parents = network[str(var)]["parents"]
        prob_true = 0.0
        if len(parents) > 0:
            # look at the parents to determine probability of being true
            row = find_corresponding_rows([1 if current_evidence[p] else 0 for p in parents], parents, parents)[0]
            prob_true += probabilities[row]
        else:
            prob_true += probabilities[0]

        # if the variable is in evidence, update the weight
        if var in evidence.keys():
            weight *= prob_true if evidence[var] else (1 - prob_true)
        else: # otherwise set in the current evidence the value of this variable to be according to the probability given ancestors' values
            current_evidence[var] = random.random() < prob_true
        
    query_values = [1 if current_evidence[q] else 0 for q in queries]
    # find the entry in our probability distribution that corresponds with this combination of query values
    posn = find_corresponding_rows(query_values, queries, queries)[0]

    return weight, posn

def find_weight(current_evidence: dict[int,bool], original_evidence: dict[int,bool], network: dict) -> float:
    """Helper method to calculate the likelihood weight of a certain assignment for all variables given which variables were the original evidence

    Args:
        current_evidence (dict[int,bool]): all variable assignments
        original_evidence (dict[int,bool]): original set of evidence variables with their values
        network (dict): underlying bayesian network

    Returns:
        float: likelihood weighting for the total variable assignment
    """
    weight = 1.0
    for v in original_evidence.keys():
        parents = network[str(v)]["parents"]
        probabilities = [pair[1] for pair in network[str(v)]["prob"]]
        prob_true = 0.0
        if len(parents) > 0:
            # look at the parents to determine probability of being true
            row = find_corresponding_rows([1 if current_evidence[p] else 0 for p in parents], parents, parents)[0]
            prob_true += probabilities[row]
        else:
            prob_true += probabilities[0]
        weight *= prob_true if original_evidence[v] else (1 - prob_true)
    
    return weight

def disect_trees(network: dict) -> dict[int,dict]:
    """Helper function to take a polytree Bayesian Network and return a map from each variable to its respective subtree bayesian network

    Args:
        network (dict): polytree bayesian network

    Returns:
        dict[int,dict]: mapping from each node to its respective polytree
    """
    node_sets = DisjointSetCollection()
    for v in network.keys():
        node_sets.add_element(id=int(v))
    
    # Each disjoint set in the collection corresponds with a different bayesian network
    for v in network.keys():
        # Join this node with all its parents
        node_v = int(v)
        for parent in network[v]["parents"]:
            node_sets.join(node_v, parent)
    
    # Now create a set of dictionaries - each a connected bayesian network (directed acyclic graph)
    network_map = {}
    for v in network.keys():
        node_v = int(v)
        respective_set_idx = node_sets.get(node_v)
        if respective_set_idx not in network_map.keys():
            network_map[respective_set_idx] = {}
        # Copy the entry into this subset network
        network_map[respective_set_idx][v] = network[v]

    return network_map

def handle_dag_variable_elimination(dag: dict, queries: list[int], evidence: dict[int,bool]) -> np.array:
    """Helper method to compute variable elimination when only dealing with a directed acyclic graph and not a polytree

    Args:
        dag (dict): connected bayesian network
        queries (list[int]): list of query variables
        evidence (dict[int,bool]): set of evidence variables with their values

    Returns:
        np.array: probability distribution pertaining to the query variable values
    """
    if len(queries) == 0: # we do not care about any of the variables in this dag
        return np.array([1.0])

    # grab the list of factors and each factor has its own probability distribution - which will depend on its parents should they exist
    factor_index_to_factor, var_to_factor_indices = create_factors(dag, evidence)

    # ultimately, factors will be merged, and thus their index will correspond to the same set variable - we do not want that showing up multiple times when we consider the relevant factors to a variable below
    factor_tracker = DisjointSetCollection()
    for i in range(len(factor_index_to_factor)):
        factor_tracker.add_element(id=i)

    # figure out which variables need to be eliminated
    query_set = set(queries)
    evidence_set = set([pair[0] for pair in evidence.items()])
    hidden_vars = [int(i) for i in dag.keys() if int(i) not in query_set and int(i) not in evidence_set]
    # sort the hidden variables by the number of relevant factors
    hidden_vars.sort(key=lambda x : -len(var_to_factor_indices[x]))

    # now we go through and eliminate each hidden variable
    handle_vars(vars=hidden_vars, eliminate=True, factor_index_to_factor=factor_index_to_factor, factor_tracker=factor_tracker,var_to_factor_indices=var_to_factor_indices)
    # this function also returns a factor
    result = handle_vars(vars=queries, eliminate=False, factor_index_to_factor=factor_index_to_factor, factor_tracker=factor_tracker,var_to_factor_indices=var_to_factor_indices)
    return result / np.sum(result) # for normalization

def handle_dag_gibbs_sampling(iterations: int, network: dict, queries: list[int], evidence: dict[int,bool]) -> np.array:
    """Helper method to handle gibbs sampling for one directed acyclic bayesian network, with its respective list of query variables and evidence

    Args:
        iterations (int): number of iterations to perform this
        network (dict): directed acyclic bayesian network
        queries (list[int]): list of query variables
        evidence (dict[int,bool]): map of evidence variables to their values

    Returns:
        np.array: probability distribution corresponding to the query variable values
    """
    if len(queries) == 0:
        # we do not care about any of the variables in this polytree
        return np.array([1.0])

    prob_distribution = np.zeros(shape=1<<len(queries))
    
    # create list of non-evidence variables
    non_evidence_variables = [int(i) for i in network.keys() if int(i) not in evidence.keys()]
    
    current_evidence = copy.deepcopy(evidence)
    # randomly initialize all the variables
    for v in non_evidence_variables:
        current_evidence[v] = (random.random() < 0.5)

    for _ in range(iterations):
        posn = perform_gibbs_sampling(non_evidence_variables, current_evidence, network, queries)
        prob_distribution[posn] += 1.0

    return prob_distribution / iterations # normalization into a probability

def handle_dag_likelihood_weighting(iterations: int, network: dict, queries: list[int], evidence: dict[int,bool]) -> np.array:
    """Helper method to handle likelihood weighting for one directed acyclic bayesian network for the given number of iterations

    Args:
        iterations (int): number of iterations to run the algorithm
        network (dict): underlying DAG bayesian network
        queries (list[int]): list of query variables
        evidence (dict[int,bool]): set of evidence variables with their values

    Returns:
        np.array: resulting probability distribution for the query variables
    """
    if len(queries) == 0:
        # not interested in any variables in this DAG network
        return np.array([1.0])

    prob_distribution = np.zeros(shape=1<<len(queries))
    
    # we're going to need to sort all of our variables topologically, because that's the order that we'll need to set their values to
    sorted_vars = topologically_sort(network=network)
    
    for _ in range(iterations):
        current_evidence = copy.deepcopy(evidence)
        # go through each non-evidence variable and assign it a value randomly according to its probability
        weight, posn = perform_likelihood_weighting(network, sorted_vars, queries, evidence, current_evidence)
        # update the distribution accordingly
        prob_distribution[posn] += weight

    return prob_distribution / np.sum(prob_distribution) # normalization into a probability

def handle_dag_metropolis_hastings(iterations: int, p: float, network: dict, queries: list[int], evidence: dict[int,bool]) -> np.array:
    """Helper method to perform metropolis hastings approximate sampling on a directed acyclic single-tree bayesian network

    Args:
        iterations (int): number of iterations to run the algorithm
        p (float): probability of using Gibbs Sampling to yield the next state during each iteration
        network (dict): underlying DAG bayesian network
        queries (list[int]): list of query variables
        evidence (dict[int,bool]): set of evidence variables with their values

    Returns:
        np.array: resulting probability distribution for the query variables
    """
    if len(queries) == 0:
        return np.array([1.0])

    # We'll use the alternative method fo metropolis hastings described in the README
    prob_distribution = np.zeros(shape=1 << len(queries))

    # create list of non-evidence variables
    non_evidence_variables = [int(i) for i in network.keys() if int(i) not in evidence.keys()]

    # also create list of topologically sorted variables
    sorted_vars = topologically_sort(network=network)

    prev_weighted_likelihood = 0.0
    for _ in range(iterations):
        current_evidence = copy.deepcopy(evidence)
        if random.random() < p:
            # randomly initialize all the variables
            for v in non_evidence_variables:
                current_evidence[v] = (random.random() < 0.5)
            # Perform Gibbs sampling for our next state and accept it
            posn = perform_gibbs_sampling(non_evidence_variables, current_evidence, network, queries)
            prev_weighted_likelihood = find_weight(current_evidence, evidence, network)
            # update the distribution accordingly
            prob_distribution[posn] += 1.0
        else:
            # Perform likelihood weighting for our next state, but only move to it if the state is more likely than our previous state
            old_evidence = copy.deepcopy(evidence)
            weight, posn = perform_likelihood_weighting(network, sorted_vars, queries, evidence, current_evidence)
            if weight < prev_weighted_likelihood:
                # Reject the next state because it has a lower probability than the previous one obtained from likelihood weighting
                current_evidence = old_evidence
            else:
                # Accept the next state
                prev_weighted_likelihood = weight
                prob_distribution[posn] += weight

    return prob_distribution / np.sum(prob_distribution)

def break_up_polytree(entire_network: dict, queries: list[int], evidence: dict[int,bool]) -> tuple[dict[int,dict], dict[int,list[int]], dict[int,list[int]]]:
    """Helper method to break up a polytree into individual trees, and for each tree keep track of which query and evidence variables it has

    Args:
        entire_network (dict): polytree bayesian network
        queries (list[int]): list of query variables
        evidence (dict[int,bool]): evidence variables with their values

    Returns:
        tuple[dict[int,dict], dict[int,list[int]], dict[int,list[int]]]: mapping from nodes to their respective DAGS along with a mapping from each DAG to its query variables, and evidence variables
    """
    # if given a polytree, break it up into different trees
    dag_map = disect_trees(entire_network)
    # map of each dag index to all of its query and evidence variables
    query_collections = {} 
    evidence_collections = {}
    for i, dag in dag_map.items():
        query_collections[i] = []
        evidence_collections[i] = []
        for v in queries:
            if str(v) in dag.keys():
                query_collections[i].append(v)
        for v in evidence.keys():
            if str(v) in dag.keys():
                evidence_collections[i].append(v)
        # we'll sort the variables, which will affect the value order in the soon-to-be-calulated probability distributions
        query_collections[i].sort()
    return dag_map, query_collections, evidence_collections

def join_distributions(distributions: list[np.array]) -> np.array:
    """Helper method to join a list of independent probability distributions together

    Args:
        distributions (list[np.array]): list of probability distributions

    Returns:
        np.array: resulting merged numpy array
    """
    num_vars = 0
    for arr in distributions:
        num_vars += int(math.log2(len(arr)))
    
    # join together two at a time
    product = distributions[0]
    for arr in distributions[1:]:
        intermediate = np.zeros(shape=len(arr)*len(product))
        idx = 0
        for i in range(len(product)):
            for j in range(len(arr)):
                intermediate[idx] = product[i] * arr[j]
                idx += 1
        product = intermediate

    # sanity check
    assert len(product) == (1 << num_vars)
    return product