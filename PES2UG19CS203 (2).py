import queue
import copy
"""
You can create any other helper funtions.
Do not modify the given functions
"""

"""
PES2UG19CS203-T.LOHITH SRINIVAS
"""



def A_star_Traversal(cost, heuristic, start_point, goals):
    n = len(cost)                                              
    visited = [0 for i in range(n)]                             
    frontier_priority_queue = queue.PriorityQueue()             
    frontier_priority_queue.put((heuristic[start_point], ([start_point], start_point, 0)))
    while(frontier_priority_queue.qsize() != 0):
        total_estimated_cost, nodes_tuple = frontier_priority_queue.get()
        A_star_path_till_node = nodes_tuple[0]
        node = nodes_tuple[1]
        node_cost = nodes_tuple[2]
        if visited[node] == 0:
            visited[node] = 1
            if node in goals:
                return A_star_path_till_node
            for neighbour_node in range(1, n):
                if cost[node][neighbour_node] > 0 and visited[neighbour_node] == 0:
                    total_cost_till_node = node_cost + cost[node][neighbour_node]
                    estimated_total_cost = total_cost_till_node + heuristic[neighbour_node]
                    A_star_path_till_neighbour_node = copy.deepcopy(A_star_path_till_node)
                    A_star_path_till_neighbour_node.append(neighbour_node)
                    frontier_priority_queue.put((estimated_total_cost, (A_star_path_till_neighbour_node, neighbour_node, total_cost_till_node)))
    return list()






def DFS_Traversal_HELPer(cost,start_point,goals,Visited,path):
        if start_point not in Visited:
            Visited.add(start_point)
            path.append(start_point)
            if(start_point not in goals):
                for k in range (len(cost[start_point])):
                    if cost[start_point][k]>0:
                        if k not in Visited:
                           DFS_Traversal_HELPer(cost,k,goals,Visited,path)
                           break
            
                



def DFS_Traversal(cost, start_point, goals):
    """
    Perform DFS Traversal and find the optimal path 
        cost: cost matrix (list of floats/int)
        start_point: Staring node (int)
        goals: Goal states (list of ints)
    Returns:
        path: path to goal state obtained from DFS(list of ints)
    """
    path = []
    # TODO
    Visited=set()
    DFS_Traversal_HELPer(cost,start_point,goals,Visited,path)
    return path
