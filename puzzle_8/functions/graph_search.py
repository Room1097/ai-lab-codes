from . import helpers

def bfs(initial, goal):
    frontier = [initial]  
    explored = set()

    # print(frontier)
    
    while frontier:
        # print(frontier)
        curr_node = frontier.pop(0)  
        # print(curr_node.state)
        
        if helpers.is_goal(curr_node.state, goal):
            # print('yes')
            return helpers.backtrack_path(curr_node)
        
        explored.add(tuple(curr_node.state))
        
        for action in helpers.possible_action(curr_node.state):
            new_state = helpers.make_action(curr_node.state, action)
            new_state_tuple = tuple(new_state)
            
            if new_state_tuple not in explored:
                child_node = helpers.Node(new_state, curr_node, action)
                frontier.append(child_node)  
                explored.add(new_state_tuple) 
    
    return []


def dfs(initial, goal):
    frontier = [initial]  
    explored = set()
    
    while frontier:
        curr_node = frontier.pop() 
        
        if helpers.is_goal(curr_node.state, goal):
            return helpers.backtrack_path(curr_node)
        
        explored.add(tuple(curr_node.state))
        
        for action in helpers.possible_action(curr_node.state):
            new_state = helpers.make_action(curr_node.state, action)
            new_state_tuple = tuple(new_state)
            
            if new_state_tuple not in explored:
                child_node = helpers.Node(new_state, curr_node, action)
                frontier.append(child_node)  
                explored.add(new_state_tuple) 
    
    return []
