# Watt-Our Optimization Package

## Important Function Specs

### LMPTimeseriesBase
- create_branch_from_df() correctly creates link of LMPs given a df of the form of LMPDataFrame from either a generated head node or the specified node (with on_node) and returns self. 

- get_node_list() should return a list of all nodes

#### Tree
- append() should add the specified new_node to the existing_node.next and refactor all relevant tree data (size and branches). If there is no specified existing node, new_node should become the head

- append_dummy() should append the specified dummy node to the specified existing node and increase the count of dummies. 

- iter_nodes() should create an iterable of all nodes

- add_branch() should add the specified tree (in the function this is called branch) to the specified node. It should update the relevant information about the tree (size, branches, and dummy_nodes)

- merge_trees() should merge the two trees by using the head of the specified first_tree for all branches (and discarding the head of the second tree). It should also update all relevant tree data (size, branches, etc.)

### optimization

- optimize_battery_control() should take a specified battery and LMPTimeseries and correctly optimize. For example, prices of 0, 10, 0 ($/MWh) with a battery of max_charge/discharge = 1 (MW) with timesteps of one hour should have an obj_value of 10. 

