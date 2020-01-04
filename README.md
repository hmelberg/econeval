# Welcome to EconEval
(Work in progress, not ready for production)

## define a model
hip = nx.DiGraph()

## add states to the model
states = [('healthy', {'utility':0.9, 'cost': 0}),
          ('dead', {'utility':0, 'cost': 0}),
          ('sick', {'utility':0.7, 'cost': 1000})]
hip.add_nodes_from(states)

## add transition probabilities
transitions = [('healthy', 'healthy', {'p':0.98}),
               ('healthy', 'sick', {'p':0.01}),
               ('healthy', 'dead', {'p':0.01}),
               ('sick', 'sick', {'p':0.5}),
               ('sick', 'dead', {'p':0.3}),
               ('sick', 'healthy', {'p':0.2}),
               ('dead', 'dead', {'p':1})]

hip.add_edges_from(transitions)

## analyse

### Valculate cost-efficiency
hip.cost_per_qualy()

### Probability sensitivity analysis
hip.psa()

### Value of information analysis
hip.evpi()

