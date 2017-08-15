# -*- coding: utf-8 -*-
"""
Created on Fri Aug  4 14:53:46 2017

@author: hmelberg
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
import random
from nxpd import draw


#%% helper functions

def dist(name, *args, **kwargs):
    """returns a random number 
    
    name: string
        name of distribution
    
    args: parameters of the distribution, position based
    
    kwargs:parameters of the distribution, keyword based
    
    example
    -------
    >>> dist('normal', 5, 2)
        
    """
    return eval(f'np.random.{name}(*{args}, **{kwargs})')

                 
#%% EconEval model definition class 

class DecisionTree(object):
    def __init__(self):      
        self.data={}
        self.edge={}
    
    def add_data(self, name, data):
        self.data[name] = data
    
    def add_data_csv(self, name, file, **kwargs):
        self.data[name] = pd.read_csv(file, kwargs)
    
    def lookup_dict(self, lookup):
        """
        lookup_dict(mortality[('male', 'above 65'])
        """
        #hmm security problem here?
        value=eval(f"self.{lookup}")
        return value
    
    def lookup_df(self, df, query):
        """
        lookup=(df='mortality', gender='male', age=='above 65'])
        """
        df.query
        return self.data[name][labels]

    @property
    def datalist(self):
        return self.data.keys()
    
    def import_from_nx(self, nxg):
        """
        Import dec tree from a networkx graph structure
        
        nxg: a network graph 
           
        """
        self.add_many_states(nxg.nodes(data=True))
        self.add_many_edges(nxg.edges(data=True))   
       
    def _to_nx_state(self):
        statelist = []
        for state in self.state:
            info=(state, self.state[state])
            statelist.append(info)
        return statelist
        
    def _to_nx_edge(self):    
        edgelist = []
        for from_state in self.edge:
            for to_state in self.edge[from_state].keys():
                info=(from_state, to_state, self.edge[from_state][to_state])
                edgelist.append(info)
        return edgelist
    
    def plot(self):
        states = self._to_nx_state()
        edges = self._to_nx_edge()
        g = nx.DiGraph()
        g.add_nodes_from(states)
        g.add_edges_from(edges)
            
        return draw(g, format = 'svg')
        
    def add_many_states(self, state, attr=None, order=None):
        """
            Adds multiple states and associated attributes to the model
            
            Example
            -------
            states = [('healthy', {'u': 0.8, 'c':0}), 
                      ('dead',  {'u':0, 'c':0}),
                      ('sick', {'u':0.2, 'c':1000})]
            
            testmodel.add_many_states(states)
        """
        
        self.state.update(dict(state))
        self.states.append(list(self.state.keys()))
          
    def add_state(self, state, attr=None, order=None):
        """
        Adds a node with some optinal attributes (in a dictionary)
        
        state: string
            label of node
        
        attr: dict
            attributes of the node
            
        Example
        -------
        hipmodel.add_state('very sick', {'u':0.1, 'c':5000}) 
        """
        self.state.update({state: attr})
        self.states.append(state)
            
    def del_state(self, state):
        """
        Deletes a state
                        
        state: string
            label of node
        """
        #todo: update connections automatically after deleting a state?
        #in which case: default rule for distributing probability?
        
        for state in self.states:
            if st[0]==state:
                self.states.remove(st)
                break
            print('Error: State not found')
    
    def add_edge(self, from_state, to_state, attr):
        """
        Adds connections between states and give the connection attributes
        
        from_state: string
                    start state of the edge
        
        to_state: string
                    end state of the edge
        
        attr: dict
                    attributes of the edge
        
        Example
        -------
                
        hipmodel.add_edge('healthy', 'sick', {'p': 0.95})
            
        """
        self.edge[from_state] = {to_state:attr}
        self.pmatrix=self.np_matrix()               

    def add_many_edges(self, edges):
        """
        Adds edges between states and give the edges attributes
        
        
        Example
        -------
        
        transitions = [('healthy', 'healthy', {'p': 0.95}),
               ('healthy', 'sick', {'p':0.04}),
               ('healthy', 'dead', {'p': 0.01}),
               ('sick', 'sick', {'p':0.5}),
               ('sick', 'dead', {'p':0.3}),
               ('sick', 'healthy', {'p':0.2}),
               ('dead', 'dead', {'p':1})]
   
        hipmodel.add_edge(transitions)
            
        """
        
        states = set([v[0] for v in edges])
        for from_state in states:
            self.edge[from_state] = {v[1]:v[2] for v in edges if from_state==v[0]}
        self.pmatrix=self.np_matrix()
    
    def np_matrix(self):
        matrix=[]
        states = self.state.keys()
        for from_state in states:
            plist=[]
            for to_state in states:
                print(from_state, to_state)
                try: 
                    p = self.edge[from_state][to_state]['p']
                except:
                    p = 0
                plist.append(p)
            matrix.append(plist)
        return matrix
    

    def expected(self, var):
        g=self.to_nx()
        end_nodes = [x for x in g.nodes_iter() if g.out_degree(x)==0 and g.in_degree(x)==1]
        start_node = [x for x in g.nodes_iter() if g.out_degree(x)>0 and g.in_degree(x)==0]
        start_node = start_node[0]
                
        path_to = {}
        for node in end_nodes:
            path_to[node] = list(nx.all_simple_paths(
                    g, source=start_node, target=node))[0]
           
        pr = {}
        values = {}
        for end_node, path in path_to.items():
            pr[end_node] = 1
            values[end_node] = 0
            n_nodes=len(path)
            
            for i, node in enumerate(path):
                if i<n_nodes-1:
                    p = self.edge[node][path[i+1]]['p']
                    if isinstance(p, str):
                        p=eval(p)
                    value = self.edge[node][path[i+1]][var]
                    if isinstance(value, str):
                        value=eval(value)
                    pr[end_node] = pr[end_node] * p
                    values[end_node] = values[end_node] + value
                                    
        expected_value={}
        for end_node, path in path_to.items():
            expected_value[end_node] = pr[end_node] * values[end_node]
            
        return sum(expected_value.values())
                         
    def simulate(self, var, n=1000, plot=True):
        sim_results = [self.expected(var) for x in range(n)]
        if plot:
            sim = pd.Series(sim_results) 
            sim.plot.hist()
        return sim_results
    
    def explore_parameters(self, param, plot=True):
        
        
        sim_results = [self.expected(var) for x in range(n)]
        if plot:
            sim = pd.Series(sim_results) 
            sim.plot.hist()
        return sim_results
    
    def psa(self, xvar, yvar, n=1000, plot=True):
        df=pd.DataFrame()
        df['xvar']=self.simulate(xvar, n=n)
        df['yvar']=self.simulate(yvar, n=n)
        df.plot.scatter(x='xvar', y='yvar')
        return df
    
    def to_nx(self):
        nodes = self._to_nx_state()
        edges = self._to_nx_edge()
        g=nx.DiGraph()
        g.add_nodes_from(nodes)
        g.add_edges_from(edges)
        return g
    
    def to_qe(self):
        pass
    
    def to_json(self):
        pass
    
    def save(self, format='pickle'):
        pass
#    
##%% mobile xray
##adding states with attributes
#g=nx.DiGraph()
#
## tree structure (with costs and probabilities)
#tree = [('patient', 'xray', {'p': 1, 'c':1650}),
#               ('xray', 'need', {'p':0.8, 'c':4000}),
#               ('xray', 'no_need', {'p': 0.2, 'c':0}),
#               ('need', 'inpatient', {'p':0.032, 'c':"dist('normal', 5000, 50)"}),
#               ('need', 'outpatient', {'p':0.368, 'c':5000}),
#               ('need', 'home', {'p':0.6, 'c':0})]
#
#
#g.add_edges_from(tree)
#g.nodes()
#
#
##%%
#
#xray = DecisionTree()
#xray.import_from_nx(g)
#
#xray.state
#xray.edge
#
#xray.to_nx()
#
#xray.expected('c')
#xray.simulate('c')
#
#xray.psa(xvar='u', yvar='c')






#%% liver diagnostic

states = [('susceptible',           {'u':lookup_csv(dfs, valuevar='p_die_stI', attr=['Age']), 'c':100}),
          ('low_fibrosis_index',    {'u':0.5, 'c':100}),
          ('high_fibrosis_index',   {'u':0.5, 'c':100}),          
          ('advanced_fibrosis',     {'u':0.4, 'c':1000}),
          ('no_advanced_fibrosis',  {'u':0.4, 'c':1000}),
          ('low_elf',               {'u':0.4, 'c':1000}),
          ('high_elf',              {'u':0.4, 'c':1000}),
          ('low_ela',               {'u':0.4, 'c':1000}),
          ('high_ela',              {'u':0.4, 'c':1000}),
          ('quit_drink',            {'u':0.4, 'c':1000}),
          ('dead',                  {'u':0.4, 'c':1000})]      

#adding edges between states (with probabilities)
transitions = [
    ('susceptible', 'low_fibrosis_index', {'p': 58/128}),
    ('susceptible', 'high_fibrosis_index', {'p': 70/128}),
    
    ('low_fibrosis_index', 'susceptible', {'p': 50/58}),
    ('low_fibrosis_index', 'quit_drink', {'p': 8/58}),
         
    ('high_fibrosis_index', 'low_elf', {'p': 60/70}),
    ('high_fibrosis_index', 'high_elf', {'p': 10/70}),
    
    ('low_elf', 'susceptible', {'p': 45/60}),
    ('low_elf', 'quit_drink', {'p': 15/60}),
    
    ('high_elf', 'low_ela', {'p': 0/10}),
    ('high_elf', 'high_ela', {'p': 10/10}),
    
    ('low_ela', 'no_advanced_fibrosis', {'p': 0/10}),
    ('low_ela', 'advanced_fibrosis', {'p': 10/10}),
    
    ('high_ela', 'no_advanced_fibrosis', {'p': 7/59}),
    ('high_ela', 'advanced_fibrosis', {'p': 52/59})]

