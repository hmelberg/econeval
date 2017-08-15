

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
#import quantecon as qe
import random

from mesa import Agent, Model
from mesa.time import RandomActivation, BaseScheduler
from mesa.datacollection import DataCollector

from nxpd import draw
from ipywidgets import interact, interactive, widgets, fixed, interact_manual, VBox, HBox
from IPython.display import display

from collections import defaultdict
import time
from collections import Counter, defaultdict

#%matplotlib inline

pd.__version__
#%% structure
#model needs:
#- states (with attributes)
#- transition matrix (with attributes)
#- model attributes
#    - including start values and updaters (if relevant)
#- agent attributes
#    - including start values and updaters (if relevant)
#- recorders
#- data
#- parameters


#%% helper functions for agent simulation
def update_history(self):
 
    if self.new_state == self.state:
        self.time_in +=1
        self.sum_time[self.state] +=1
    else:
        self.previous_state = self.state
        #self.state = new_state
        self.time_in = 0 
        self.history.append({self.new_state: self.attr['age']})      
        self.model.history[self.unique_id].append((self.attr['age'],
                                                  self.new_state,
                                                  self.attr['utility_sum'],
                                                  self.attr['cost_sum']))
        self.model.sum_agents[self.new_state] += 1
        self.model.sum_agents[self.previous_state] -= 1


#%%
class SimModel(Model):
    """A model with N agents."""

    def __init__(self, N, model):
        self.num_agents = N
        self.schedule = BaseScheduler(self)
        self.model_step = 0
        self.attr = {}
               
        # transfer properties from graph structure to model
        #idea: no need for a graph at all, just a class of info!
        #make graph optional, and add a module to extract info from graph
        #in that case, perhaps delete all copying below and just use the object
        
        self.info = model.info 
        self.data = model.data
        self.recorder = model.recorder
        self.states = model.state.keys()
        self.state = model.state
        self.edge = model.edge
        self.attr = model.model_attr
        self.agent_attr = model.agent_attr
                
        # state probabilities
        self.next_state = {}
        self.pr_next_state = defaultdict(list)
        
        for from_state in self.states:
            self.next_state[from_state] = list(model.edge[from_state].keys())
            for to_state in self.next_state[from_state]:
                p = model.edge[from_state][to_state]['p']
                self.pr_next_state[from_state].append(p)
        
               
        # model attributes
        for attr, value in self.attr.items():
            if callable(value): 
                self.attr[attr] = value()
            elif isinstance(value, numbers.Number):
                self.attr[attr] = value
            else:
                print("Error: The attribute must be assigned a function or a fixed number")
        
            
        # collect node attributes from graph
        #node_attr = {attr for state in g.nodes() for attr in g.node[state].keys()}
        self.state_attr = defaultdict()
        self.state_attr = {state : model.state[state] for state in model.state.keys()}
        
        # state attributes that are functions get an explicit value based on the function here
        for state in model.state.keys():
            for attr in self.state_attr[state].keys(): #what if a state has no attributes, errro here?
                if 'dist' in str(self.state_attr[state][attr]):
                    self.state_attr[state][attr]=eval(self.state_attr[state][attr])
                    
    
        
    
        #default collect and counters
        self.history = {}
        self.sum_agents = {}

        for state in self.states:
            self.sum_agents[state] = 0
         
        #collects attributes to be updated in a list (updates only apply to agents who are alive and )
        self.agent_updates = {k : v['update'] for k, v in self.agent_attr.items() if (self.agent_attr[k]['update']!=False) & (self.agent_attr[k]['condition']=='alive')}
          
        # Create agents
        for i in range(self.num_agents):
            a = SimAgent(i, self)
            self.schedule.add(a)
        
        ##initalize datacollection
              
        #create list of tables used to record values
        tables = {self.recorder[k]['table']: self.recorder[k]['collect_vars'] for k in self.recorder} 
        
        collect = {state: eval("lambda m: m.sum_agents['{state}']".format(state=state)) for state in self.states}

#        self.datacollector = DataCollector(model_reporters=collect, 
#                                           tables=tables)
        
        self.datacollector = DataCollector(model_reporters=collect, 
                                           agent_reporters={"history": lambda a: a.history}, 
                                           tables=tables)
    
    
    def step(self):
        self.datacollector.collect(self)
        
        # check for model level triggers
        for name in self.recorder.keys():
            if self.recorder[name]['level']=='model':
                if self.recorder[name]['trigger'](self):
                    values = {var: self.attr[var] for var in self.recorder[name]['collect_vars']}
                    self.datacollector.add_table_row(self.recorder[name], row=values)
        
        self.model_step += 1
        self.schedule.step()

    def run_model(self, n):
        for i in range(n):
            self.step()
#%%
df = pd.read_csv('C:/Users/hmelberg/Google Drive/galaxy/p_die_bc.csv')

def lookup(table, valuevar, query = 'age = 42 & gender = male'):
    return df.query(query).loc[:,valuevar].values[0]


lookup(df, valuevar='p_die_stI', query="Age==25")

def lookup_index(table, use_attr):
    pass

 #menoization? use decorator, caching?     
def lookup_csv(table, 
               value_var,
               use_attr=['age', 'gender'], 
               default=0.1, 
               warning=True, 
               table_attr_scheme={'age':'age'}):
        try:
            for attr in use_attr:
                v = self.attr[attr]
                condition = f"{attr} = {v}"
                conditions.append(condition)
            query = " & ".join(conditions)             
            #query = " & ".join([f'attr[{attr}] = {self.attr[{attr}]}' for attr in use_attr])
            return self.model.data[table].query(query).loc[:,valuevar].values[0]
        except:
            return default
        
def lookup_dict(table, use_attr=['age', 'gender'], default=0.1, warning=True):   
     try:
         find = (self.attr[attr] for attr in use_attr)
         return self.model.data[table][find]
     except:
         return default

def lookup_df(name, use_attr):
    lookupdict = {attr: self.attr[attr] for attr in use_attr}
    select = pd.Series(list(lookupdict))
    return name[select].value()

def load_lookup_table(file, filetype='csv', **kwargs):
    if filetype=='csv':
        df = pd.read_csv(file, kwargs)
    else:
        print('Error: {filetype} is an unknown filetype')
    return df
#%%
df = pd.read_clipboard(decimal=',')
#%%
# please forgive me, God, for this code
def lookup_factory(df, 
                   function, 
                   expression, 
                   default, 
                   warning = False, 
                   filetype='csv',
                   **kwargs):
    
    template="""def {function}(self):
{conditions} 
    else: 
        return {default}

        """
    expression = "elif agent.age > {from_age} & agent.age > {to_age} & agent.gender == {gender} : return {mortality}"
    expression = "    " + expression
    agent_vars = expression.split('agent.')[1:]
    agent_vars = {word.split(' ',1)[0] for word in agent_vars}
    
    convert_from = [f"agent.{var}" for var in agent_vars]
    convert_to = [f"self.attr['{var}']" for var in agent_vars]
    
    for n, var in enumerate(convert_from):
        expression = expression.replace(var, convert_to[n])
        #print(var, convert_to[n], expression)
    
    conditions=[]    
    for row in range(len(df)):
        insert_values = df.iloc[row, :].to_dict()
        for k, v in insert_values.items():
            if isinstance(v, str):
                v=f"'{v}'"
                insert_values[k]=v
                
        conditions.append(expression.format(**insert_values))
        
    conditions = "\n".join(conditions)
    conditions = conditions.replace('    elif', '    if', 1)
    template = template.format(function=function,
                    conditions=conditions,
                    default=default)
    return template


a = lookup_factory(df,function='pr_backgroundmortality', default=0.1, expression=expression)

print(a)
#%%

class SimAgent(Agent):
    """ An agent with some attributes and counters"""
    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)

        self.state = 'healthy'
        self.new_state = None
              
        self.attr = {}
        
        for att, value in self.model.agent_attr.items():
            if callable(value['start']):
                self.attr[att] = value['start']()
            else:
                self.attr[att] = value['start']
                          
        self.time_in=0
        self.sum_time = defaultdict()
        
        for state in self.model.states:
            self.sum_time[state] = 0

        self.history = [{0: self.state}]
                        
        self.model.history[self.unique_id] = [(self.attr['age'], self.state, 0, 0)]
                        
        self.previous_state = None
        
        self.model.sum_agents[self.state] +=1

    def step(self):
        if self.state=='dead':
            return
        
        # get list of possible next states and compute their probabilities
        next_states = self.model.next_state[self.state] 
        probs = self.model.pr_next_state[self.state]
        
        # som probs are numbers, some are functions that have to be called to get probs
        probs = [pr(self) if callable(pr) else pr for pr in probs]
        
        # make sure the probs sum to one
        # make pr of staying in same state 1 - other probs
        nextdict = dict(zip(next_states, probs))
        nextdict.pop(self.state)
        rest = 1 - sum(nextdict.values())
        nextdict[self.state]=rest
#        print(nextdict.keys())
#        print(nextdict.values())
        new_state = np.random.choice(list(nextdict.keys()), size=1, p=list(nextdict.values()))[0]
        
        self.new_state=new_state
        # update history
        
        update_history(self)
        
        # check for triggers for agent recorders
        for name in self.model.recorder.keys():
            if self.model.recorder[name]['leve']=='agent':
                if self.model.recorder[name]['trigger'](self):
                    values = {var: self.attr[var] for var in self.model.recorder[name]['collect_vars']}
                    self.model.datacollector.add_table_row(self.model.recorder[name], row=values)
        
        # update agent attributes
        for att, update in self.model.agent_updates.items():
            self.attr[att]=update(self, att)
                    
        #update agent aggregates (function of attributes etc)
        
        self.state = new_state



#%% helper functions

def dist(name, *args, **kwargs):
    """returns a random number 
    
    name: string
        name of distribution
    args: parameters of the distribution, position based
    kwargs:parameters of the distribution, keyword based
    
    dist('normal')
    """
    return eval(f'np.random.{name}(*{args}, **{kwargs})')

dist('normal', 5, 2)

def normal(dist, args):
    return np.random.normal(*args)

def add_one(self, attr, **kwargs):
    a = self.attr[attr] + 1
    return a

def add_state_value(self, attr, **kwargs):
    a = self.attr[attr] + self.model.state_attr[self.state]
    return a


#%% define a markov cohort model
g = nx.DiGraph()
           
# adding states with attributes
states = [  ('healthy', {'u' : 0.8, 
                         'c' : 0}),
            ('dead',    {'u' : 0, 
                         'c' : 0}),
            ('sick',    {'u' : 0.5, 
                         'c' : 1000})]
states
            
g.add_nodes_from(states)

#adding edges between states (with probabilities)
transitions = [('healthy', 'healthy', {'p': 0.95}),
               ('healthy', 'sick', {'p':0.04}),
               ('healthy', 'dead', {'p': 0.01}),
               ('sick', 'sick', {'p':0.5}),
               ('sick', 'dead', {'p':0.3}),
               ('sick', 'healthy', {'p':0.2}),
               ('dead', 'dead', {'p':1})]

g.add_edges_from(transitions)

#%% draw notebook
draw(g, show='ipynb', format = 'png')

#%% draw browser
draw(g, format = 'svg')

#%% gui to create states and edges (in jupyter notebook)

class graph_gui():
    def __init__(self, g=nx.DiGraph()):
        self.graph=g
        self.add_nodew=widgets.Text(placeholder='Type label of new state')
        self.add_node_buttonw=widgets.Button(description='Add state')
        self.del_node_buttonw=widgets.Button(description='Del state')
               
        self.node_box=VBox([self.add_nodew, 
                            HBox([self.add_node_buttonw, self.del_node_buttonw])
                            ])
        
        self.select_nodew=widgets.Dropdown(description='Select state')
        
        self.add_node_attr_labelw=widgets.Text(placeholder='Type label of new state attribute')
        self.add_node_attr_valuew=widgets.Text(placeholder='Type value of new state attribute')
        self.add_node_attr_buttonw=widgets.Button(description='Add attribute to state')
        self.del_node_attr_buttonw=widgets.Button(description='Del attribute from state')
        
        self.node_attr_box=VBox([self.select_nodew, 
                                 VBox([self.add_node_attr_labelw, self.add_node_attr_valuew]), 
                                 HBox([self.add_node_attr_buttonw, self.del_node_attr_buttonw])
                                 ])
        
        self.from_nodew = widgets.Dropdown(description='From state', options=[])
        self.to_nodew = widgets.Dropdown(description='To state', options=[])
        self.from_to_pw=widgets.Text(placeholder = 'Type probability', description = 'Probability', value='0')
        self.add_edge_buttonw=widgets.Button(description='Add edge')
        self.del_edge_buttonw=widgets.Button(description='Del edge')
        
        self.edge_box=VBox([VBox([self.from_nodew, self.to_nodew]),
                            self.from_to_pw,                                     
                            HBox([self.add_edge_buttonw, self.del_edge_buttonw])
                            ])
        
        self.add_edge_attr_labelw=widgets.Text(placeholder='Type label of new edge attribute')
        self.add_edge_attr_valuew=widgets.Text(placeholder='Type value of new edge attribute')
        self.add_edge_attr_buttonw=widgets.Button(description='Add attribute to edge')
        self.del_edge_attr_buttonw=widgets.Button(description='Del attribute from edge')
        
        self.edge_attr_box=VBox([VBox([self.add_edge_attr_labelw, self.add_edge_attr_valuew]), 
                                    HBox([self.add_edge_attr_buttonw, self.del_edge_attr_buttonw])])     
                                 
        self.graphw=widgets.Image()
                                 
        self.left_menu = VBox([self.node_box,
                               self.node_attr_box,
                               self.edge_box,
                               self.edge_attr_box])
    
        self.box=HBox([self.left_menu, self.graphw])
        
        display(self.box)
        
        self.add_nodew.on_submit(self.add_node)
        self.del_node_buttonw.on_click(self.del_node)
        self.add_edge_buttonw.on_click(self.add_edge)
        self.del_edge_buttonw.on_click(self.del_edge)
        
        self.add_node_attr_buttonw.on_click(self.add_node_attr)
        self.add_edge_attr_buttonw.on_click(self.add_edge_attr)
        
    def update_nodelist(self):
        a=self.graph.nodes()
        self.select_nodew.options = a
        self.from_nodew.options = a
        self.to_nodew.options = a 
        
    def add_node(self, text):
        self.graph.add_node(self.add_nodew.value)
        self.update_nodelist()
        self.select_nodew.value=self.add_nodew.value
        self.from_nodew.value = self.add_nodew.value
        self.to_nodew.value = self.add_nodew.value
        self.add_nodew.value = ""
        self.showg()  
    
    def del_node(self, text):
        if add_nodew.value in self.graph.nodes():
            self.graph.remove_node(self.add_nodew.value)
            update_nodelist(self)
            self.add_nodew.value = ""
            self.showg()
        else:
            pass
          
    def add_edge(self, text):
        self.graph.add_edge(self.from_nodew.value, self.to_nodew.value)
        self.showg()
    
    def del_edge(self, text):
        self.graph.remove_edge(self.from_nodew.value, self.to_nodew.value)
        self.showg()
    
    def add_node_attr(self, text):
        self.graph.node[self.select_nodew.value][self.add_node_attr_labelw.value] = self.add_node_attr_valuew.value
    
    def add_edge_attr(self, text):
        self.graph.edge[self.from_nodew.value][self.to_nodew.value][self.add_edge_attr_labelw.value] = self.add_edge_attr_valuew.value
                
    def showg(self):
        draw(self.graph, filename='testmarkovnx.png', show='ipynb', format='png')
        file = open("testmarkovnx.png", "rb")
        image = file.read()
        self.graphw.value=image

#%% show graph creation gui      
a = graph_gui()
a


#%% EconEval model definition class 
 
class EconEval(object):
    def __init__(self):
        
        self.states=[]
        self.transitions=None
        
        self.agent_attr={}
        self.model_attr={}
        self.data={}
        self.info={}
        self.recorder={}
        self.state=dict()
        self.edge={}
        self.pmatrix = None
        self.sim_results=None
        self.agents_in_state=None
    
    def add_data(self, name, data):
        self.data['name'] = data
    
    def add_data_csv(self, name, file, **kwargs):
        self.data['name'] = pd.read_csv(file, kwargs)
    
    def lookup_dict(self, name, labels):
        """
        lookup('mortality', ('male', 'above 65'))
        """
        return self.data[name][labels]
    
    @property
    def datalist():
        return self.data.keys()
    
    def import_from_nx(self, nxg):
        """
        Import econ eval model from a networkx graph structure
        
        nxg: network graph structure
            The nxg object must have a nodes and edges
            In addition the class will copy the following (if they exist):
                - agent_attr
                - model_attr
                - data
                - info
                - recorder
                - etc
                
        """
        # copy the nodes and edges from nx graph into EconEval class object
        # includes the attributes of nodes and edges
        self.add_many_states(nxg.nodes(data=True))
        self.add_many_edges(nxg.edges(data=True))   
        
        # copy other optional information from nx graph object into EconEval object
        
        if hasattr(nxg.graph, 'agent_attr'): 
            self.agent_attr=nxg.graph['agent_attr']          
        if hasattr(nxg.graph, 'model_attr'): 
            self.model_attr= nxg.graph['model_attr']
        if hasattr(nxg.graph, 'data'): 
            self.data = nxg.graph['data']
        if hasattr(nxg.graph, 'info'): 
            self.info = nxg.graph['info']
        if hasattr(nxg.graph, 'recorder'): 
            self.recorder = nxg.graph['recorder']
        if hasattr(nxg.graph, 'etc'): 
            self.etc = nxg.graph['etc']
            
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
        #todo: upfate connections automatically after deleting a state?
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
    
    
    def add_agent_attr(self, label, 
                       start=0, 
                       update= None, 
                       condition='alive'):
        """
        Adds attributes to the agents
        
        label : string
            name of the attribute
        
        start : object
            the start value of the attribute
            can be string, number, or a function
        
        update : function
            a function that specifies how the attribute should be updated
            default is None (the attribute is not updated)
                    
                
        Examples
        --------
        liver.add_agent_attr('age', update=lambda self, attr: self.attr['age'] +1 )
        liver.add_agent_attr('age', update=add_one)
        """
        
        self.agent_attr[label]={'start':start, 
                                     'update': update, 
                                     'condition':condition}
        
    def del_agent_attr(self, label):
        if label in self.agent_attr.keys():
            del(self.agent_attr[label])
        else:
            print(f'Attribute {label} does not exit')
    
    def del_many_agent_attr(self, labels):
        """Deletes a list of agent attributes"""
        for label in labels: 
            self.del_agent_attr(label)
    
    def add_recorder(self, label, trigger, collect=[], level='agent updates'):
        """
        Adds a recorder to the model
        
        label: string
            name of the recorder
        trigger: function
            specifies the conditions when the recording should happen
        collect: list of strings
            a list of the attributes to be collected when the recording is triggered
        level: string ('agent' or 'model', default is 'agent') 
            determines whether the trigger is checked each time an agent is updated or after each step in the model
            
        """    
        self.recorder[label]={'trigger': trigger, 'collect': collect}
      
    def add_many_recorders(self, recorderlist):
        """
        Adds a recorder to trace key information
        
        Example
        -------
        Record age and gender for agents when they die
        
        record['dead'] = {'table':'dead', 
                          'trigger':lambda self: self.new_state=='dead', 
                          'collect_vars':['age', 'gender']}
        """
        for label, recorder in recorders.items():
            self.add.recorder(label, **recorder)

    def del_recorder(self, label):
        if label in self.agent_recorder.keys():
            del(self.agent_recorder[label])
        else:
            print(f'Error: Recorder with name {label} does not exist')
    
    def del_many_recorders(self, labels):
        for label in labels:
            self.del_recorder(label)
                        
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
    
    def adjust_tmatrix(self, tmatrix):
        """
        Makes sure that the row of each matrix sum to one
        
        Makes the probability ofstaying in the same state  1- sum of 
        probability of all other transition probabilities 
        """
        for i, prs in enumerate(tmatrix):
            print(prs, prs[i])
            prs[i] = 1-(sum(prs) - prs[i])
            print(prs[i])
            tmatrix[i]=prs
        return tmatrix
                 
    def simulate_markov(self, intervention=None, individuals=1000, steps=100, init=None, plot=True):
        print(self.state.keys())
        mc = qe.MarkovChain(self.pmatrix, state_values=list(self.state.keys()))

        results = pd.DataFrame()
        results_sum = pd.DataFrame()
        
        for individual in range(individuals):
            results[individual]= mc.simulate(ts_length = steps, init = init)    
        self.sim_results=model.history
        
        count_agents_in_states = {period: Counter(self.sim_results.iloc[period,:]) for period in range(steps)}
        self.agents_in_states = pd.DataFrame(count_agents_in_states).T.fillna(0)
        
        if plot:
            return self.agents_in_states.plot()
        else:
            return results
        
    def simulate_agent(self, intervention=None, individuals=1000, steps=100, init=None, out='plot'):
        model = SimModel(1000, model=self)
        for i in range(100):
           model.step()
           print(model.sum_agents)
        self.sim_results = model.history
        self.data['state_history'] = model.history
        model.sum_agents
        
        modeldf = model.datacollector.get_model_vars_dataframe()
#        agentdf = model.datacollector.get_agent_vars_dataframe()
#        tabledf = model.datacollector.get_table_dataframe('dead')
#        tabledf20 = model.datacollector.get_table_dataframe('age20')
        
        
        if out == 'plot':
            modeldf.plot()
        if out == 'history':
            return self.sim_results
#        elif out == 'icer'   :
#             
#            u = [v[-1][-1] for v in disease.sim_results.values()]
#            c
#            return (u,c)
#        
#        for k, v in disease.sim_results.items():
            
        
        
        #a = pd.DataFrame.from_dict(disease.sim_results, orient='columns')
        #modeldf = model.datacollector.get_model_vars_dataframe()
        #agentdf = model.datacollector.get_agent_vars_dataframe()
#    def psa(self, nsim):
#        results={}
#        for x in range(nsim)
#            results[0] = simulate_agent(self, our='icer')
        
        
    def to_nx(self):
        pass
    
    def to_qe(self):
        pass
    
    def to_json(self):
        pass
    
    def save(self, format='pickle'):
        pass
    
    def psa(self, n=10):
        sim_results_df = pd.DataFrame()
        for i in range(n):
            sim_results = self.simulate_agent(out='history')
            c=[]
            for agent in sim_results.keys():
                for row in sim_results[agent]:
                    b = (agent, *row)
                    c.append(b)
                d = pd.DataFrame(c, columns=['pid', 'state', 'age', 'u', 'c'])
                ucsum = d.groupby('pid').last()[['u', 'c']].sum()
                usum = ucsum['u']
                csum = ucsum['c']
                umean= usum/d.pid.nunique()
                cmean= csum/d.pid.nunique()
            sim_results_df[i] = ucsum
        return sim_results_df

        
    

 
#%% disease model
g = nx.DiGraph()

states = [('healthy', {'u': 0.8, 'c':0}), 
          ('dead',  {'u':0, 'c':0}),
          ('sick', {'u':0.2, 'c':1000})]


g.add_states_from(states)
#%%

disease = EconEval()
disease.import_from_nx(g)

disease.state

disease.plot()





draw(g)



disease.add_state('healthy', {'u':0.9, 'c':0})

disease.add_many_states(states)


disease.add_state('dead', {'u':0, 'c':0})
disease.add_edge('heathy', 'dead', {'p':0.4})

disease.import_from_nx(g)
#disease.simulate_markov()

disease.add_agent_attr('age', update=add_one)

def p_death(agent):
    if agent.age<50:
        p=0.1
    else:
        p=0.2
    return p



disease.add_agent_attr('utility_sum', update=add_state_value)

disease.add_agent_attr('utility_sum', update=lambda self, attr: self.attr['utility_sum']+self.model.state_attr[self.state]['u'] )
disease.add_agent_attr('cost_sum', update=lambda self, attr: self.attr['cost_sum']+self.model.state_attr[self.state]['c'] )

disease.psa()
a
a.plot.scatter(x='u', y = 'c')
a = a.T
disease.state

disease.agent_attr

disease.simulate_markov()
disease.simulate_agent()


disease.sim_results.values()

# again, I ask forgiveness

c=[]
for agent in disease.sim_results.keys():
    for row in disease.sim_results[agent]:
        b = (agent, *row)
        c.append(b)
d = pd.DataFrame(c, columns=['pid', 'state', 'age', 'u', 'c'])
ucsum = d.groupby('pid').last()[['u', 'c']].sum()
usum = ucsum['u']
csum = ucsum['c']
umean= usum/d.pid.nunique()
cmean= csum/d.pid.nunique()

#%% agent attributes
agent_attr = defaultdict()

agent_attr['age'] = {'start':0, 'update': lambda self, attr: self.attr['age'] + 1, 'condition': 'alive'}

#alternative formulation: use a named function, not a lambda

agent_attr['age'] = {'start':0, 'update': add_one, 'condition': 'alive'}
agent_attr['cumsum_utility'] = {'start':0, 'update' : lambda self, attr: self.attr['cumsum_utility']+self.model.state_attr[self.state]['u'], 'condition':'alive'}
agent_attr['cumsum_cost'] = {'start':0, 'update' : lambda self, attr: self.attr['cumsum_cost']+self.model.state_attr[self.state]['c'], 'condition':'alive'}
agent_attr['state'] = {'start': 'healthy', 'update': False,  'condition':'alive'}
agent_attr['male'] = {'start':lambda: np.random.binomial(1, p=0.51), 'update': False,  'condition':'alive'}

g.graph['agent_attr'] = agent_attr

#%% model attr
model_attr = {}
g.graph['model_attr'] = model_attr

#%% recorder

record={}
record['dead'] = {'table':'dead', 'trigger':lambda self: self.new_state=='dead', 'collect_vars':['age', 'cumsum_utility', 'cumsum_cost']}
record['age20'] = {'table':'age20', 'trigger':lambda self: self.attr['age']==20, 'collect_vars':['age', 'male', 'cumsum_utility', 'cumsum_cost']}   
g.graph['record'] = record


#%% general model info
g.graph['info'] = {'time_scale': 365, 
                    'structure':'Markov cohort',
                    'autorecord': True}
data={}
g.graph['data']=data




states
transitions
agent_attr
record

#%% liver states and transitions
#states = [('noelf_noela',   {'u':0.6, 'c':100}), 
#          ('elf_noela',     {'u':0.5, 'c':100}),
#            ('noelf_ela',   {'u':0.4, 'c':1000})
#            ('elf_ela'),    {'u':0.1, 'c':5000}]

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




#for from_state in liver.state.keys():
#    for to_state in liver.state.keys():
#        print(f"('{from_state}', '{to_state}', {{'p': 0.10}}),")

#%% liver model explore

liver = EconEval()

liver.add_many_states(states)
liver.add_many_edges(transitions)

liver.add_agent_attr('age', lambda self, attr: self.attr['cumsum_utility']+self.model.state_attr[self.state]['u'] )

liver.states
liver.edge['susceptible']['high_fibrosis_index']['p']

liver.simulate_markov()

liver.plot()

#%% test model
# adding states with attributes
states = [  ('healthy', {'u' : 'dist("normal",5,1)', 
                         'c' : 0}),
            ('dead',    {'u' : 0, 
                         'c' : 0}),
            ('sick',    {'u' : 0.2, 
                         'c' : 1000})]
states
            
g.add_nodes_from(states)


testmodel=EconEval()
testmodel.import_from_nx(g)

testmodel.state

eval(testmodel.state['healthy']['u'])
testmodel.add_agent_attr('age', update=add_one)


testmodel.add_agent_attr('age', update=lambda self, attr : self.attr['age']+1)
testmodel.agent_attr

testmodel.simulate_agent()

#%% mobil røntgen


states = [  ('Mobil', 
                 {'u' : 0, 
                  'c' : 1560}),
            ('OK',    
                 {'u' : 0, 
                  'c' : 0}),
            ('Sykehus',    
                 {'u' : 0.4 , 
                  'c' : 15000}),
            ('Hjemmebehandling',    
                 {'u' : 0.4 , 
                 'c' : 1000})]
states
            
g.add_nodes_from(states)

transitions = [('Mobil', 'OK', {'p': 0.163}),
               ('Mobil', 'Sykehus', {'p':0.175}),
               ('Mobil', 'Hjemmebehandling', {'p': 0.715})]

g.add_edges_from(transitions)


testmodel=EconEval()
testmodel.import_from_nx(g)

testmodel.state

eval(testmodel.state['healthy']['u'])
testmodel.add_agent_attr('age', update=add_one)


testmodel.add_agent_attr('age', update=lambda self, attr : self.attr['age']+1)
testmodel.agent_attr

testmodel.simulate_agent()



#%% hip model explore

hipmodel=EconEval()  

hipmodel.add_state('very sick', {'u':0.1, 'c':5000})
hipmodel.add_state('very dead')
hipmodel.edge['dead']['dead']['p']

hipmodel.state['healthy']['u'] = 0.9
hipmodel.state['healthy']['d'] = 0.7

hipmodel.state['very dead']['f'] = 0.3


hipmodel.add_state([('very dead'), ('extremely dead')])

hipmodel.add_states(states)

hipmodel.add_edge(transitions)

hipmodel.edge
hipmodel.pmatrix
hipmodel.states
hipmodel.simulate(init='healthy')




#%%

a=dict_keys(['sick', 'dead', 'healthy'])
b=dict_values([0.04, 0.01, 0.95])
np.random.choice(a, size=1, p=b)[0]
#%%    
model.history

model.sum_agents

modeldf = model.datacollector.get_model_vars_dataframe()
agentdf = model.datacollector.get_agent_vars_dataframe()
tabledf = model.datacollector.get_table_dataframe('dead')
tabledf20 = model.datacollector.get_table_dataframe('age20')


modeldf.plot()
#%% get data
modeldf = model.datacollector.get_model_vars_dataframe()
agentdf = model.datacollector.get_agent_vars_dataframe()

modeldf.plot()
#%%

model.dead_record
df = pd.DataFrame(model.dead_record).T
df['u_c_ratio'] = df.sum_cost/df.sum_utility
df.u_c_ratio.plot.hist()

#%%


#%% try comparing two models
g2 = g.copy()
g2
g2['sick']['healthy']['p']
g2['sick']['healthy']['p']=0.2
g2['sick']['dead']['p']
g2['sick']['dead']['p']=0.1
# create a gui to set ps and also helper functions to helt it sum t one, and change time periods

model1 = TestModel(100, g=g)
model2 = TestModel(100, g=g2)
utility1= []
utility2 = []
cost1 = []
cost2 = []
diffcost=[]
diffutility=[]

for model_run in range(100):
    model1 = TestModel(100, g=g)
    model2 = TestModel(100, g=g2)
    for period in range(100):
        model1.step()
        model2.step()
    m1results = model1.datacollector.get_table_dataframe('dead')
    m2results = model2.datacollector.get_table_dataframe('dead')
    
    u1=m1results.cumsum_utility.mean()
    u2=m2results.cumsum_utility.mean()
    
    utility1.append(u1)
    utility2.append(u2)
    
    c1=m1results.cumsum_cost.mean()
    c2=m2results.cumsum_cost.mean()
    
    cost1.append(c1)
    cost2.append(c2)
    
    diffcost.append(c1-c2)
    diffutility.append(u1-u2)
fig, ax = plt.subplots()
ax.scatter(x=diffcost, y = diffutility, color='blue')
ax.scatter(x=cost1, y = utility1, color='blue')
ax.scatter(x=cost2, y = utility2, color='red')
ax.set_xlim(1400,1700)
ax.set_ylim(1250,1700)


model1 = TestModel(100, g=g)
model2 = TestModel(100, g=g2)
utility1= []
utility2 = []
cost1 = []
cost2 = []
for model_run in range(10):
    model1 = TestModel(100, g=g)
    model2 = TestModel(100, g=g2)
    for period in range(100):
        model1.step()
    m1results = model1.datacollector.get_table_dataframe('dead')
    utility1.append(m1results['cumsum_utility'].mean())
    cost1.append(m1results['cumsum_cost'].mean())

fig, ax = plt.subplots()
ax.scatter(x=cost1, y = utility1, color='blue')
ax.scatter(x=cost2, y = utility2, color='red')
ax.set_xlim(1400,1700)
ax.set_ylim(1250,1700)


    
    print(model2.sum_agents)

#%%
model2df = model2.datacollector.get_model_vars_dataframe()
agent2df = model2.datacollector.get_agent_vars_dataframe()

model2df.plot()

plt.scatter(x=modeldf.index, y = modeldf.dead.values)
plt.scatter(x=model2df.index, y = model2df.dead.values, color = 'red')
#%%
model.sum_agents
%matplotlib inline


agent_wealth = [a.wealth for a in model.schedule.agents]
plt.hist(agent_wealth)

modeldf = model.datacollector.get_model_vars_dataframe()

agentdf = model.datacollector.get_agent_vars_dataframe()


end_wealth = agent_wealth.xs(99, level="Step")["Wealth"]
end_wealth.hist(bins=range(agent_wealth.Wealth.max()+1))

model2.history
