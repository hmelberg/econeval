# -*- coding: utf-8 -*-
"""
Created on Wed Aug 16 00:41:12 2017

@author: hmelberg_adm
"""

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
