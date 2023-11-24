import math

import dgl
import torch.nn
import torch.nn as nn
import torch.nn.functional as F
import dgl.function as fn
from dgl.nn.functional import edge_softmax



# GCN model / GAT Model

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim,
                 bias =True, act = F.relu):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        self.act = act

        self.linear1 = nn.Linear(input_dim,hidden_dim,bias)
        self.linear2 = nn.Linear(hidden_dim,output_dim,bias)

    def forward(self, h):
        h = self.linear1(h)
        h = self.act(h)
        h = self.linear2(h)
        return h


class GraphConv(nn.Module):
    def __init__(self, hidden_dim, act= F.relu, dropout_rate = 0.2):
        super().__init__()

        self.act = act
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = dropout_rate
        self.linear = nn.Linear(hidden_dim, hidden_dim, bias=False)

    def forward(self, graph, training=False):
        h0 = graph.ndata['h']
        graph.update_all(fn.copy_u('h','m'), fn.sum('m','u_')) # fn.sum() <- pooling
        h = self.act(self.linear(graph.ndata['u_'])) + h0
        h = self.norm(h)

        h = F.dropout(h, p=self.dropout, training=training)

        graph.ndata['h'] = h
        return graph

class GraphAttention(nn.Module):
    def __init__(self,hidden_dim,num_heads=4,bias_mlp=True,
                 dropout_rate = 0.2, act=F.relu):
        super().__init__()

        self.mlp = MLP(
            input_dim= hidden_dim,
            hidden_dim=2*hidden_dim,
            output_dim=hidden_dim,
            bias= bias_mlp,
            act= act
        )
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.splitted_dim = hidden_dim // num_heads

        self.dropout = dropout_rate

        self.w1 = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.w3 = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.w4 = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.w5 = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.w6 = nn.Linear(hidden_dim, hidden_dim, bias=False)

        self.act = act
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self,graph,training=False):

        h0 = graph.ndata['h']
        e_ij = graph.edata['e_ij']

        graph.ndata['u'] = self.w1(h0).view(-1, self.num_heads, self.splitted_dim)
        graph.ndata['v'] = self.w2(h0).view(-1, self.num_heads, self.splitted_dim)
        graph.edata['x_ij'] = self.w3(h0).view(-1, self.num_heads, self.splitted_dim)

        graph.apply_edges(fn.v_add_e('v','x_ij','m'))
        graph.apply_edges(fn.u_mul_e('u','m','attn'))
        graph.edata['attn'] = edge_softmax(graph, graph.edata['attn'] / math.sqrt(self.splitted_dim))

        graph.ndata['k'] = self.w4(h0).view(-1, self.num_heads, self.splitted_dim)
        graph.edata['x_ij'] = self.w5(h0).view(-1, self.num_heads, self.splitted_dim)
        graph.apply_edges(fn.v_add_e('k','x_ij','m'))

        graph.edata['m'] = graph.edata['attn'] * graph.edata['m']
        graph.update_all(fn.copy_e('m','m'), fn.sum('m','h'))

        h = self.w6(h0) + graph.ndata['h'].view(-1, self.num_heads, self.splitted_dim)
        h = self.norm(h)

        h = h + self.mlp(h)
        h = self.norm(h)

        h = F.dropout(h, p=self.dropout, training=training)

        graph.ndata['h'] = h
        return graph

# Model

class GCN_GAT_Model(nn.Module):
    def __init__(self, model_type,
                 num_layers=4, hidden_dim = 64,
                 num_heads =4, dropout_rate = 0.2,
                 bias_mlp=True, readout='sum',
                 act=F.relu, initial_node_dim = 59,
                 initial_edge_dim=6, is_classification=False):
        super().__init__()

        self.num_layers = num_layers
        self.embedding_node = nn.Linear(initial_node_dim,hidden_dim, bias=False)
        self.embedding_edge = nn.Linear(initial_edge_dim,hidden_dim, bias=False)
        self.readout = readout

        self.mp_layers = torch.nn.ModuleList()
        for _ in range(self.num_layers):
            mp_layer = None
            if model_type == 'gcn':
                mp_layer = GraphConv(
                    hidden_dim=hidden_dim,
                    dropout_rate=dropout_rate,
                    act=act
                )
            elif model_type == 'gat':
                mp_layer = GraphAttention(
                    hidden_dim=hidden_dim,
                    num_heads=num_heads,
                    bias_mlp=bias_mlp,
                    dropout_rate=dropout_rate,
                    act=act
                )
            else:
                raise  ValueError('Invalid model type , only [gcn, gat]')

            self.mp_layers.append(mp_layer)

        self.linear_out = nn.Linear(hidden_dim, 1, bias=False)

        self.is_clasification = is_classification
        if self.is_clasification:
            self.sigmoid = F.sigmoid


    def forward(self, graph, training=False):
        h = self.embedding_node(graph.ndata['h'].float())
        e_ij = self.embedding_edge(graph.edata['e_ij'].float())
        graph.ndata['h'] = h
        graph.edata['e_ij'] = e_ij

        # updata GCN or GAT
        for i in range(self.num_layers):
            graph = self.mp_layers[i](
                graph=graph,
                training=training
            )

        out = dgl.readout_nodes(graph, 'h', op=self.readout)
        out = self.linear_out(out)

        if self.is_clasification:
            out = self.sigmoid(out)

        return out