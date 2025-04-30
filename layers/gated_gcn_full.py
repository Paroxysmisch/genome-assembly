import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import dgl.function as fn


class SymGAT(nn.Module):
    def __init__(self, in_channels, out_channels, batch_norm, dropout=None, residual=True):
        super().__init__()
        if dropout:
            self.dropout = dropout
        else:
            self.dropout = 0.0
        self.batch_norm = batch_norm
        self.residual = residual

        if in_channels != out_channels:
            self.residual = False

        self.fc = nn.Linear(in_channels, out_channels, bias=False)
        self.fc_e = nn.Linear(in_channels, out_channels, bias=False)
        self.attn_fc = nn.Linear(3 * out_channels, 1, bias=False)
        self.attn_fc_e = nn.Linear(3 * out_channels, 1, bias=False)

        dtype=torch.float32
        self.B_1 = nn.Linear(in_channels, out_channels, dtype=dtype)
        self.B_2 = nn.Linear(in_channels, out_channels, dtype=dtype)
        self.B_3 = nn.Linear(in_channels, out_channels, dtype=dtype)

        self.mix_node_edge_info = nn.Linear(2 * out_channels, out_channels)

        self.reset_parameters()

        if batch_norm: # batch normalization
            # self.bn_h = nn.BatchNorm1d(out_channels, track_running_stats=True)
            # self.bn_e = nn.BatchNorm1d(out_channels, track_running_stats=True)
            self.bn_h = nn.BatchNorm1d(out_channels, track_running_stats=True)
            self.bn_e = nn.BatchNorm1d(out_channels, track_running_stats=True)
        else: # layer normalization
            self.bn_h = nn.LayerNorm(out_channels)
            self.bn_e = nn.LayerNorm(out_channels)

    def reset_parameters(self):
        """Reinitialize learnable parameters."""
        gain = nn.init.calculate_gain("relu")
        nn.init.xavier_normal_(self.fc.weight, gain=gain)
        nn.init.xavier_normal_(self.attn_fc.weight, gain=gain)

    def edge_attention(self, edges):
        z2 = torch.cat([edges.src["z"], edges.data["z_e"], edges.dst["z"]], dim=1)
        a = self.attn_fc(z2)
        a_e = self.attn_fc_e(z2)
        return {"e": F.leaky_relu(a), "e_e": F.leaky_relu(a_e)}

    def message_func(self, edges):
        return {"z": edges.src["z"], "e": edges.data["e"], "z_e": edges.data["z_e"], "e_e": edges.data["e_e"]}

    def reduce_func(self, nodes):
        alpha = F.softmax(nodes.mailbox["e"], dim=1)
        alpha_e = F.softmax(nodes.mailbox["e_e"], dim=1)
        h = torch.sum(alpha * nodes.mailbox["z"], dim=1)
        h_e = torch.sum(alpha_e * nodes.mailbox["z_e"], dim=1)
        h = self.mix_node_edge_info(torch.cat([h, h_e], dim=-1))
        return {"h": h}

    def forward(self, g, h, e):
        with g.local_scope():
            h_in = h.clone()
            e_in = e.clone()

            g.ndata['B1h'] = self.B_1(h)
            g.ndata['B2h'] = self.B_2(h)
            g.edata['B3e'] = self.B_3(e)

            g_reverse = dgl.reverse(g, copy_ndata=True, copy_edata=True)

            g.apply_edges(fn.u_add_v('B1h', 'B2h', 'B12h'))
            e_ji = g.edata['B12h'] + g.edata['B3e']
            e_ji = F.relu(e_ji)
            if self.residual:
                e_ji = e_ji + e_in

            g_reverse.apply_edges(fn.u_add_v('B2h', 'B1h', 'B21h'))
            e_ik = g_reverse.edata['B21h'] + g_reverse.edata['B3e']
            e_ik = F.relu(e_ik)
            if self.residual:
                e_ik = e_ik + e_in

            z = self.fc(h)
            z_e = self.fc_e(e_ji)
            g.ndata["z"] = z
            g.edata["z_e"] = z_e
            g.apply_edges(self.edge_attention)
            g.update_all(self.message_func, self.reduce_func)
            g_reverse.ndata["z"] = z
            g_reverse.edata["z_e"] = z_e
            g_reverse.apply_edges(self.edge_attention)
            g_reverse.update_all(self.message_func, self.reduce_func)

            h = g.ndata.pop("h") + g_reverse.ndata.pop("h")
            h = self.bn_h(h)
            h = F.relu(h)
            if self.residual:
                h = h + h_in
            h = F.dropout(h, self.dropout, training=self.training)

            return h, e_ji


class GAT(nn.Module):
    def __init__(self, in_channels, out_channels, batch_norm, dropout=None, residual=True):
        super().__init__()
        if dropout:
            self.dropout = dropout
        else:
            self.dropout = 0.0
        self.batch_norm = batch_norm
        self.residual = residual

        if in_channels != out_channels:
            self.residual = False

        self.fc = nn.Linear(in_channels, out_channels, bias=False)
        self.fc_e = nn.Linear(in_channels, out_channels, bias=False)
        self.attn_fc = nn.Linear(3 * out_channels, 1, bias=False)
        self.attn_fc_e = nn.Linear(3 * out_channels, 1, bias=False)

        dtype=torch.float32
        self.B_1 = nn.Linear(in_channels, out_channels, dtype=dtype)
        self.B_2 = nn.Linear(in_channels, out_channels, dtype=dtype)
        self.B_3 = nn.Linear(in_channels, out_channels, dtype=dtype)

        self.mix_node_edge_info = nn.Linear(2 * out_channels, out_channels)

        self.reset_parameters()

        if batch_norm: # batch normalization
            # self.bn_h = nn.BatchNorm1d(out_channels, track_running_stats=True)
            # self.bn_e = nn.BatchNorm1d(out_channels, track_running_stats=True)
            self.bn_h = nn.BatchNorm1d(out_channels, track_running_stats=True)
            self.bn_e = nn.BatchNorm1d(out_channels, track_running_stats=True)
        else: # layer normalization
            self.bn_h = nn.LayerNorm(out_channels)
            self.bn_e = nn.LayerNorm(out_channels)

    def reset_parameters(self):
        """Reinitialize learnable parameters."""
        gain = nn.init.calculate_gain("relu")
        nn.init.xavier_normal_(self.fc.weight, gain=gain)
        nn.init.xavier_normal_(self.attn_fc.weight, gain=gain)

    def edge_attention(self, edges):
        z2 = torch.cat([edges.src["z"], edges.data["z_e"], edges.dst["z"]], dim=1)
        a = self.attn_fc(z2)
        a_e = self.attn_fc_e(z2)
        return {"e": F.leaky_relu(a), "e_e": F.leaky_relu(a_e)}

    def message_func(self, edges):
        return {"z": edges.src["z"], "e": edges.data["e"], "z_e": edges.data["z_e"], "e_e": edges.data["e_e"]}

    def reduce_func(self, nodes):
        alpha = F.softmax(nodes.mailbox["e"], dim=1)
        alpha_e = F.softmax(nodes.mailbox["e_e"], dim=1)
        h = torch.sum(alpha * nodes.mailbox["z"], dim=1)
        h_e = torch.sum(alpha_e * nodes.mailbox["z_e"], dim=1)
        h = self.mix_node_edge_info(torch.cat([h, h_e], dim=-1))
        return {"h": h}

    def forward(self, g, h, e):
        with g.local_scope():
            h_in = h.clone()
            e_in = e.clone()

            g.ndata['B1h'] = self.B_1(h)
            g.ndata['B2h'] = self.B_2(h)
            g.edata['B3e'] = self.B_3(e)

            g.apply_edges(fn.u_add_v('B1h', 'B2h', 'B12h'))
            e_ji = g.edata['B12h'] + g.edata['B3e']
            e_ji = F.relu(e_ji)
            if self.residual:
                e_ji = e_ji + e_in

            z = self.fc(h)
            z_e = self.fc_e(e_ji)
            g.ndata["z"] = z
            g.edata["z_e"] = z_e
            g.apply_edges(self.edge_attention)
            g.update_all(self.message_func, self.reduce_func)

            h = g.ndata.pop("h")
            h = self.bn_h(h)
            h = F.relu(h)
            if self.residual:
                h = h + h_in
            h = F.dropout(h, self.dropout, training=self.training)

            return h, e_ji

class SymGatedGCN(nn.Module):
    """
    Symmetric GatedGCN, based on the idea of  GatedGCN from 'Residual Gated Graph ConvNets'
    paper by Xavier Bresson and Thomas Laurent, ICLR 2018.
    https://arxiv.org/pdf/1711.07553v2.pdf
    """
    def __init__(self, in_channels, out_channels, batch_norm, dropout=None, residual=True):
        super().__init__()
        if dropout:
            # print(f'Using dropout: {dropout}')
            self.dropout = dropout
        else:
            # print(f'Using dropout: 0.00')
            self.dropout = 0.0
        self.batch_norm = batch_norm
        self.residual = residual

        if in_channels != out_channels:
            self.residual = False

        dtype=torch.float32

        self.A_1 = nn.Linear(in_channels, out_channels, dtype=dtype)
        self.A_2 = nn.Linear(in_channels, out_channels, dtype=dtype)
        self.A_3 = nn.Linear(in_channels, out_channels, dtype=dtype)
        
        self.B_1 = nn.Linear(in_channels, out_channels, dtype=dtype)
        self.B_2 = nn.Linear(in_channels, out_channels, dtype=dtype)
        self.B_3 = nn.Linear(in_channels, out_channels, dtype=dtype)

        if batch_norm: # batch normalization
            # self.bn_h = nn.BatchNorm1d(out_channels, track_running_stats=True)
            # self.bn_e = nn.BatchNorm1d(out_channels, track_running_stats=True)
            self.bn_h = nn.BatchNorm1d(out_channels, track_running_stats=True)
            self.bn_e = nn.BatchNorm1d(out_channels, track_running_stats=True)
        else: # layer normalization
            self.bn_h = nn.LayerNorm(out_channels) 
            self.bn_e = nn.LayerNorm(out_channels) 

    def message_forward(self, edges):
        """Message function used on the original graph."""
        A2h_j = edges.src['A2h']
        e_ji = edges.src['B1h'] + edges.dst['B2h'] + edges.data['B3e']  # e_ji = B_1*h_j + B_2*h_i + B_3*e_ji
        if self.batch_norm:
            e_ji = self.bn_e(e_ji)
        e_ji = F.relu(e_ji)
        if self.residual:
            e_ji = e_ji + edges.data['e']
        return {'A2h_j': A2h_j, 'e_ji': e_ji}

    def reduce_forward(self, nodes):
        """Reduce function used on the original graph."""
        A2h_j = nodes.mailbox['A2h_j']
        e_ji = nodes.mailbox['e_ji']
        sigma_ji = torch.sigmoid(e_ji)
        h_forward = torch.sum(sigma_ji * A2h_j, dim=1) / (torch.sum(sigma_ji, dim=1) + 1e-6)
        return {'h_forward': h_forward}

    def message_backward(self, edges):
        """Message function used on the reverse graph."""
        A3h_k = edges.src['A3h']
        e_ik = edges.dst['B1h'] + edges.src['B2h'] + edges.data['B3e']  # e_ik = B_1*h_i + B_2*h_k + B_3*e_ik
        if self.batch_norm:
            e_ik = self.bn_e(e_ik)
        e_ik = F.relu(e_ik)
        if self.residual:
            e_ik = e_ik + edges.data['e']
        return {'A3h_k': A3h_k, 'e_ik': e_ik}

    def reduce_backward(self, nodes):
        """Reduce function used on the reverse graph."""
        A3h_k = nodes.mailbox['A3h_k']
        e_ik = nodes.mailbox['e_ik']
        sigma_ik = torch.sigmoid(e_ik)
        h_backward = torch.sum(sigma_ik * A3h_k, dim=1) / (torch.sum(sigma_ik, dim=1) + 1e-6)
        return {'h_backward': h_backward}

    def forward(self, g, h, e):
        """Return updated node representations."""
        with g.local_scope():
            h_in = h.clone()
            e_in = e.clone()

            g.ndata['h'] = h
            g.edata['e'] = e

            g.ndata['A1h'] = self.A_1(h)
            g.ndata['A2h'] = self.A_2(h)
            g.ndata['A3h'] = self.A_3(h)

            g.ndata['B1h'] = self.B_1(h)
            g.ndata['B2h'] = self.B_2(h)
            g.edata['B3e'] = self.B_3(e)

            g_reverse = dgl.reverse(g, copy_ndata=True, copy_edata=True)

            # Reference: https://github.com/graphdeeplearning/benchmarking-gnns/blob/master-dgl-0.6/layers/gated_gcn_layer.py

            # Forward-message passing
            g.apply_edges(fn.u_add_v('B1h', 'B2h', 'B12h'))
            e_ji = g.edata['B12h'] + g.edata['B3e']
            # e_ji = self.bn_e(e_ji)
            e_ji = F.relu(e_ji)
            if self.residual:
                e_ji = e_ji + e_in
            g.edata['e_ji'] = e_ji
            g.edata['sigma_f'] = torch.sigmoid(g.edata['e_ji'])
            g.update_all(fn.u_mul_e('A2h', 'sigma_f', 'm_f'), fn.sum('m_f', 'sum_sigma_h_f'))
            g.update_all(fn.copy_e('sigma_f', 'm_f'), fn.sum('m_f', 'sum_sigma_f'))
            g.ndata['h_forward'] = g.ndata['sum_sigma_h_f'] / (g.ndata['sum_sigma_f'] + 1e-6)

            # Backward-message passing
            g_reverse.apply_edges(fn.u_add_v('B2h', 'B1h', 'B21h'))
            e_ik = g_reverse.edata['B21h'] + g_reverse.edata['B3e']
            # e_ik = self.bn_e(e_ik)
            e_ik = F.relu(e_ik)
            if self.residual:
                e_ik = e_ik + e_in
            g_reverse.edata['e_ik'] = e_ik
            g_reverse.edata['sigma_b'] = torch.sigmoid(g_reverse.edata['e_ik'])
            g_reverse.update_all(fn.u_mul_e('A3h', 'sigma_b', 'm_b'), fn.sum('m_b', 'sum_sigma_h_b'))
            g_reverse.update_all(fn.copy_e('sigma_b', 'm_b'), fn.sum('m_b', 'sum_sigma_b'))
            g_reverse.ndata['h_backward'] = g_reverse.ndata['sum_sigma_h_b'] / (g_reverse.ndata['sum_sigma_b'] + 1e-6)

            h = g.ndata['A1h'] + g.ndata['h_forward'] + g_reverse.ndata['h_backward']

            h = self.bn_h(h)

            h = F.relu(h)

            if self.residual:
                h = h + h_in

            h = F.dropout(h, self.dropout, training=self.training)
            e = g.edata['e_ji']

            return h, e


class GatedGCN(nn.Module):
    """
    GatedGCN layer, idea based on 'Residual Gated Graph ConvNets'
    paper by Xavier Bresson and Thomas Laurent, ICLR 2018.
    https://arxiv.org/pdf/1711.07553v2.pdf
    """
    def __init__(self, in_channels, out_channels, batch_norm, dropout=None, residual=True):
        super().__init__()
        if dropout:
            # print(f'Using dropout: {dropout}')
            self.dropout = dropout
        else:
            # print(f'Using dropout: 0.00')
            self.dropout = 0.0
        self.batch_norm = batch_norm
        self.residual = residual

        if in_channels != out_channels:
            self.residual = False

        dtype=torch.float32

        self.A_1 = nn.Linear(in_channels, out_channels, dtype=dtype)
        self.A_2 = nn.Linear(in_channels, out_channels, dtype=dtype)
        self.A_3 = nn.Linear(in_channels, out_channels, dtype=dtype)  # Not used
        
        self.B_1 = nn.Linear(in_channels, out_channels, dtype=dtype)
        self.B_2 = nn.Linear(in_channels, out_channels, dtype=dtype)
        self.B_3 = nn.Linear(in_channels, out_channels, dtype=dtype)

        if batch_norm: # batch normalization
            # self.bn_h = nn.BatchNorm1d(out_channels, track_running_stats=True)
            # self.bn_e = nn.BatchNorm1d(out_channels, track_running_stats=True)
            self.bn_h = nn.Identity(out_channels, track_running_stats=True)
            self.bn_e = nn.Identity(out_channels, track_running_stats=True)
        else: # layer normalization
            self.bn_h = nn.LayerNorm(out_channels) 
            self.bn_e = nn.LayerNorm(out_channels) 

    def forward(self, g, h, e):
        """Return updated node representations."""
        with g.local_scope():
            h_in = h.clone()
            e_in = e.clone()
            
            # print(g.num_edges())
            # print(e.shape)

            g.ndata['h'] = h
            g.edata['e'] = e

            g.ndata['A1h'] = self.A_1(h)
            g.ndata['A2h'] = self.A_2(h)
            # g.ndata['A3h'] = self.A_3(h)

            g.ndata['B1h'] = self.B_1(h)
            g.ndata['B2h'] = self.B_2(h)
            g.edata['B3e'] = self.B_3(e)

            # Reference: https://github.com/graphdeeplearning/benchmarking-gnns/blob/master-dgl-0.6/layers/gated_gcn_layer.py

            # Forward-message passing
            g.apply_edges(fn.u_add_v('B1h', 'B2h', 'B12h'))
            e_ji = g.edata['B12h'] + g.edata['B3e']
            e_ji = self.bn_e(e_ji)
            e_ji = F.relu(e_ji)
            if self.residual:
                e_ji = e_ji + e_in
            g.edata['e_ji'] = e_ji
            g.edata['sigma_f'] = torch.sigmoid(g.edata['e_ji'])
            g.update_all(fn.u_mul_e('A2h', 'sigma_f', 'm_f'), fn.sum('m_f', 'sum_sigma_h_f'))
            g.update_all(fn.copy_e('sigma_f', 'm_f'), fn.sum('m_f', 'sum_sigma_f'))
            g.ndata['h_forward'] = g.ndata['sum_sigma_h_f'] / (g.ndata['sum_sigma_f'] + 1e-6)

            h = g.ndata['A1h'] + g.ndata['h_forward']

            h = self.bn_h(h)

            h = F.relu(h)

            if self.residual:
                h = h + h_in

            # h = F.dropout(h, self.dropout, training=self.training)
            e = g.edata['e_ji']

            return h, e
