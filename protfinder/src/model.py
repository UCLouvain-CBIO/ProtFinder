import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from loss import binarize3

__author__ = 'Grover'

class Classifier(nn.Module):
    '''
    Returns class log likelihoods for a batch of datapoints. Uses a DAG to connect classes.
    '''
    def __init__(self, embed_size, n_classes, hidden_size=256, num_layers=1, device=None):
        '''
        Initializes class variables.

        input : embed_size <int> : embedding size
        input : hidden_size <int> : hidden size for the LSTM
        input : n_classes <int> : number of classes
        input : num_layers <int> : number of layers of LSTM
        '''
        super(Classifier, self).__init__()
        self.device = device
        self.num_layers = num_layers
        self.embed_size = embed_size
        '''
        # Build Linear Layers for beginning of DAGs
        
        self.lstm1_1 = nn.Linear(embed_size, hidden_size)
        self.lstm1_2 = nn.Linear(embed_size, hidden_size)
        self.lstm1_3 = nn.Linear(embed_size, hidden_size)
        self.lstm1_4 = nn.Linear(embed_size, hidden_size)
        self.lstm1_5 = nn.Linear(embed_size, hidden_size)
        self.lstm1_6 = nn.Linear(embed_size, hidden_size)
        self.lstm1_7 = nn.Linear(embed_size, hidden_size)
        self.lstm1_8 = nn.Linear(embed_size, hidden_size)
        self.lstm1_9 = nn.Linear(embed_size, hidden_size)
        self.lstm1_10 = nn.Linear(embed_size, hidden_size)
        self.lstm1_11 = nn.Linear(embed_size, hidden_size)
        
        # Build Linear Layers for other members of DAGs
        self.lstm2_1 = nn.Linear(embed_size+1, hidden_size)
        self.lstm2_2 = nn.Linear(embed_size+1, hidden_size)
        self.lstm2_3 = nn.Linear(embed_size+1, hidden_size)
        self.lstm2_4 = nn.Linear(embed_size+1, hidden_size)
        self.lstm2_5 = nn.Linear(embed_size+1, hidden_size)
        self.lstm2_6 = nn.Linear(embed_size+1, hidden_size)
        self.lstm2_7 = nn.Linear(embed_size+1, hidden_size)
        self.lstm2_8 = nn.Linear(embed_size+1, hidden_size)
        self.lstm2_9 = nn.Linear(embed_size+1, hidden_size)
        self.lstm2_10 = nn.Linear(embed_size+1, hidden_size)
        self.lstm2_11 = nn.Linear(embed_size+1, hidden_size)
        self.lstm2_12 = nn.Linear(embed_size+1, hidden_size)
        self.lstm2_13 = nn.Linear(embed_size+1, hidden_size)
        self.lstm2_14 = nn.Linear(embed_size+1, hidden_size)
        self.lstm2_15 = nn.Linear(embed_size+1, hidden_size)
        self.lstm2_16 = nn.Linear(embed_size+1, hidden_size)
        self.lstm2_17 = nn.Linear(embed_size+1, hidden_size)
        
        '''
        self.lstm1_1 = nn.LSTM(
            embed_size, hidden_size, num_layers=num_layers, 
            batch_first=True, bidirectional=False)
        self.lstm1_2 = nn.LSTM(
            embed_size, hidden_size, num_layers=num_layers, 
            batch_first=True, bidirectional=False)
        self.lstm1_3 = nn.LSTM(
            embed_size, hidden_size, num_layers=num_layers, 
            batch_first=True, bidirectional=False)
        self.lstm1_4 = nn.LSTM(
            embed_size, hidden_size, num_layers=num_layers, 
            batch_first=True, bidirectional=False)
        self.lstm1_5 = nn.LSTM(
            embed_size, hidden_size, num_layers=num_layers, 
            batch_first=True, bidirectional=False)
        self.lstm1_6 = nn.LSTM(
            embed_size, hidden_size, num_layers=num_layers, 
            batch_first=True, bidirectional=False)
        self.lstm1_7 = nn.LSTM(
            embed_size, hidden_size, num_layers=num_layers, 
            batch_first=True, bidirectional=False)
        self.lstm1_8 = nn.LSTM(
            embed_size, hidden_size, num_layers=num_layers, 
            batch_first=True, bidirectional=False)
        self.lstm1_9 = nn.LSTM(
            embed_size, hidden_size, num_layers=num_layers, 
            batch_first=True, bidirectional=False)
        self.lstm1_10 = nn.LSTM(
            embed_size, hidden_size, num_layers=num_layers, 
            batch_first=True, bidirectional=False)
        self.lstm1_11 = nn.LSTM(
            embed_size, hidden_size, num_layers=num_layers, 
            batch_first=True, bidirectional=False)
        
        # Build LSTMs for other members of DAGs
        self.lstm2_1 = nn.LSTM(
            embed_size+1, hidden_size, num_layers=num_layers, 
            batch_first=True, bidirectional=False)
        self.lstm2_2 = nn.LSTM(
            embed_size+1, hidden_size, num_layers=num_layers, 
            batch_first=True, bidirectional=False)
        self.lstm2_3 = nn.LSTM(
            embed_size+1, hidden_size, num_layers=num_layers, 
            batch_first=True, bidirectional=False)
        self.lstm2_4 = nn.LSTM(
            embed_size+1, hidden_size, num_layers=num_layers, 
            batch_first=True, bidirectional=False)
        self.lstm2_5 = nn.LSTM(
            embed_size+1, hidden_size, num_layers=num_layers, 
            batch_first=True, bidirectional=False)
        self.lstm2_6 = nn.LSTM(
            embed_size+1, hidden_size, num_layers=num_layers, 
            batch_first=True, bidirectional=False)
        self.lstm2_7 = nn.LSTM(
            embed_size+1, hidden_size, num_layers=num_layers, 
            batch_first=True, bidirectional=False)
        self.lstm2_8 = nn.LSTM(
            embed_size+1, hidden_size, num_layers=num_layers, 
            batch_first=True, bidirectional=False)
        self.lstm2_9 = nn.LSTM(
            embed_size+1, hidden_size, num_layers=num_layers, 
            batch_first=True, bidirectional=False)
        self.lstm2_10 = nn.LSTM(
            embed_size+1, hidden_size, num_layers=num_layers, 
            batch_first=True, bidirectional=False)
        self.lstm2_11 = nn.LSTM(
            embed_size+1, hidden_size, num_layers=num_layers, 
            batch_first=True, bidirectional=False)
        self.lstm2_12 = nn.LSTM(
            embed_size+1, hidden_size, num_layers=num_layers, 
            batch_first=True, bidirectional=False)
        self.lstm2_13 = nn.LSTM(
            embed_size+1, hidden_size, num_layers=num_layers, 
            batch_first=True, bidirectional=False)
        self.lstm2_14 = nn.LSTM(
            embed_size+1, hidden_size, num_layers=num_layers, 
            batch_first=True, bidirectional=False)
        self.lstm2_15 = nn.LSTM(
            embed_size+1, hidden_size, num_layers=num_layers, 
            batch_first=True, bidirectional=False)
        self.lstm2_16 = nn.LSTM(
            embed_size+1, hidden_size, num_layers=num_layers, 
            batch_first=True, bidirectional=False)
        self.lstm2_17 = nn.LSTM(
            embed_size+1, hidden_size, num_layers=num_layers, 
            batch_first=True, bidirectional=False)
        

        # Define Fully Connected Layer for each class
        self.out1 = nn.Linear(hidden_size, n_classes)
        self.out2 = nn.Linear(hidden_size, n_classes)
        self.out3 = nn.Linear(hidden_size, n_classes)
        self.out4 = nn.Linear(hidden_size, n_classes)
        self.out5 = nn.Linear(hidden_size, n_classes)
        self.out6 = nn.Linear(hidden_size, n_classes)
        self.out7 = nn.Linear(hidden_size, n_classes)
        self.out8 = nn.Linear(hidden_size, n_classes)
        self.out9 = nn.Linear(hidden_size, n_classes)
        self.out10 = nn.Linear(hidden_size, n_classes)
        self.out11 = nn.Linear(hidden_size, n_classes)
        self.out12 = nn.Linear(hidden_size, n_classes)
        self.out13 = nn.Linear(hidden_size, n_classes)
        self.out14 = nn.Linear(hidden_size, n_classes)
        self.out15 = nn.Linear(hidden_size, n_classes)
        self.out16 = nn.Linear(hidden_size, n_classes)
        self.out17 = nn.Linear(hidden_size, n_classes)
        self.out18 = nn.Linear(hidden_size, n_classes)
        self.out19 = nn.Linear(hidden_size, n_classes)
        self.out20 = nn.Linear(hidden_size, n_classes)
        self.out21 = nn.Linear(hidden_size, n_classes)
        self.out22 = nn.Linear(hidden_size, n_classes)
        self.out23 = nn.Linear(hidden_size, n_classes)
        self.out24 = nn.Linear(hidden_size, n_classes)
        self.out25 = nn.Linear(hidden_size, n_classes)
        self.out26 = nn.Linear(hidden_size, n_classes)
        self.out27 = nn.Linear(hidden_size, n_classes)
        self.out28 = nn.Linear(hidden_size, n_classes)


        # Define Softmax Layer
        self.softmax = nn.LogSoftmax(dim=1)
        
    def forward(self, inp, batch_size):
        '''
        input : inp <torch tensor> : Input matrix of dimension (batch size, embedding size)
        input : batch_size <int> : Batch size

        return : output <torch tensor> : Ouput matrix of dimension (batch size, num classes) 
        '''
        # Common for all classes
        
        inp = inp.view(batch_size, 1, self.embed_size)
        latent_feats = inp.float()  #[batch_size, 1, hidden_size]
        # latent_feats = latent_feats[:, :, torch.randperm(latent_feats.size()[2])]
        
        # Cell Junction, Focal Adhesion Sites

        try:
            output_cell_junc, _ = self.lstm1_1(latent_feats)
        except:
            output_cell_junc = self.lstm1_1(latent_feats)
        output_cell_junc = output_cell_junc.squeeze(1)
        output_cell_junc = self.softmax(self.out1(output_cell_junc)) #[batch_size, n_classes]
        
        output_cell_junc = output_cell_junc.unsqueeze(1)    #[batch_size, 1, n_classes]
        
        # _, x = output_cell_junc.max(dim=2)
        x = binarize3(output_cell_junc, class_id=1, device=self.device)

        # latent_feats = latent_feats[:, :, torch.randperm(latent_feats.size()[2])]
        
        fas_feats = torch.cat((latent_feats.squeeze(1), x.float()), dim=1).unsqueeze(1)

        try:
            output_fas, _ = self.lstm2_1(fas_feats)
        except:
            output_fas = self.lstm2_1(fas_feats)
        output_fas = output_fas.squeeze(1)
        output_fas = self.softmax(self.out2(output_fas)) #[batch_size, n_classes]
        
        output_fas = output_fas.unsqueeze(1)    #[batch_size, 1, n_classes]

        
        # Cytokinetic Bridge

        # latent_feats = latent_feats[:, :, torch.randperm(latent_feats.size()[2])]

        try:
            output_cyto_bridge, _ = self.lstm1_2(latent_feats)
        except:
            output_cyto_bridge = self.lstm1_2(latent_feats)
        output_cyto_bridge = output_cyto_bridge.squeeze(1)
        output_cyto_bridge = self.softmax(self.out3(output_cyto_bridge)) #[batch_size, n_classes]
        
        output_cyto_bridge = output_cyto_bridge.unsqueeze(1)    #[batch_size, 1, n_classes]
        

        # Midbody, Midbody ring

        # latent_feats = latent_feats[:, :, torch.randperm(latent_feats.size()[2])]

        try:
            output_midbody, _ = self.lstm1_3(latent_feats)
        except:
            output_midbody = self.lstm1_3(latent_feats)
        output_midbody = output_midbody.squeeze(1)
        output_midbody = self.softmax(self.out4(output_midbody)) #[batch_size, n_classes]
        
        output_midbody = output_midbody.unsqueeze(1)    #[batch_size, 1, n_classes]
        
        # _, x = output_midbody.max(dim=2)
        x = binarize3(output_midbody, class_id=15, device=self.device)
        
        # latent_feats = latent_feats[:, :, torch.randperm(latent_feats.size()[2])]

        mid_feats = torch.cat((latent_feats.squeeze(1), x.float()), dim=1).unsqueeze(1)

        try:
            output_midr, _ = self.lstm2_2(mid_feats)
        except:
            output_midr = self.lstm2_2(mid_feats)
        output_midr = output_midr.squeeze(1)
        output_midr = self.softmax(self.out5(output_midr))   #[batch_size, n_classes]
        
        output_midr = output_midr.unsqueeze(1)  #[batch_size, 1, n_classes]


        # Plasma Membrane

        # latent_feats = latent_feats[:, :, torch.randperm(latent_feats.size()[2])]
        
        try:
            output_plasma_memb, _ = self.lstm1_4(latent_feats)
        except:
            output_plasma_memb = self.lstm1_4(latent_feats)
        output_plasma_memb = output_plasma_memb.squeeze(1)
        output_plasma_memb = self.softmax(self.out6(output_plasma_memb)) #[batch_size, n_classes]
        
        output_plasma_memb = output_plasma_memb.unsqueeze(1)    #[batch_size, 1, n_classes]
        

        # Cytosol

        # latent_feats = latent_feats[:, :, torch.randperm(latent_feats.size()[2])]

        try:
            output_cytosol, _ = self.lstm1_5(latent_feats)
        except:
            output_cytosol = self.lstm1_5(latent_feats)
        output_cytosol = output_cytosol.squeeze(1)
        output_cytosol = self.softmax(self.out7(output_cytosol)) #[batch_size, n_classes]
        
        output_cytosol = output_cytosol.unsqueeze(1)    #[batch_size, 1, n_classes]
        

        # Vesicles, Endoplasmic Reticulum

        # latent_feats = latent_feats[:, :, torch.randperm(latent_feats.size()[2])]

        try:
            output_vesicles, _ = self.lstm1_6(latent_feats)
        except:
            output_vesicles = self.lstm1_6(latent_feats)
        output_vesicles = output_vesicles.squeeze(1)
        output_vesicles = self.softmax(self.out8(output_vesicles)) #[batch_size, n_classes]
        
        output_vesicles = output_vesicles.unsqueeze(1)    #[batch_size, 1, n_classes]
        
        # _, x_ves = output_vesicles.max(dim=2)
        x_ves = binarize3(output_vesicles, class_id=27, device=self.device)

        # latent_feats = latent_feats[:, :, torch.randperm(latent_feats.size()[2])]

        er_feats = torch.cat((latent_feats.squeeze(1), x_ves.float()), dim=1).unsqueeze(1)

        try:
            output_er, _ = self.lstm2_3(er_feats)
        except:
            output_er = self.lstm2_3(er_feats)
        output_er = output_er.squeeze(1)
        output_er = self.softmax(self.out9(output_er))   #[batch_size, n_classes]
        
        output_er = output_er.unsqueeze(1)  #[batch_size, 1, n_classes]

        # Golgi Apparatus

        # latent_feats = latent_feats[:, :, torch.randperm(latent_feats.size()[2])]

        golgi_feats = torch.cat((latent_feats.squeeze(1), x_ves.float()), dim=1).unsqueeze(1)

        try:
            output_golgi, _ = self.lstm2_4(golgi_feats)
        except:
            output_golgi = self.lstm2_4(golgi_feats)
        output_golgi = output_golgi.squeeze(1)
        output_golgi = self.softmax(self.out10(output_golgi))   #[batch_size, n_classes]
        
        output_golgi = output_golgi.unsqueeze(1)  #[batch_size, 1, n_classes]

        # Endosomes

        # latent_feats = latent_feats[:, :, torch.randperm(latent_feats.size()[2])]

        endo_feats = torch.cat((latent_feats.squeeze(1), x_ves.float()), dim=1).unsqueeze(1)

        try:
            output_endo, _ = self.lstm2_5(endo_feats)
        except:
            output_endo = self.lstm2_5(endo_feats)
        output_endo = output_endo.squeeze(1)
        output_endo = self.softmax(self.out11(output_endo))   #[batch_size, n_classes]
        
        output_endo = output_endo.unsqueeze(1)  #[batch_size, 1, n_classes]

        # Mitchondria

        # latent_feats = latent_feats[:, :, torch.randperm(latent_feats.size()[2])]

        mito_feats = torch.cat((latent_feats.squeeze(1), x_ves.float()), dim=1).unsqueeze(1)

        try:
            output_mito, _ = self.lstm2_6(mito_feats)
        except:
            output_mito = self.lstm2_6(mito_feats)
        output_mito = output_mito.squeeze(1)
        output_mito = self.softmax(self.out12(output_mito))   #[batch_size, n_classes]
        
        output_mito = output_mito.unsqueeze(1)  #[batch_size, 1, n_classes]

        # Peroxisomes

        # latent_feats = latent_feats[:, :, torch.randperm(latent_feats.size()[2])]

        pero_feats = torch.cat((latent_feats.squeeze(1), x_ves.float()), dim=1).unsqueeze(1)

        try:
            output_pero, _ = self.lstm2_7(pero_feats)
        except:
            output_pero = self.lstm2_7(pero_feats)
        output_pero = output_pero.squeeze(1)
        output_pero = self.softmax(self.out13(output_pero))   #[batch_size, n_classes]
        
        output_pero = output_pero.unsqueeze(1)  #[batch_size, 1, n_classes]

        # Lysosomes

        # latent_feats = latent_feats[:, :, torch.randperm(latent_feats.size()[2])]

        lyso_feats = torch.cat((latent_feats.squeeze(1), x_ves.float()), dim=1).unsqueeze(1)

        try:
            output_lyso, _ = self.lstm2_8(lyso_feats)
        except:
            output_lyso = self.lstm2_8(lyso_feats)
        output_lyso = output_lyso.squeeze(1)
        output_lyso = self.softmax(self.out14(output_lyso))   #[batch_size, n_classes]
        
        output_lyso = output_lyso.unsqueeze(1)  #[batch_size, 1, n_classes]

        # Nuclear Membrane

        # latent_feats = latent_feats[:, :, torch.randperm(latent_feats.size()[2])]

        nucl_memb_feats = torch.cat((latent_feats.squeeze(1), x_ves.float()), dim=1).unsqueeze(1)

        try:
            output_nucl_memb, _ = self.lstm2_9(nucl_memb_feats)
        except:
            output_nucl_memb = self.lstm2_9(nucl_memb_feats)
        output_nucl_memb = output_nucl_memb.squeeze(1)
        output_nucl_memb = self.softmax(self.out15(output_nucl_memb))   #[batch_size, n_classes]
        
        output_nucl_memb = output_nucl_memb.unsqueeze(1)  #[batch_size, 1, n_classes]

        # Nucleoplasm, Nuclear Bodies, Nuclear Speckles

        # latent_feats = latent_feats[:, :, torch.randperm(latent_feats.size()[2])]

        nucleoplasm_feats = torch.cat((latent_feats.squeeze(1), x_ves.float()), dim=1).unsqueeze(1)

        try:
            output_nucleoplasm, _ = self.lstm2_10(nucleoplasm_feats)
        except:
            output_nucleoplasm = self.lstm2_10(nucleoplasm_feats)
        output_nucleoplasm = output_nucleoplasm.squeeze(1)
        output_nucleoplasm = self.softmax(self.out16(output_nucleoplasm)) #[batch_size, n_classes]
        
        output_nucleoplasm = output_nucleoplasm.unsqueeze(1)    #[batch_size, 1, n_classes]
        
        # _, x = output_nucleoplasm.max(dim=2)
        x = binarize3(output_nucleoplasm, class_id=24, device=self.device)

        # latent_feats = latent_feats[:, :, torch.randperm(latent_feats.size()[2])]

        nucl_bod_feats = torch.cat((latent_feats.squeeze(1), x.float()), dim=1).unsqueeze(1)

        try:
            output_nucl_bod, _ = self.lstm2_11(nucl_bod_feats)
        except:
            output_nucl_bod = self.lstm2_11(nucl_bod_feats)
        output_nucl_bod = output_nucl_bod.squeeze(1)
        output_nucl_bod = self.softmax(self.out17(output_nucl_bod))   #[batch_size, n_classes]
        
        output_nucl_bod = output_nucl_bod.unsqueeze(1)  #[batch_size, 1, n_classes]

        # _, x = output_nucl_bod.max(dim=2)
        x = binarize3(output_nucl_bod, class_id=19, device=self.device)

        # latent_feats = latent_feats[:, :, torch.randperm(latent_feats.size()[2])]

        nucl_spec_feats = torch.cat((latent_feats.squeeze(1), x.float()), dim=1).unsqueeze(1)

        try:
            output_nucl_spec, _ = self.lstm2_12(nucl_spec_feats)
        except:
            output_nucl_spec = self.lstm2_12(nucl_spec_feats)
        output_nucl_spec = output_nucl_spec.squeeze(1)
        output_nucl_spec = self.softmax(self.out18(output_nucl_spec))   #[batch_size, n_classes]
        
        output_nucl_spec = output_nucl_spec.unsqueeze(1)  #[batch_size, 1, n_classes]

        # Nucleoli, Nucleoli Fibrillar Center

        # latent_feats = latent_feats[:, :, torch.randperm(latent_feats.size()[2])]

        nucleoli_feats = torch.cat((latent_feats.squeeze(1), x_ves.float()), dim=1).unsqueeze(1)

        try:
            output_nucleoli, _ = self.lstm2_13(nucleoli_feats)
        except:
            output_nucleoli = self.lstm2_13(nucleoli_feats)
        output_nucleoli = output_nucleoli.squeeze(1)
        output_nucleoli = self.softmax(self.out19(output_nucleoli)) #[batch_size, n_classes]
        
        output_nucleoli = output_nucleoli.unsqueeze(1)    #[batch_size, 1, n_classes]
        
        # _, x = output_nucleoli.max(dim=2)
        x = binarize3(output_nucl_spec, class_id=22, device=self.device)

        # latent_feats = latent_feats[:, :, torch.randperm(latent_feats.size()[2])]

        nfc_feats = torch.cat((latent_feats.squeeze(1), x.float()), dim=1).unsqueeze(1)

        try:
            output_nfc, _ = self.lstm2_14(nfc_feats)
        except:
            output_nfc = self.lstm2_14(nfc_feats)
        output_nfc = output_nfc.squeeze(1)
        output_nfc = self.softmax(self.out20(output_nfc))   #[batch_size, n_classes]
        
        output_nfc = output_nfc.unsqueeze(1)  #[batch_size, 1, n_classes]


        # Cytoplasmic Bodies

        # latent_feats = latent_feats[:, :, torch.randperm(latent_feats.size()[2])]

        try:
            output_cyto_bodies, _ = self.lstm1_7(latent_feats)
        except:
            output_cyto_bodies = self.lstm1_7(latent_feats)
        output_cyto_bodies = output_cyto_bodies.squeeze(1)
        output_cyto_bodies = self.softmax(self.out21(output_cyto_bodies)) #[batch_size, n_classes]
        
        output_cyto_bodies = output_cyto_bodies.unsqueeze(1)    #[batch_size, 1, n_classes]
        

        # Lipid Droplets

        # latent_feats = latent_feats[:, :, torch.randperm(latent_feats.size()[2])]

        try:
            output_lipid, _ = self.lstm1_8(latent_feats)
        except:
            output_lipid = self.lstm1_8(latent_feats)
        output_lipid = output_lipid.squeeze(1)
        output_lipid = self.softmax(self.out22(output_lipid)) #[batch_size, n_classes]
        
        output_lipid = output_lipid.unsqueeze(1)    #[batch_size, 1, n_classes]
        

        # Intermediate Filaments

        # latent_feats = latent_feats[:, :, torch.randperm(latent_feats.size()[2])]

        try:
            output_inter_fila, _ = self.lstm1_9(latent_feats)
        except:
            output_inter_fila = self.lstm1_9(latent_feats)
        output_inter_fila = output_inter_fila.squeeze(1)
        output_inter_fila = self.softmax(self.out23(output_inter_fila)) #[batch_size, n_classes]
        
        output_inter_fila = output_inter_fila.unsqueeze(1)    #[batch_size, 1, n_classes]
        

        # Actin Filaments

        # latent_feats = latent_feats[:, :, torch.randperm(latent_feats.size()[2])]

        try:
            output_actin_fila, _ = self.lstm1_10(latent_feats)
        except:
            output_actin_fila = self.lstm1_10(latent_feats)
        output_actin_fila = output_actin_fila.squeeze(1)
        output_actin_fila = self.softmax(self.out24(output_actin_fila)) #[batch_size, n_classes]
        
        output_actin_fila = output_actin_fila.unsqueeze(1)    #[batch_size, 1, n_classes]


        # Microtubules, Mitotic Spindle

        # latent_feats = latent_feats[:, :, torch.randperm(latent_feats.size()[2])]

        try:
            output_micro, _ = self.lstm1_11(latent_feats)
        except:
            output_micro = self.lstm1_11(latent_feats)
        output_micro = output_micro.squeeze(1)
        output_micro = self.softmax(self.out25(output_micro)) #[batch_size, n_classes]
        
        output_micro = output_micro.unsqueeze(1)    #[batch_size, 1, n_classes]
        
        _, micro_x = output_micro.max(dim=2)

        # latent_feats = latent_feats[:, :, torch.randperm(latent_feats.size()[2])]

        mito_spindle_feats = torch.cat((latent_feats.squeeze(1), micro_x.float()), dim=1).unsqueeze(1)

        try:
            output_mito_spindle, _ = self.lstm2_15(mito_spindle_feats)
        except:
            output_mito_spindle = self.lstm2_15(mito_spindle_feats)
        output_mito_spindle = output_mito_spindle.squeeze(1)
        output_mito_spindle = self.softmax(self.out26(output_mito_spindle))   #[batch_size, n_classes]
        
        output_mito_spindle = output_mito_spindle.unsqueeze(1)  #[batch_size, 1, n_classes]


        # Microtubules, Centrosome, Centriolar Satellite

        # latent_feats = latent_feats[:, :, torch.randperm(latent_feats.size()[2])]

        centrosome_feats = torch.cat((latent_feats.squeeze(1), micro_x.float()), dim=1).unsqueeze(1)

        try:
            output_centrosome, _ = self.lstm2_16(centrosome_feats)
        except:
            output_centrosome = self.lstm2_16(centrosome_feats)
        output_centrosome = output_centrosome.squeeze(1)
        output_centrosome = self.softmax(self.out27(output_centrosome))   #[batch_size, n_classes]
        
        output_centrosome = output_centrosome.unsqueeze(1)  #[batch_size, 1, n_classes]

        _, x = output_centrosome.max(dim=2)

        # latent_feats = latent_feats[:, :, torch.randperm(latent_feats.size()[2])]

        centr_sat_feats = torch.cat((latent_feats.squeeze(1), x.float()), dim=1).unsqueeze(1)

        try:
            output_centr_sat, _ = self.lstm2_17(centr_sat_feats)
        except:
            output_centr_sat = self.lstm2_17(centr_sat_feats)
        output_centr_sat = output_centr_sat.squeeze(1)
        output_centr_sat = self.softmax(self.out28(output_centr_sat))   #[batch_size, n_classes]
        
        output_centr_sat = output_centr_sat.unsqueeze(1)  #[batch_size, 1, n_classes]
        
        # Concatenating all the ouptut probabilities
        output = torch.cat((
            output_actin_fila,
            output_cell_junc,
            output_centr_sat,
            output_centrosome,
            output_cyto_bridge,
            output_cyto_bodies,
            output_cytosol,
            output_er,
            output_endo, 
            output_fas,
            output_golgi,
            output_inter_fila,
            output_lipid,
            output_lyso,
            output_micro,
            output_midbody,
            output_midr,
            output_mito,
            output_mito_spindle,
            output_nucl_bod,
            output_nucl_memb,
            output_nucl_spec,
            output_nucleoli,
            output_nfc,
            output_nucleoplasm,
            output_pero,
            output_plasma_memb,
            output_vesicles), dim=1)
        return output

class SimpleClassifier(nn.Module):
    '''
    Returns class log likelihoods for a batch of datapoints. Does not have connected classes.
    '''
    def __init__(self, embed_size, n_classes, hidden_size=256, num_layers=1):
        '''
        Initializes class variables.

        input : embed_size <int> : embedding size
        input : hidden_size <int> : hidden size for the LSTM
        input : n_classes <int> : number of classes
        input : num_layers <int> : number of layers of LSTM
        '''
        super(SimpleClassifier, self).__init__()
        
        self.num_layers = num_layers
        self.embed_size = embed_size

        # Build LSTMs for beginning of DAGs
        self.lstm1_1 = nn.LSTM(
            embed_size, hidden_size, num_layers=num_layers, 
            batch_first=True, bidirectional=False)
        self.lstm1_2 = nn.LSTM(
            embed_size, hidden_size, num_layers=num_layers, 
            batch_first=True, bidirectional=False)
        self.lstm1_3 = nn.LSTM(
            embed_size, hidden_size, num_layers=num_layers, 
            batch_first=True, bidirectional=False)
        self.lstm1_4 = nn.LSTM(
            embed_size, hidden_size, num_layers=num_layers, 
            batch_first=True, bidirectional=False)
        self.lstm1_5 = nn.LSTM(
            embed_size, hidden_size, num_layers=num_layers, 
            batch_first=True, bidirectional=False)
        self.lstm1_6 = nn.LSTM(
            embed_size, hidden_size, num_layers=num_layers, 
            batch_first=True, bidirectional=False)
        self.lstm1_7 = nn.LSTM(
            embed_size, hidden_size, num_layers=num_layers, 
            batch_first=True, bidirectional=False)
        self.lstm1_8 = nn.LSTM(
            embed_size, hidden_size, num_layers=num_layers, 
            batch_first=True, bidirectional=False)
        self.lstm1_9 = nn.LSTM(
            embed_size, hidden_size, num_layers=num_layers, 
            batch_first=True, bidirectional=False)
        self.lstm1_10 = nn.LSTM(
            embed_size, hidden_size, num_layers=num_layers, 
            batch_first=True, bidirectional=False)
        self.lstm1_11 = nn.LSTM(
            embed_size, hidden_size, num_layers=num_layers, 
            batch_first=True, bidirectional=False)
        
        self.lstm1_12 = nn.LSTM(
            embed_size, hidden_size, num_layers=num_layers, 
            batch_first=True, bidirectional=False)
        self.lstm1_13 = nn.LSTM(
            embed_size, hidden_size, num_layers=num_layers, 
            batch_first=True, bidirectional=False)
        self.lstm1_14 = nn.LSTM(
            embed_size, hidden_size, num_layers=num_layers, 
            batch_first=True, bidirectional=False)
        self.lstm1_15 = nn.LSTM(
            embed_size, hidden_size, num_layers=num_layers, 
            batch_first=True, bidirectional=False)
        self.lstm1_16 = nn.LSTM(
            embed_size, hidden_size, num_layers=num_layers, 
            batch_first=True, bidirectional=False)
        self.lstm1_17 = nn.LSTM(
            embed_size, hidden_size, num_layers=num_layers, 
            batch_first=True, bidirectional=False)
        self.lstm1_18 = nn.LSTM(
            embed_size, hidden_size, num_layers=num_layers, 
            batch_first=True, bidirectional=False)
        self.lstm1_19 = nn.LSTM(
            embed_size, hidden_size, num_layers=num_layers, 
            batch_first=True, bidirectional=False)
        self.lstm1_20 = nn.LSTM(
            embed_size, hidden_size, num_layers=num_layers, 
            batch_first=True, bidirectional=False)
        self.lstm1_21 = nn.LSTM(
            embed_size, hidden_size, num_layers=num_layers, 
            batch_first=True, bidirectional=False)
        self.lstm1_22 = nn.LSTM(
            embed_size, hidden_size, num_layers=num_layers, 
            batch_first=True, bidirectional=False)
        self.lstm1_23 = nn.LSTM(
            embed_size, hidden_size, num_layers=num_layers, 
            batch_first=True, bidirectional=False)
        self.lstm1_24 = nn.LSTM(
            embed_size, hidden_size, num_layers=num_layers, 
            batch_first=True, bidirectional=False)
        self.lstm1_25 = nn.LSTM(
            embed_size, hidden_size, num_layers=num_layers, 
            batch_first=True, bidirectional=False)
        self.lstm1_26 = nn.LSTM(
            embed_size, hidden_size, num_layers=num_layers, 
            batch_first=True, bidirectional=False)
        self.lstm1_27 = nn.LSTM(
            embed_size, hidden_size, num_layers=num_layers, 
            batch_first=True, bidirectional=False)
        self.lstm1_28 = nn.LSTM(
            embed_size, hidden_size, num_layers=num_layers, 
            batch_first=True, bidirectional=False)

        # Define Fully Connected Layer for each class
        self.out1 = nn.Linear(hidden_size, n_classes)
        self.out2 = nn.Linear(hidden_size, n_classes)
        self.out3 = nn.Linear(hidden_size, n_classes)
        self.out4 = nn.Linear(hidden_size, n_classes)
        self.out5 = nn.Linear(hidden_size, n_classes)
        self.out6 = nn.Linear(hidden_size, n_classes)
        self.out7 = nn.Linear(hidden_size, n_classes)
        self.out8 = nn.Linear(hidden_size, n_classes)
        self.out9 = nn.Linear(hidden_size, n_classes)
        self.out10 = nn.Linear(hidden_size, n_classes)
        self.out11 = nn.Linear(hidden_size, n_classes)
        self.out12 = nn.Linear(hidden_size, n_classes)
        self.out13 = nn.Linear(hidden_size, n_classes)
        self.out14 = nn.Linear(hidden_size, n_classes)
        self.out15 = nn.Linear(hidden_size, n_classes)
        self.out16 = nn.Linear(hidden_size, n_classes)
        self.out17 = nn.Linear(hidden_size, n_classes)
        self.out18 = nn.Linear(hidden_size, n_classes)
        self.out19 = nn.Linear(hidden_size, n_classes)
        self.out20 = nn.Linear(hidden_size, n_classes)
        self.out21 = nn.Linear(hidden_size, n_classes)
        self.out22 = nn.Linear(hidden_size, n_classes)
        self.out23 = nn.Linear(hidden_size, n_classes)
        self.out24 = nn.Linear(hidden_size, n_classes)
        self.out25 = nn.Linear(hidden_size, n_classes)
        self.out26 = nn.Linear(hidden_size, n_classes)
        self.out27 = nn.Linear(hidden_size, n_classes)
        self.out28 = nn.Linear(hidden_size, n_classes)
        
        # Define Softmax Layer for each class
        self.softmax = nn.LogSoftmax(dim=1)
        
    def forward(self, inp, batch_size):
        '''
        input : inp <torch tensor> : Input matrix of dimension (batch size, embedding size)
        input : batch_size <int> : Batch size

        return : output <torch tensor> : Ouput matrix of dimension (batch size, num classes) 
        '''
        # Common for all classes
        
        inp = inp.view(batch_size, 1, self.embed_size)
        latent_feats = inp.float()  #[batch_size, 1, hidden_size]
        
        # Cell Junction, Focal Adhesion Sites

        output_cell_junc, _ = self.lstm1_1(latent_feats)
        output_cell_junc = output_cell_junc.squeeze(1)
        output_cell_junc = self.softmax(self.out1(output_cell_junc)) #[batch_size, n_classes]
        
        output_cell_junc = output_cell_junc.unsqueeze(1)    #[batch_size, 1, n_classes]
        

        output_fas, _ = self.lstm1_12(latent_feats)
        output_fas = output_fas.squeeze(1)
        output_fas = self.softmax(self.out2(output_fas)) #[batch_size, n_classes]
        
        output_fas = output_fas.unsqueeze(1)    #[batch_size, 1, n_classes]

        
        # Cytokinetic Bridge

        output_cyto_bridge, _ = self.lstm1_2(latent_feats)
        output_cyto_bridge = output_cyto_bridge.squeeze(1)
        output_cyto_bridge = self.softmax(self.out3(output_cyto_bridge)) #[batch_size, n_classes]
        
        output_cyto_bridge = output_cyto_bridge.unsqueeze(1)    #[batch_size, 1, n_classes]
        

        # Midbody, Midbody ring

        output_midbody, _ = self.lstm1_3(latent_feats)
        output_midbody = output_midbody.squeeze(1)
        output_midbody = self.softmax(self.out4(output_midbody)) #[batch_size, n_classes]
        
        output_midbody = output_midbody.unsqueeze(1)    #[batch_size, 1, n_classes]
        
        
        output_midr, _ = self.lstm1_13(latent_feats)
        output_midr = output_midr.squeeze(1)
        output_midr = self.softmax(self.out5(output_midr))   #[batch_size, n_classes]
        
        output_midr = output_midr.unsqueeze(1)  #[batch_size, 1, n_classes]


        # Plasma Membrane

        output_plasma_memb, _ = self.lstm1_4(latent_feats)
        output_plasma_memb = output_plasma_memb.squeeze(1)
        output_plasma_memb = self.softmax(self.out6(output_plasma_memb)) #[batch_size, n_classes]
        
        output_plasma_memb = output_plasma_memb.unsqueeze(1)    #[batch_size, 1, n_classes]
        

        # Cytosol

        output_cytosol, _ = self.lstm1_5(latent_feats)
        output_cytosol = output_cytosol.squeeze(1)
        output_cytosol = self.softmax(self.out7(output_cytosol)) #[batch_size, n_classes]
        
        output_cytosol = output_cytosol.unsqueeze(1)    #[batch_size, 1, n_classes]
        

        # Vesicles, Endoplasmic Reticulum

        output_vesicles, _ = self.lstm1_6(latent_feats)
        output_vesicles = output_vesicles.squeeze(1)
        output_vesicles = self.softmax(self.out8(output_vesicles)) #[batch_size, n_classes]
        
        output_vesicles = output_vesicles.unsqueeze(1)    #[batch_size, 1, n_classes]
        
        
        output_er, _ = self.lstm1_14(latent_feats)
        output_er = output_er.squeeze(1)
        output_er = self.softmax(self.out9(output_er))   #[batch_size, n_classes]
        
        output_er = output_er.unsqueeze(1)  #[batch_size, 1, n_classes]

        # Golgi Apparatus

        output_golgi, _ = self.lstm1_15(latent_feats)
        output_golgi = output_golgi.squeeze(1)
        output_golgi = self.softmax(self.out10(output_golgi))   #[batch_size, n_classes]
        
        output_golgi = output_golgi.unsqueeze(1)  #[batch_size, 1, n_classes]

        # Endosomes

        output_endo, _ = self.lstm1_16(latent_feats)
        output_endo = output_endo.squeeze(1)
        output_endo = self.softmax(self.out11(output_endo))   #[batch_size, n_classes]
        
        output_endo = output_endo.unsqueeze(1)  #[batch_size, 1, n_classes]

        # Mitchondria

        output_mito, _ = self.lstm1_17(latent_feats)
        output_mito = output_mito.squeeze(1)
        output_mito = self.softmax(self.out12(output_mito))   #[batch_size, n_classes]
        
        output_mito = output_mito.unsqueeze(1)  #[batch_size, 1, n_classes]

        # Peroxisomes

        output_pero, _ = self.lstm1_18(latent_feats)
        output_pero = output_pero.squeeze(1)
        output_pero = self.softmax(self.out13(output_pero))   #[batch_size, n_classes]
        
        output_pero = output_pero.unsqueeze(1)  #[batch_size, 1, n_classes]

        # Lysosomes

        output_lyso, _ = self.lstm1_19(latent_feats)
        output_lyso = output_lyso.squeeze(1)
        output_lyso = self.softmax(self.out14(output_lyso))   #[batch_size, n_classes]
        
        output_lyso = output_lyso.unsqueeze(1)  #[batch_size, 1, n_classes]

        # Nuclear Membrane

        output_nucl_memb, _ = self.lstm1_20(latent_feats)
        output_nucl_memb = output_nucl_memb.squeeze(1)
        output_nucl_memb = self.softmax(self.out15(output_nucl_memb))   #[batch_size, n_classes]
        
        output_nucl_memb = output_nucl_memb.unsqueeze(1)  #[batch_size, 1, n_classes]

        # Nucleoplasm, Nuclear Bodies, Nuclear Speckles

        output_nucleoplasm, _ = self.lstm1_21(latent_feats)
        output_nucleoplasm = output_nucleoplasm.squeeze(1)
        output_nucleoplasm = self.softmax(self.out16(output_nucleoplasm)) #[batch_size, n_classes]
        
        output_nucleoplasm = output_nucleoplasm.unsqueeze(1)    #[batch_size, 1, n_classes]
        
        
        output_nucl_bod, _ = self.lstm1_22(latent_feats)
        output_nucl_bod = output_nucl_bod.squeeze(1)
        output_nucl_bod = self.softmax(self.out17(output_nucl_bod))   #[batch_size, n_classes]
        
        output_nucl_bod = output_nucl_bod.unsqueeze(1)  #[batch_size, 1, n_classes]

        
        output_nucl_spec, _ = self.lstm1_23(latent_feats)
        output_nucl_spec = output_nucl_spec.squeeze(1)
        output_nucl_spec = self.softmax(self.out18(output_nucl_spec))   #[batch_size, n_classes]
        
        output_nucl_spec = output_nucl_spec.unsqueeze(1)  #[batch_size, 1, n_classes]

        # Nucleoli, Nucleoli Fibrillar Center

        output_nucleoli, _ = self.lstm1_24(latent_feats)
        output_nucleoli = output_nucleoli.squeeze(1)
        output_nucleoli = self.softmax(self.out19(output_nucleoli)) #[batch_size, n_classes]
        
        output_nucleoli = output_nucleoli.unsqueeze(1)    #[batch_size, 1, n_classes]
        
        
        output_nfc, _ = self.lstm1_25(latent_feats)
        output_nfc = output_nfc.squeeze(1)
        output_nfc = self.softmax(self.out20(output_nfc))   #[batch_size, n_classes]
        
        output_nfc = output_nfc.unsqueeze(1)  #[batch_size, 1, n_classes]


        # Cytoplasmic Bodies

        output_cyto_bodies, _ = self.lstm1_7(latent_feats)
        output_cyto_bodies = output_cyto_bodies.squeeze(1)
        output_cyto_bodies = self.softmax(self.out21(output_cyto_bodies)) #[batch_size, n_classes]
        
        output_cyto_bodies = output_cyto_bodies.unsqueeze(1)    #[batch_size, 1, n_classes]
        

        # Lipid Droplets

        output_lipid, _ = self.lstm1_8(latent_feats)
        output_lipid = output_lipid.squeeze(1)
        output_lipid = self.softmax(self.out22(output_lipid)) #[batch_size, n_classes]
        
        output_lipid = output_lipid.unsqueeze(1)    #[batch_size, 1, n_classes]
        

        # Intermediate Filaments

        output_inter_fila, _ = self.lstm1_9(latent_feats)
        output_inter_fila = output_inter_fila.squeeze(1)
        output_inter_fila = self.softmax(self.out23(output_inter_fila)) #[batch_size, n_classes]
        
        output_inter_fila = output_inter_fila.unsqueeze(1)    #[batch_size, 1, n_classes]
        

        # Actin Filaments

        output_actin_fila, _ = self.lstm1_10(latent_feats)
        output_actin_fila = output_actin_fila.squeeze(1)
        output_actin_fila = self.softmax(self.out24(output_actin_fila)) #[batch_size, n_classes]
        
        output_actin_fila = output_actin_fila.unsqueeze(1)    #[batch_size, 1, n_classes]


        # Microtubules, Mitotic Spindle

        output_micro, _ = self.lstm1_11(latent_feats)
        output_micro = output_micro.squeeze(1)
        output_micro = self.softmax(self.out25(output_micro)) #[batch_size, n_classes]
        
        output_micro = output_micro.unsqueeze(1)    #[batch_size, 1, n_classes]
        
        
        output_mito_spindle, _ = self.lstm1_26(latent_feats)
        output_mito_spindle = output_mito_spindle.squeeze(1)
        output_mito_spindle = self.softmax(self.out26(output_mito_spindle))   #[batch_size, n_classes]
        
        output_mito_spindle = output_mito_spindle.unsqueeze(1)  #[batch_size, 1, n_classes]


        # Centrosome, Centriolar Satellite

        output_centrosome, _ = self.lstm1_27(latent_feats)
        output_centrosome = output_centrosome.squeeze(1)
        output_centrosome = self.softmax(self.out27(output_centrosome))   #[batch_size, n_classes]
        
        output_centrosome = output_centrosome.unsqueeze(1)  #[batch_size, 1, n_classes]

        
        output_centr_sat, _ = self.lstm1_28(latent_feats)
        output_centr_sat = output_centr_sat.squeeze(1)
        output_centr_sat = self.softmax(self.out28(output_centr_sat))   #[batch_size, n_classes]
        
        output_centr_sat = output_centr_sat.unsqueeze(1)  #[batch_size, 1, n_classes]
        
        # Concatenating all the ouptut probabilities
        output = torch.cat((
            output_actin_fila,
            output_cell_junc,
            output_centr_sat,
            output_centrosome,
            output_cyto_bridge,
            output_cyto_bodies,
            output_cytosol,
            output_er,
            output_endo, 
            output_fas,
            output_golgi,
            output_inter_fila,
            output_lipid,
            output_lyso,
            output_micro,
            output_midbody,
            output_midr,
            output_mito,
            output_mito_spindle,
            output_nucl_bod,
            output_nucl_memb,
            output_nucl_spec,
            output_nucleoli,
            output_nfc,
            output_nucleoplasm,
            output_pero,
            output_plasma_memb,
            output_vesicles), dim=1)
        return output

if __name__ == "__main__":
    x = np.array([[1, 2, 2, 0], [3, 5, -2, 1]])
    device = torch.device("cuda")
    model = SimpleClassifier(embed_size=4, hidden_size=3, n_classes=5).to(device)
    
    x = torch.from_numpy(x).to(device)
    y = model(x, batch_size=2)
    print(y.shape)
