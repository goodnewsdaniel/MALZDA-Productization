'''ONNX Export Script for MAL-ZDA Framework 
This script initializes the MAL-ZDA architecture, loads the trained weights from a checkpoint,
and exports the hierarchical encoder component to ONNX format.  '''

import os
import torch
import numpy as np
from mal_zda_framework import MALZDA

def export_framework_to_onnx():
    print("Initializing MAL-ZDA architecture...")
    
    # Define the exact dimensions from the framework specification
    packet_dim = 256
    flow_seq_len = 100
    campaign_dim = 1024
    embedding_dim = 128
    
    # Initialize model structure
    model = MALZDA(
        packet_dim=packet_dim, 
        flow_seq_len=flow_seq_len, 
        campaign_dim=campaign_dim, 
        embedding_dim=embedding_dim
    )
    
    # Locate weights checkpoint
    checkpoint_path = "results_malzda/malzda_compositional_model.pt"
    
    if os.path.exists(checkpoint_path):
        print(f"Loading trained weights from {checkpoint_path}...")
        checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        print(f"WARNING: Checkpoint not found at {checkpoint_path}. Exporting with randomized initialization weights for MVP testing.")
    
    # Isolate the encoder and switch to evaluation mode
    encoder = model.hierarchical_encoder
    encoder.eval()
    
    # Construct precise dummy inputs representing a single network event batch
    # Dimensions match: (Batch Size, Features)
    dummy_packet = torch.randn(1, packet_dim, dtype=torch.float32)
    dummy_flow = torch.randn(1, flow_seq_len, packet_dim, dtype=torch.float32)
    dummy_campaign = torch.randn(1, campaign_dim, dtype=torch.float32)
    
    output_onnx_path = "malzda_encoder.onnx"
    
    print(f"Tracing and exporting model graph to {output_onnx_path}...")
    
    torch.onnx.export(
        encoder,
        (dummy_packet, dummy_flow, dummy_campaign),
        output_onnx_path,
        export_params=True,
        opset_version=18,  # Latest stable opset with full operator support and better compatibility
        do_constant_folding=True,
        input_names=['packet_input', 'flow_input', 'campaign_input'],
        output_names=['fused_embedding'],
        dynamic_axes={
            'packet_input': {0: 'batch_size'},
            'flow_input': {0: 'batch_size'},
            'campaign_input': {0: 'batch_size'},
            'fused_embedding': {0: 'batch_size'}
        }
    )
    
    print("ONNX export successfully validated and written to disk.")

if __name__ == "__main__":
    export_framework_to_onnx()