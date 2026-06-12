'''
Zero-Day Inference Engine
This module defines the ZeroDayInferenceEngine class, 
which manages class prototypes and computes anomaly scores for incoming query embeddings. 
It uses Euclidean distance to compare the query embedding against known class prototypes 
and determines if the input is a zero-day candidate based on a predefined anomaly threshold. 
The engine is designed to be lightweight and efficient for real-time inference in edge environments.
'''

import os
from typing import Optional

import numpy as np

class ZeroDayInferenceEngine:
    def __init__(self, prototype_path: Optional[str] = None):
        """
        Manages the support set class prototypes in memory.
        In an MVP, these prototypes are initialized from the training phases.
        """
        self.embedding_dim = 128
        
        # In a full deployment, these are loaded from a serialized file or database.
        # Here we mock the classes mapping: 0=Normal, 1=DDoS, 2=PortScan, 3=Infiltration, 4=Zero-Day Candidate
        if prototype_path and os.path.exists(prototype_path):
            self.prototypes = np.load(prototype_path)
        else:
            # Generate static baseline vectors representing standard class prototypes for MVP verification
            print("Initializing baseline prototype embeddings...")
            self.prototypes = {
                0: np.random.randn(self.embedding_dim).astype(np.float32),  # Normal Traffic
                1: np.random.randn(self.embedding_dim).astype(np.float32),  # Known Attack Type A
                2: np.random.randn(self.embedding_dim).astype(np.float32)   # Known Attack Type B
            }

    def compute_anomaly_score(self, query_embedding: np.ndarray) -> dict:
        """
        Calculates the Euclidean distances from the input embedding to all known prototypes.
        Determines the zero-day anomaly score based on proximity deviation.
        """
        distances = {}
        for class_id, proto_vector in self.prototypes.items():
            # Traditional Euclidean Distance formula: sqrt(sum((a - b)^2))
            dist = np.linalg.norm(query_embedding - proto_vector)
            distances[class_id] = float(dist)
            
        # Identify closest known profile
        closest_class = min(distances, key=lambda k: distances[k])
        min_distance = distances[closest_class]
        
        # Compute normalized anomaly index (higher means farther away from any known baseline)
        # In an operational context, an absolute distance anomaly threshold is used (e.g., threshold > 15.0)
        anomaly_threshold = 12.0
        anomaly_score = min(1.0, min_distance / anomaly_threshold)
        
        return {
            "closest_known_class": closest_class,
            "anomaly_score": anomaly_score,
            "raw_distances": distances,
            "is_zero_day_candidate": bool(anomaly_score > 0.85)
        }