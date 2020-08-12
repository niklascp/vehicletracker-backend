"""Persistant model store"""

import os
import pathlib
import json
from datetime import datetime

import joblib
from typing import (Dict, List)

class LocalModelStore():
    """This class should only facilitate storage of models."""
    
    def __init__(self, path):
        self.path = path
        self.models : Dict[str, ] = {}        
        self.spatial_map : Dict[str, List[str]] = {}

    def load_metadata(self):
        """Loads metadata from disk."""    
        if not os.path.exists(self.path):
            os.makedirs(self.path)

        path = pathlib.Path(self.path)
        for file_name in path.glob('**/metadata.json'):
            with open(file_name, 'r') as f:
                model_metadata = json.load(f)
                self.add_model(model_metadata)

    def add_model(self, model_metadata):
        if not 'hash' in model_metadata:
            return

        model_hash_hex = model_metadata['hash']
        model_hash_simple_len = 6
        model_ref = model_hash_hex[:model_hash_simple_len]

        while model_ref in self.models and self.models[model_ref]['hash'] != model_hash_hex:
            model_hash_simple_len += 1
            model_ref = model_hash_hex[:model_hash_simple_len]

        model_metadata['ref'] = model_ref        
        exists = model_ref in self.models
        
        self.models[model_ref] = model_metadata

        for spatial_ref in model_metadata['spatialRefs']:
            if not spatial_ref in self.spatial_map:
                self.spatial_map[spatial_ref] = []
            if not exists:
                self.spatial_map[spatial_ref].append(model_ref)

    def list_models(self, model_name, spatial_ref, temporal_ref):
        """List relevent models for a given spatial and temporal reference"""
        if isinstance(spatial_ref, str):
            candidates = [self.models[ref] for ref in self.spatial_map.get(spatial_ref) or []]
        else:
            candidates = self.models.values()
            
        if model_name:
            candidates = filter(lambda x: x['model'] == model_name, candidates)

        if temporal_ref:
            iso_time = temporal_ref.isoformat()
            latest = {}
            for c in candidates:
                key = c['model'] + ':' +str(c['spatialRefs']) #TODO: This might not make sense?
                if key in latest:
                    if latest[key]['time'] < c['time'] and c['time'] <= iso_time:
                        latest[key] = c
                else:
                    latest[key] = c

            candidates = latest.values()

        return list(candidates)
