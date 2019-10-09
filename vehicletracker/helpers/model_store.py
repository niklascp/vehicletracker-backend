"""Persistant model store"""

import os
import json
from datetime import datetime

import joblib
from typing import (Dict, List)

class LocalModelStore():
    
    def __init__(self, path):
        self.path = path
        self.models : Dict[str, ] = {}
        self.spatial_map : Dict[str, List[str]] = {}

    def load_metadata(self):
        """Loads metadata from disk."""    
        if not os.path.exists(self.path):
            os.makedirs(self.path)

        for file_name in os.listdir(self.path):
            if not file_name.endswith(".json"):
                continue

            metadata_file_path = os.path.join(self.path, file_name)

            with open(metadata_file_path, 'r') as f:
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

        if not model_metadata['linkRef'] in self.spatial_map:
            self.spatial_map[model_metadata['linkRef']] = []
        if not exists:
            self.spatial_map[model_metadata['linkRef']].append(model_ref)


    def list_models(self, model_name, spatial_ref, temporal_ref):
        """List relevent models for a given spatial and temporal reference"""
        if spatial_ref:
            candidates = [self.models[ref] for ref in self.spatial_map[spatial_ref]]
        else:
            candidates = self.models.values()
            
        if model_name:
            candidates = filter(lambda x: x['model'] == model_name, candidates)

        if temporal_ref:
            iso_time = temporal_ref.isoformat()
            latest = {}
            for c in candidates:
                key = c['model'] + ':' + c['linkRef']
                if key in latest:
                    if latest[key]['time'] < c['time'] and c['time'] <= iso_time:
                        latest[key] = c
                else:
                    latest[key] = c

            candidates = latest.values()

        return list(candidates)

    def load_model(self, model_hash):
        pass
        #for file_name in os.listdir(MODEL_CACHE_PATH):
        #    if (file_name.endswith(".json")):
        #        _LOGGER.info(f'Loading cached model from data from {file_name}')
        #        metadata_file_path = os.path.join(MODEL_CACHE_PATH, file_name)
        #        model_file_path = os.path.splitext(os.path.join(MODEL_CACHE_PATH, file_name))[0] + '.joblib'
        #        
        #        with open(metadata_file_path, 'r') as f:
        #            model_metadata = json.load(f)
        #        with open(model_file_path, 'rb') as f:
        #            model = joblib.load(f)
        #        
        #        LINK_MODELS[model_metadata['linkRef']] = { 'model': model, 'metadata': model_metadata }
