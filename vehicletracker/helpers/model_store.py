"""Persistant model store"""

import os
import shutil
import pathlib
import json
from datetime import datetime
import logging

import joblib
from typing import (Dict, List)

_LOGGER = logging.getLogger(__name__)

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

        # For models that specifically returns spatial refs, update the lookup 
        if 'spatialRefs' in model_metadata:
            for spatial_ref in model_metadata['spatialRefs']:
                if not spatial_ref in self.spatial_map:
                    self.spatial_map[spatial_ref] = []
                if not exists:
                    self.spatial_map[spatial_ref].append(model_ref)

    def list_models(self, model_name, spatial_ref, temporal_ref):
        """List relevent models for a given spatial and temporal reference"""
        if isinstance(spatial_ref, str):
            candidates = [self.models[ref] for ref in self.spatial_map.get(spatial_ref) or []]
            if ':' in spatial_ref:  # Link spatial ref - TODO: This should be made more explicit.
                candidates.extend([x for x in self.models.values() if x['type'] == 'Link' and x['spatialRef'] == 'LN:*'])
            else:                   # Stop spatial ref - TODO: This should be made more explicit.
                candidates.extend([x for x in self.models.values() if x['type'] == 'Dwell' and x['spatialRef'] == 'SP:*'])
        else:
            candidates = self.models.values()
            
        if model_name:
            candidates = list(filter(lambda x: x['model'] == model_name, candidates))

        if temporal_ref:
            iso_time = temporal_ref.replace(tzinfo=None).isoformat()
            latest = {}
            for c in candidates:
                key = c['model']
                model_iso_time =  c['time'] if c['time'] is not 'latest' else c['trained']
                if key in latest:
                    latest_iso_time =  latest[key]['time'] if latest[key]['time'] is not 'latest' else latest[key]['trained']
                    if latest_iso_time < model_iso_time and model_iso_time <= iso_time:
                        latest[key] = c
                elif model_iso_time <= iso_time:
                    latest[key] = c

            candidates = latest.values()

        return list(candidates)

    def delete_model(self, model_ref : str):
        # Start by removing reference from spatial map, will cause the model to be stoped being used.
        for spatial_ref, model_refs in self.spatial_map.items():
            if model_ref in model_refs:
                model_refs.remove(model_ref)
        # Lookup hash and model name and remove from disk
        hash = self.models[model_ref]['hash']
        model_name = self.models[model_ref]['model']
        path = pathlib.Path(self.path) / model_name / hash
        _LOGGER.info("Deleteting model directory %s", path)
        shutil.rmtree(path)
        # Finally pop it from the model collection and publish message for predictor components to free up resorces.
        model = self.models.pop(model_ref, None)
