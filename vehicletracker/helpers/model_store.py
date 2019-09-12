import os
import joblib
import json

class LocalModelStore():
    
    def __init__(self, path):
        self.path = path

    def load_metadata(self):
        self.models = {}

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

        while model_hash_hex[:model_hash_simple_len] in self.models:
            model_hash_simple_len += 1

        model_metadata['hash'] = model_hash_hex[:model_hash_simple_len]

        self.models[model_metadata['hash']] = model_metadata

    def list_models(self):
        return list(self.models.values())

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
