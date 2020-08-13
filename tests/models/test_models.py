class MockModel():

    def __init__(self, node):
        pass

    def train(self, parameters):
        return {
            'type': 'mock',
            'spatialRefs': 'mock'
        }

    def save(self, model_store, metadata):
        pass
    
    def restore(self, model_store, metadata):
        pass
