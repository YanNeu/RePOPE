

class VLMEvaluator():
    
    def __init__(self, vlm_name, *args, **kwargs):
        self.vlm_name = vlm_name
        self.load_vlm(*args, **kwargs)

    def load_vlm(self, *args, **kwargs):
         """ Loads the model and processor. """
         pass

    def evaluate_dataset(self, data_dicts, *args, **kwargs):
        """ 
        Evaluates the model on the given images and queries. 
        Returns a list of model response strings for each image-query in the dictionaries.
        """
        pass


    