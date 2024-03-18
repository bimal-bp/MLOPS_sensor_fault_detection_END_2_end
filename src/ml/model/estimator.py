class TargetValueMapping:
    def __init__(self):
        self.neg = 0
        self.pos = 1

    def reverse_mapping(self):
        mapping_response = self.__dict__
        return dict(zip(mapping_response.values(), mapping_response.keys()))

class SensorModel:
    def __init__(self,preprocessor,model):
        try:
            self.preprocessor = preprocessor 
            self.model = model 
        except Exception as e:
            raise e

    def predict(self,x):
        try:
            x_transform = self.preprocessor.transform(x)
            y_hat = self.model.predict(x_transform)
            return y_hat
        except Exception as e:
            raise e 