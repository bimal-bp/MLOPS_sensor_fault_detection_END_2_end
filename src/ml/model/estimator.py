class TargetValueMapping:
    def __init__(self):
        self.neg = 0
        self.pos = 1

    def reverse_mapping(self):
        mapping_response = self.__dict__
        return dict(zip(mapping_response.values(), mapping_response.keys()))
