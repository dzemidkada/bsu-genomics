import yaml

class Config:
    def __init__(self, path=None):
        self._dict = {}
        with open(path, 'r') as stream:
            try:
                self._dict = yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                print(exc)
    
    def __getitem__(self, key):
        return self._dict[key]