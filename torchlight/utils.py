import argparse

class DictAction(argparse.Action):
    """
    Allow argparse arguments like:
    --arg key1=value1 key2=value2
    """
    def __call__(self, parser, namespace, values, option_string=None):
        result = {}
        for value in values:
            key, val = value.split('=')
            result[key] = val
        setattr(namespace, self.dest, result)
