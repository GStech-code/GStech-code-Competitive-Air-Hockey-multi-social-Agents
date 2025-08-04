_REGISTERED_POLICIES = {}
def register_policy(name=None):
    def decorator(cls):
        _REGISTERED_POLICIES[name or cls.__name__] = cls
        return cls
    return decorator

def get_team_policy(name):
    return _REGISTERED_POLICIES[name]