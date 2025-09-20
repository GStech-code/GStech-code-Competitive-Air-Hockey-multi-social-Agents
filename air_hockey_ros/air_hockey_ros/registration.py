_REGISTERED_POLICIES = {}
_REGISTERED_SIMULATIONS = {}

def register_policy(name=None):
    def decorator(cls):
        _REGISTERED_POLICIES[name or cls.__name__] = cls
        return cls
    return decorator
    
def register_simulation(name=None):
    def decorator(cls):
        _REGISTERED_SIMULATIONS[name or cls.__name__] = cls
        return cls
    return decorator

def get_team_policy(name):
    return _REGISTERED_POLICIES[name]

def get_simulation(name):
    return _REGISTERED_SIMULATIONS[name]