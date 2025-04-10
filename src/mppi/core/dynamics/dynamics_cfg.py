from isaaclab.utils import configclass

from dataclasses import MISSING

from .car_dynamics import SimpleCarDynamics, SimpleCarDynamicsNoAction

@configclass
class SimpleCarDynamicsCfg:
    
    class_type: type = SimpleCarDynamics

    x_dim: int = 14
    '''agent state dimension'''

    feat_dim: int | None = MISSING
    '''feature dimension. If None, no features are concatenated'''

    concatenate_feats: bool = MISSING
    '''concatenate features to rollout states'''

    wheelbase: float = MISSING
    '''wheelbase'''

    throttle_to_wheelspeed: float = MISSING
    '''throttle to wheelspeed'''

    steering_max: float = MISSING
    '''maximum steering angle'''

    dt: float = MISSING
    '''time step'''

@configclass
class SimpleCarDynamicsNoActionCfg(SimpleCarDynamicsCfg):
    '''
    Configuration class for SimpleCarDynamics
    '''
    class_type: type = SimpleCarDynamicsNoAction
    '''class type for the dynamics model'''

    x_dim: int = 12
    '''agent state dimension excludes action'''
