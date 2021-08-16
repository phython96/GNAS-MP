import os
from tensorboardX import SummaryWriter

class record_run:
    
    def __init__(self, comment = ''):
        self.comment = comment


    def __call__(self, func):
        
        def wrapped_func(*args, **kwargs):
            writer    = args[0].writer
            # geno_path = args[0].args.load_genotypes
            i_epoch   = args[1]
            stage     = args[-1]

            result = func(*args, **kwargs)
            writer.add_scalar(f'{stage}_loss', result['loss'], global_step = i_epoch)
            writer.add_scalar(f'{stage}_metric', result['metric'], global_step = i_epoch)
        
            return result

        return wrapped_func