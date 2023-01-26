import nest
from itertools import product

nest.ResetKernel()
nest.SetKernelStatus({'local_num_threads': 4})  # Adapt if necessary

nest.print_time = False
#nest.overwrite_files = True

##########################################
## Define parameter ranges to be tested
## Example procedure:
a = [1,2,3,4]
b = [5,6,7,8]
c = [9,10,11,12]
lst = [a,b,c]

## Create list containing all conditions to be tested
conditions = product(*lst)

## Simulate for values of condition
for c in conditions:
    pass
    ## Pass values
    
    ## create network
    
    ## run simulation
    
    ## Get output from and analyze condition

    ## Save output of condition

    ## Repeat for all combinations of values

