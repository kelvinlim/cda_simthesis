#! /usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import os
import glob
import sys
import json
from pathlib import Path
import textwrap
from glob import glob


#from dotenv import dotenv_values  # pip install python-dotenv
import yaml # pip install pyyaml

import math
from picause import StructuralEquationDagModel
import os

"""



"""

__version_info__ = ('0', '1', '1')
__version__ = '.'.join(__version_info__)

version_history = \
"""
0.1.1 - changed naming of output files
0.1.0 - initial version  
"""
    


class SimData:
    
    def __init__(self, **kwargs):
        
        # load self.config
        self.config = {}
        for key, value in kwargs.items():
            self.config[key] = value

        # read in .env file
        # if 'env' in self.config:
        #     self.config.update(dotenv_values(self.config['env']))
        
        # read in yaml config file
        if 'config' in self.config:
            with open(self.config['config'], 'r') as f:
                yaml_config = yaml.safe_load(f)
            self.config.update(yaml_config)
            
        pass

    def version(self):
        """
        Return the version of the program.
        """

        return self.__version__
    
    def cmd(self, command):
        """
        Execute the command.
        """

        if command == 'sim':
            self.sim()
        else:
            print(f"Unknown command: {command}")
            sys.exit(1)
            
    def sim(self):
        """
        Perform simulation using parameters in config.yaml
        """
        sim_config = self.config.get('simulation', {})
        num_iterations = sim_config.get('iterations', 1)
        num_vars = sim_config.get('num_variables', 15)
        num_edges = sim_config.get('num_edges', 27)

        # read in the lists
        num_samples_list = sim_config.get('num_samples', [ 10000 ])
        effect_size_list = sim_config.get('effect_sizes', [ 0.1, 0.2, 0.3, 0.4 ])

        data_dir = sim_config.get('data_directory', 'sim_data/')
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)

        # try with different number of data points
        for num_data_points in num_samples_list:
            # try with a different effect sizes
            for effect_size in effect_size_list:
                for iter in range(num_iterations):
                    df, sem, count = self.create_single_df(effect_size=effect_size,
                                            num_data_points=num_data_points,
                                            num_vars=num_vars,
                                            num_edges=num_edges,
                                            seed=sim_config.get('seed', None))
                    # print(f"effect_size: {effect_size} count: {count}")
                    
                    # create the filename based es and iteration
                    # filename = f"rows-{num_data_points}_vars-{num_vars}_edges-{num_edges}_es-{effect_size}_iter-{iter:03d}.csv"
                    filename = f"sub-{iter+1:03d}_vars-{num_vars}_edges-{num_edges}_es-{effect_size}.csv"

                    # save the df to a csv file
                    path = os.path.join(data_dir, filename)
                    df.to_csv(path, index=False)
                    
                    # save graph description to a txt file
                    sem_file = filename.replace('.csv', '.txt')
                    sem_path = os.path.join(data_dir, sem_file)
                    with open(sem_path, 'w') as f:
                        f.write(sem.__str__())
                        # add the count of iterations needed for convergence
                        f.write(f"\nIterations: {count}\n")
                    
                    print(f"{filename} count: {count}")
                    pass
    
    def create_single_df(self, effect_size=0.1, seed=None, num_vars=14, num_edges=12,
                        num_data_points=100):
        """
        create a single instance of SEM DAG model. It checks
        if the residual overflow condition is violated and if so, 
        it creates a new instance until the condition is met.
        

        Args:
            effect_size (float, optional): _description_. Defaults to 0.1.
            seed (_type_, optional): _description_. Defaults to None.
            num_vars (_type_, optional): _description_. Defaults to 15.
            num_edges (_type_, optional): _description_. Defaults to 27.
            num_data_points (_type_, optional): _description_. Defaults to 100.

        Returns:
            sem: the SEM
            count: the number of iterations needed
        """
        
        beta = math.sqrt(effect_size)
        
        count = 1
        
        sem = StructuralEquationDagModel(num_var=num_vars, 
                                        num_edges=num_edges, 
                                        seed=seed, 
                                        beta=beta)
        # check if the residual overflow condition is violated - want False
        condition_violated = sem.test_residual_overflow()

        # loop until condition_violated is False
        while condition_violated:
            sem = StructuralEquationDagModel(num_var=num_vars, 
                                            num_edges=num_edges, 
                                            seed=seed, 
                                            beta=beta)
            condition_violated = sem.test_residual_overflow()
            count += 1
            
        # create the dataset 
        df = sem.generate_data(num_data_points)
        return df, sem, count
    
if __name__ == "__main__":
    
    # provide a description of the program with format control
    description = textwrap.dedent('''\
    Compute idea density of provided text files.
    
 
    ''')
    
    parser = argparse.ArgumentParser(
        description=description, formatter_class=argparse.RawTextHelpFormatter)

    parser.add_argument("--config", type = str,
                     help="name of yaml config file in the current directory, default config.yaml",
                      default="config.yaml") 
        
    parser.add_argument("--cmd", type = str,
                    help="cmd - [sim], default sim",
                    default = 'sim')

    parser.add_argument("-H", "--history", action="store_true", help="Show program history")
     
    # parser.add_argument("--quiet", help="Don't output results to console, default false",
    #                     default=False, action = "store_true")  
    
    parser.add_argument("--verbose", type=int, help="verbose level default 2",
                         default=2) 
        
    parser.add_argument('-V', '--version', action='version', version=f'%(prog)s {__version__}')

    args = parser.parse_args()

    if args.history:
        print(f"{os.path.basename(__file__) } Version: {__version__}")
        print(version_history)
        exit(0)

    obj = SimData(  config=args.config,

                    )
    obj.cmd(args.cmd)
    
    pass

