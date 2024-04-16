import os
import sys
from tkinter import filedialog

#The config file is expected to contain two key value pairs tab separated
#One value aspen_path should specify the path to AspenModeler.exe.
#One value project_path should specify the path to the root directory in
#which the simulations have to be performed.

"""
The script prepares a template ada file using an accessory file
'selected_variables.txt' where the lines that define the variables' values
are read and substituted in the ada file with a formattable string that
will be edited by make_and_run_sims.py and launched for simulations.
"""

config = {}
with open('config.cfg') as cfg:
    for line in cfg.readlines():
        line = line.strip().split('\t')
        config[line[0]] = line[1]

projectPath = os.path.abspath(filedialog.askdirectory(initialdir=config['project_path']))

projectPath = os.path.abspath(filedialog.askdirectory(initialdir=projectPath))

if not projectPath:
    print('No project selected')
    sys.exit()

template_path = os.path.abspath(os.path.join(projectPath, 'template'))

for t in os.listdir(template_path):
    if '_TEMPLATE.ada' in t:
        continue
    elif os.path.isdir(os.path.abspath(os.path.join(template_path, t))):
        continue
    elif t == 'selected_variables.txt':
        with open(os.path.join(template_path, t)) as f:
            sel_var = f.readlines()
            print(sel_var)
    elif 'template' not in t:
        template_file = t
        with open(os.path.join(template_path, t)) as f:
            base_template = f.read()
    else:
        continue

for var in sel_var:
    var = var.replace('\n', '')
    if var:
        var = var.split('\t')
        base_template = base_template.replace(var[0], var[1])

with open(os.path.join(template_path, template_file.replace('.ada', '_TEMPLATE.ada')), 'w') as out:
    out.write(base_template)