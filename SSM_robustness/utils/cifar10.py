import os
import re
from models.SSM import SSM, S5_SSM, S6_SSM
def build_model(args, model_name):
    original_model_name = model_name
    patch_size = int(model_name.split('_')[-1]) if "patch_size" in model_name else None
    model_name = model_name.split('_')[0]
    number_of_layers = re.findall(r'(\d+)_layers', original_model_name)
    auxiliary = ("auxiliary" in original_model_name)
    num_layers = int(number_of_layers[0]) if len(number_of_layers) > 0 else args.num_layers
    print(f"########## build_model: model_name={original_model_name} num_layers={num_layers} patch_size={patch_size}##########")
    if model_name == 'SSM' or original_model_name == "SSM_relu_AdS":
        if args.use_AdSS:
            model = SSM(d_input=3, n_layers=num_layers, \
                use_AdSS=True,AdSS_Type=args.AdSS_Type,patch_size=patch_size)
        else:
            model = SSM(d_input=3, n_layers=num_layers,patch_size=patch_size)

    elif model_name == 'DSS' or original_model_name == "DSS_relu_AdS":
        if args.use_AdSS:
            model = SSM(d_input=3, n_layers=args.num_layers, mode = 'diag', \
                use_AdSS=True,AdSS_Type=args.AdSS_Type,patch_size=patch_size)
        else:
            model = SSM(d_input=3, n_layers=args.num_layers, mode = 'diag', patch_size=patch_size)
    elif model_name == 'S5':
        model = S5_SSM(d_input=3, n_layers=args.num_layers, patch_size=patch_size)
    elif model_name == 'S6':
        model = S6_SSM(d_input=3, n_layers=num_layers, reg_type=args.reg_type, seq_length=1024,use_AdSS=args.use_AdSS,AdSS_Type=args.AdSS_Type,patch_size=patch_size)
    return model
