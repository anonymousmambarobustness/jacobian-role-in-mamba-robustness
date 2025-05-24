import os
if os.getenv('CONDA_DEFAULT_ENV') == "exploring_mamba_env7":
    from models.SSM import SSM, Mega, S5_SSM, S6_SSM
from vmamba_models.SS2D_SSM import SS2D_SSM
import re
def build_model(args, model_name):
    number_of_layers = re.findall(r'(\d+)_layers', model_name)
    patch_size = int(model_name.split('_')[-1]) if "patch_size" in model_name else None
    num_layers = int(number_of_layers[0]) if len(number_of_layers) > 0 else args.num_layers
    if model_name == 'SSM' or model_name == "SSM_relu_AdS":
        if args.use_AdSS:
            print(f"$$$$$$$$$$$$$$$$$$$$$$ DEBUG: build_model: use_AdSS=TRUE $$$$$$$$$$$$$$$$$$$$$$")
            model = SSM(d_input=1, d_model=128, d_state=32, n_layers=args.num_layers, \
                use_AdSS=True,AdSS_Type=args.AdSS_Type,patch_size=patch_size)
        else:
            model = SSM(d_input=1, d_model=128, d_state=32, n_layers=args.num_layers,patch_size=patch_size)
    elif model_name == 'DSS':
        if args.use_AdSS:
            model = SSM(d_input=1, d_model=128, d_state=32, n_layers=args.num_layers, mode = 'diag', \
                use_AdSS=True,AdSS_Type=args.AdSS_Type,patch_size=patch_size)
        else:
            model = SSM(d_input=1, d_model=128, d_state=32, n_layers=args.num_layers, mode = 'diag', patch_size=patch_size)
    elif model_name == 'S5':
        model = S5_SSM(d_input=1, d_model=128, d_state=32, n_layers=args.num_layers, patch_size=patch_size)
    elif model_name == 'Mega':
        model = Mega(d_input=1, d_model=128, hidden_dim=32, n_layers=args.num_layers, seq_len=28*28, patch_size=patch_size)
    elif model_name == 'S6':
        model = S6_SSM(d_input=1, d_model=128, d_state=32, n_layers=args.num_layers, reg_type=args.reg_type, seq_length=28*28, patch_size=patch_size)
    elif model_name in ["SS2D","SS2D_1_layers"]:
        model = SS2D_SSM(d_input=1, d_model=128, d_state=32, n_layers=num_layers)

    return model