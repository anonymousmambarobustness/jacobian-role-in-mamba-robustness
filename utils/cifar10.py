import os
import re
if os.getenv('CONDA_DEFAULT_ENV') == "exploring_mamba_env7":
    from models.SSM import SSM, Mega, S5_SSM, S6_SSM
    from models.mamba_models import S6_Minimal_SSM, S6_with_activation, S6_SSM_auxiliary, S6_mixer, S6_params_activations
from vmamba_models.SS2D_SSM import SS2D_SSM
if os.getenv('CONDA_DEFAULT_ENV') == "exploring_mamba_env8":
    from models.mamba_models import RecurrentGemma, Griffin, Hawk, S6_Mamba2, ManualMamba2, ManualMamba2_with_activation

from vmamba_models.vmamba import VSSM
def build_model(args, model_name):
    original_model_name = model_name
    patch_size = int(model_name.split('_')[-1]) if "patch_size" in model_name else None
    model_name = model_name.split('_')[0]
    number_of_layers = re.findall(r'(\d+)_layers', original_model_name)
    activations = re.findall(r'with_(\w+)', original_model_name)
    auxiliary = ("auxiliary" in original_model_name)
    print(f"activations={activations} number_of_layers={number_of_layers}")
    num_layers = int(number_of_layers[0]) if len(number_of_layers) > 0 else args.num_layers
    activation_type = activations[0] if len(activations)>0 else "none"
    print(f"########## build_model: model_name={original_model_name} num_layers={num_layers} activation_type={activation_type} patch_size={patch_size}##########")
    if model_name == 'SSM' or original_model_name == "SSM_relu_AdS":
        if args.use_AdSS:
            print(f"$$$$$$$$$$$$$$$$$$$$$$ DEBUG: build_model: use_AdSS=TRUE $$$$$$$$$$$$$$$$$$$$$$")
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
    elif model_name == 'Mega':
        model = Mega(d_input=3, n_layers=args.num_layers, patch_size=patch_size)
    elif model_name == "SS2D":
        model = SS2D_SSM(d_input=3, n_layers=num_layers)
    elif "S6_params_activations" in original_model_name:
        params_activation_type = re.search(r'params_activations_(.*)', original_model_name).group(1)
        print(f"params_activation_type={params_activation_type}")
        model = S6_params_activations(d_input=3, n_layers=num_layers,use_fast_path=False,params_activation_type=params_activation_type)
    elif original_model_name == "S6_mixer":
        model = S6_mixer(d_input=3, n_layers=num_layers, patch_size=patch_size)
    elif model_name == 'S6' and auxiliary:
        print(f"DEBUG: build_model: {original_model_name} using S6_SSM_auxiliary class")
        model = S6_SSM_auxiliary(d_input=3, n_layers=num_layers, patch_size=patch_size)
    elif model_name == 'S6' and activation_type=="none":
        model = S6_SSM(d_input=3, n_layers=num_layers, reg_type=args.reg_type, seq_length=1024,use_AdSS=args.use_AdSS,AdSS_Type=args.AdSS_Type,patch_size=patch_size)
    elif model_name == 'S6' and activation_type!="none":
        model = S6_with_activation(d_input=3, n_layers=num_layers, reg_type=args.reg_type, seq_length=1024, activation=activation_type,patch_size=patch_size)
    elif original_model_name == 'S6_Minimal' and activation_type=="none":
        model = S6_SSM(d_input=3, n_layers=num_layers,reg_type=args.reg_type,patch_size=patch_size)
    elif model_name == 'ManualMamba2' and activation_type=="none":
        model = ManualMamba2(d_input=3, n_layers=num_layers, reg_type=args.reg_type, seq_length=1024, use_mem_eff_path=False,patch_size=patch_size) # inorder to use "attention_lip_bound" we need to get access to B,C,x
    elif model_name == 'ManualMamba2' and activation_type!="none":
        model = ManualMamba2_with_activation(d_input=3, n_layers=num_layers, reg_type=args.reg_type, seq_length=1024, use_mem_eff_path=False, activation=activation_type, patch_size=patch_size) # inorder to use "attention_lip_bound" we need to get access to B,C,x
    elif original_model_name == 'S6_Mamba2':
        model = S6_Mamba2(d_input=3, n_layers=num_layers, reg_type=args.reg_type, seq_length=1024)
    elif model_name == "RecurrentGemma":
        model = RecurrentGemma(d_input=3, n_layers=args.num_layers)
    elif model_name == "Griffin":
        model = Griffin(d_input=3, n_layers=num_layers)
    elif model_name == "Hawk":
        model = Hawk(d_input=3, n_layers=num_layers)
    elif model_name == "VMamba":
        model = VSSM(
            patch_size=4,
            in_chans=3,
            num_classes=10,
            depths=[2, 2, 5, 2],
            dims=96,
            # ===================
            ssm_d_state=1,
            ssm_ratio=2.0,
            ssm_rank_ratio=2.0,
            ssm_dt_rank="auto",  ####
            ssm_act_layer="silu",
            ssm_conv=3,
            ssm_conv_bias=False,
            ssm_drop_rate=0.0,
            ssm_init="v0",  #####
            forward_type="v3noz",
            # ===================
            mlp_ratio=4.0,
            mlp_act_layer="gelu",
            mlp_drop_rate=0.0,  ####
            # ===================
            drop_path_rate=0.2,
            patch_norm=True,  #####
            norm_layer="ln",  #####
            downsample_version="v3",
            patchembed_version="v2",  ###gmlp=false
            use_checkpoint=False,
        )

    return model