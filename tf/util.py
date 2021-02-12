import models
import yaml
import os
import glob
import shutil
import random


def get_config_file_name(agent_uuid: int, config_dir):
    return config_dir+"agent_"+str(agent_uuid)+".yaml"

def delete_agent_config_files(config_dir: str):
    fileList = glob.glob(config_dir + "agent_*.yaml")
    # Iterate over the list of filepaths & remove each file.
    for filePath in fileList:
        try:
            os.remove(filePath)
        except:
            print("Error while deleting file : ", filePath)


def create_config_file(agent_id: int, config_dir, pbt_input_base_dir):
    agent = models.get_agent(agent_id)
    evolution = models.get_evolution(agent["evolution"])
    evolution_size = int(evolution["steps_stop"] - evolution["steps_start"])
    network = models.get_network(evolution["network"])
    with open(config_dir + "base_agent_config.yaml") as base_config:
        cfg = yaml.safe_load(base_config)

        # Write config
        with open(get_config_file_name(agent["uuid"], config_dir), 'w') as config_file:
            name = str(network["name"]) + "_" + str(agent["uuid"])

            cfg['gpu'] = int(agent['gpu'])
            cfg['name'] = str(name)
            cfg['training']['lr_values'] = [float(agent["lr_values"])]
            cfg['training']['lr_boundaries'] = []
            cfg['training']['total_steps'] = evolution_size
            cfg['training']['checkpoint_steps'] = evolution_size
            cfg['training']['test_steps'] = int(1000)


            cfg['model']['filters'] = int(network["filters"])
            cfg['model']['policy_channels'] = int(network["policy_channels"])
            cfg['model']['residual_blocks'] = int(network["residual_blocks"])
            cfg['model']['se_ratio'] = int(network["se_ratio"])

            cfg['dataset']['input_train'] = get_pbt_train_path(evolution["iteration"], pbt_input_base_dir)
            cfg['dataset']['input_test'] = get_pbt_test_path(evolution["iteration"], pbt_input_base_dir)
            cfg['dataset']['num_chunks'] = evolution_size * 25
            cfg['dataset']['train_ratio'] = float(20/25)

            # cfg['training']['policy_loss_weight'] = policy_loss_weight
            # cfg['training']['value_loss_weight'] = value_loss_weight
            yaml.dump(cfg, config_file)


def get_base_network_path(agent_id: int, network_dir, name, steps: int) -> str:
    agent = models.get_agent(agent_id)
    return network_dir + name + "_" + str(agent["uuid"]) + "/" + name + "_" + str(agent["uuid"]) + "-" + str(steps)

def get_weights_file(agent_id: int, network_dir, name, steps: int) -> str: 
    return get_base_network_path(agent_id, network_dir, name,steps) + ".pb.gz"

def get_weights_index_file(agent_id: int, network_dir, name, steps: int) -> str:
    return get_base_network_path(agent_id, network_dir, name,steps) + ".index"

def get_weights_data_file(agent_id: int, network_dir, name, steps: int) -> str:
    return get_base_network_path(agent_id, network_dir, name,steps) + ".data-00000-of-00001"

def get_pbt_train_path(iteration: int, base_dir: str):
    return base_dir + get_pbt_train_folder(iteration)

def get_pbt_train_folder(iteration: int):
    return "evo_" + str(iteration) + "/train/"

def get_pbt_test_path(iteration: int, base_dir: str):
    return base_dir + get_pbt_test_folder(iteration)

def get_pbt_test_folder(iteration: int):
    return "evo_" + str(iteration) + "/test/"

def prepare_training_data(evolution_num: int, evolution_size: int, train_data_dir: str, test_data_dir: str, output_dir: str):
    print("Preparing data for " + str(evolution_num) + " evolutions...")
    
    try:
        shutil.rmtree(output_dir)
    except FileNotFoundError:
        print("No previous data found")

    for evolution in range(evolution_num): 
        # current_evolution = int(evolution) + int(1)
        train_path = os.path.join(output_dir, get_pbt_train_folder(evolution))
        # try:
            # os.makedirs(train_path)
        # except FileExistsError:
            # shutil.rmtree(train_path)
            # os.makedirs(train_path)
        os.makedirs(train_path)

        test_path = os.path.join(output_dir, get_pbt_test_folder(evolution))
        # try:
            # os.makedirs(test_path)
        # except FileExistsError:
            # shutil.rmtree(test_path)
            # os.makedirs(test_path)
        os.makedirs(test_path)

        ## Train data
        train_file_list = _get_file_list(train_data_dir)
        train_random_files = _get_random_files(train_file_list, evolution_size*20)
        _copy_files(train_random_files, train_data_dir, train_path)

        ## Train data
        test_file_list = _get_file_list(test_data_dir)
        test_random_files = _get_random_files(test_file_list, evolution_size*5)
        _copy_files(test_random_files, test_data_dir, test_path)

        print("Evolution " + str(evolution) + " done")

def _get_file_list(input_dir):
    return [file for file in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, file))]

def _get_random_files(file_list, N):
    return random.sample(file_list, N)

def _copy_files(random_files, input_dir, output_dir):
    for file in random_files:
        shutil.copy(os.path.join(input_dir, file), output_dir)
