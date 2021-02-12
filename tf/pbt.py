import os

import yaml
import argparse
import random
import re
import subprocess
import models
import util
import shutil
import time
from multiprocessing import Process, Pool
from functools import partial


config_dir = ""
pbt_config_file = ""
network_dir = ""
pbt_input_base_dir = ""
lc0_path = ""
name = ""
evolution_size = 0


def get_epsilon(current_lr: float) -> float:
    return random.uniform(-current_lr * 0.2, current_lr * 0.2)


def train_on_gpu(gpu_id, evolution_id): 
    for agent in models.get_agents_by_gpu(evolution_id, gpu_id):
        print("Training agent ",agent, " on gpu ", gpu_id)
        config_file = util.get_config_file_name(agent["uuid"], config_dir)
        os.system("python train.py --cfg " + config_file)


def train(evolution_id: int, gpus):
    print("Training the population...")
    models.update_evolution_time(evolution_id, "started_at_train")

    with Pool(len(gpus)) as pool:    
        pool.map(partial(train_on_gpu, evolution_id=evolution_id), gpus)
        
        pool.close()
        pool.join()

    models.update_evolution_time(evolution_id, "finished_at_train")



def play_matches(evolution_id: int, games: int):
    # play
    # ./build/release/lc0 selfplay --player1.weights=networks/dla2-64x6_1/dla2-64x6_1-5000 --player2.weights=networks/cdla2-64x6_2/dla2-64x6_2-5000 --games=50 --movetime=500000
    # save_match(1, 2, evolution_id, "tournamentstatus final P1: +1184 -15 =21 Win: 54.00% Elo: -6.95 LOS: 42.63% P1-W: +9 -3 =13 P1-B: +5 -12 =8 npm 1.000000 nodes 5615 moves 5615")

    print("Playing all validation matches...")

    models.update_evolution_time(evolution_id, "started_at_games")

    population = models.get_population(evolution_id)
    evolution = models.get_evolution(evolution_id)

    for i in range(0, len(population) - 1):
        for j in range(i + 1, len(population)):
            op_a = population[i]["id"]
            op_b = population[j]["id"]

            weights_a = util.get_weights_file(
                op_a, network_dir, name, evolution["steps_stop"])
            weights_b = util.get_weights_file(
                op_b, network_dir, name, evolution["steps_stop"])

            args = [lc0_path+"lc0", "selfplay", "--player1.weights="+weights_a, "--player2.weights=" + weights_b, "--games=" + str(games), "--movetime=500000"] 
            process = subprocess.Popen(args, stdout=subprocess.PIPE)
           

            stdout = process.communicate()[0]
            result_out = stdout.splitlines()

            save_match(op_a, op_b, evolution_id, result_out[len(result_out) - 1].decode('UTF-8'))
    
    models.update_evolution_time(evolution_id, "finished_at_games")


def init_population(evolution_id: int, model_name: str, population_size: int, gpus):
    print("Initiating population...")
    population_database_size = models.get_population_size(evolution_id)

    

    if evolution_id != -1 and population_database_size < population_size:

        for i in range(population_size - population_database_size):
            init_lr = random.uniform(0, 0.2)
            agent_id = models.create_agent(evolution_id, i, init_lr, gpus[i%len(gpus)])


            util.create_config_file(agent_id, config_dir, pbt_input_base_dir)
        return

    print("Population for Evolution " + str(evolution_id) + " already created")


# returns the ID of the match
def save_match(opponentA: int, opponentB: int, evolution_id: int, result: str) -> int:
    # Final return value of lc0
    # tournamentstatus final P1: +14 -15 =21 Win: 49.00% Elo: -6.95 LOS: 42.63% P1-W: +9 -3 =13 P1-B: +5 -12 =8 npm 1.000000 nodes 5615 moves 5615

    # match all
    match = re.findall("((\+|-|=)(\d+) )", result)
    # Get WIN and LOS percentage
    percentage = re.findall("(\d+.\d{2})%", result)

    match_id = models.create_match(opponentA, opponentB, match[0][2], match[1][2], match[2][2], match[3][2],
                                   match[4][2], match[5][2], match[6][2], match[7][2], match[8][2], percentage[0],
                                   evolution_id)

    return match_id


def build_ranking(evolution_id: int):
    # Build ranking of best and worst performing agents
    print("Ranking evo " + str(evolution_id) + "...")
    models.update_evolution_time(evolution_id, "started_at_update")

    population = models.get_population(evolution_id)

    # print(population)
    ranking = {}
    for agent in population:
        ranking[agent["id"]] = 0
    for match in models.get_matches(evolution_id):
        # print(match)
        if match["win_rate"] < 50.0:
            ranking[match["opponent2"]] += 3
            # print("verloren")
        elif match["win_rate"] > 50.0:
            ranking[match["opponent1"]] += 3
            # print("gewonnen")
        elif match["win_rate"] == 50.0:
            ranking[match["opponent1"]] += 1
            ranking[match["opponent2"]] += 1
            # print("unentschieden")

    sorted_ranking = sorted(ranking.items(), key=lambda x: x[1], reverse=True)
    # print(sorted_ranking)
    if len(sorted_ranking) > 0:
        print("Winner: " + str(sorted_ranking[0][0]))
    return sorted_ranking, sorted_ranking[0][0]


def update_population(population_sorted: list, evolution_id: int, gpus) -> int:
    print("Updating population...")
    print("from evo " + str(evolution_id))

    # Get finished evolution
    models.update_evolution_time(evolution_id, "finished_at_update")
    old_evolution = models.finish_evolution(evolution_id)

    new_evo_id = -1
    if old_evolution != None:
        # Create new evolution
        iteration = int(old_evolution['iteration']) + 1
        new_evo_id = models.create_evolution(old_evolution['network'], iteration, iteration * evolution_size,
                                             (iteration + 1) * evolution_size)
    else:
        print("not found")


    for i in range(int(len(population_sorted) / 2)):
        print(i)
        agent_good = models.get_agent(population_sorted[i][0])
        agent_bad = models.get_agent(population_sorted[len(population_sorted) - 1 - i][0])

        print(agent_good)
        print(agent_bad)

        epsilon = get_epsilon(float(agent_good["lr_values"]))

        new_lr = float(agent_good["lr_values"]) + epsilon

        new_agent_good_id = models.create_agent(
            new_evo_id, agent_good["uuid"], float(agent_good["lr_values"]), gpus[agent_good["uuid"] % len(gpus)])

        new_agent_bad_id = models.create_agent(
            new_evo_id, agent_bad["uuid"], new_lr, gpus[agent_good["uuid"] % len(gpus)])

        copy_weights(new_agent_good_id, new_agent_bad_id, new_evo_id)

        util.create_config_file(new_agent_bad_id, config_dir, pbt_input_base_dir)
        util.create_config_file(new_agent_good_id, config_dir, pbt_input_base_dir)

    return new_evo_id


def copy_weights(agent_good_id: int, agent_bad_id: int, evolution_id: int):
    evolution = models.get_evolution(evolution_id)
    print("Coping weights...")
    src = util.get_weights_file(
        agent_good_id, network_dir, name, evolution["steps_start"])
    dest = util.get_weights_file(
        agent_bad_id, network_dir, name, evolution["steps_start"])
    shutil.copyfile(src, dest)

    src = util.get_weights_index_file(
        agent_good_id, network_dir, name, evolution["steps_start"])
    dest = util.get_weights_index_file(
        agent_bad_id, network_dir, name, evolution["steps_start"])
    shutil.copyfile(src, dest)

    src = util.get_weights_data_file(
        agent_good_id, network_dir, name, evolution["steps_start"])
    dest = util.get_weights_data_file(
        agent_bad_id, network_dir, name, evolution["steps_start"])
    shutil.copyfile(src, dest)


def main(cmd):
    global pbt_config_file
    pbt_config_file = cmd.cfg.read()
    cfg = yaml.safe_load(pbt_config_file)
    print(yaml.dump(cfg, default_flow_style=False))

    if cmd.setup:
        models.setup_database()

    # PBT details
    evolutions = cfg["pbt"]["evolutions"]
    population_size = cfg["pbt"]["population_size"]
    global config_dir
    config_dir = cfg["pbt"]["config_directory"]
    global evolution_size
    evolution_size = cfg["pbt"]["evolution_size"]

    global network_dir
    network_dir = cfg["pbt"]["network_path"]

    global lc0_path
    lc0_path = cfg["pbt"]["lc0_path"]

    global pbt_input_base_dir
    pbt_input_base_dir = cfg["pbt"]["input_base"]

    evaluation_games = int(cfg["pbt"]["evaluation_games"])
    gpus = cfg['gpus']

    # Model details
    filters = cfg['model']['filters']
    residual_blocks = cfg['model']['residual_blocks']
    se_ratio = cfg['model']['se_ratio']
    policy_channels = cfg['model']['policy_channels']

    # append model settings to create name of current network
    prefix = cfg['prefix']
    global name
    name = prefix + "-" + str(filters) + "x" + str(residual_blocks)


    # Preparing data to usable size because otherwise each evolution the whole dataset (15GB, 2500000 Games) would be loaded. This performance increase
    if cmd.prep_data:
        util.prepare_training_data(evolutions, evolution_size,
                           cfg["dataset"]["input_train"], cfg["dataset"]["input_test"], pbt_input_base_dir)

    # Create a new Network configuration if the name is different than an already existing network
    network_id = models.create_network(name, filters, residual_blocks, se_ratio, policy_channels)

    


    # Cleanup old agent configs
    if network_id != -1:
        util.delete_agent_config_files(config_dir)
    else:
        network_id = models.get_network_by_name(name)["id"]

    evolution = models.get_last_evolution(network_id)
    
    if evolution is None:
        evolution_id = models.create_evolution(network_id, 0, 0 * evolution_size, 1 * evolution_size)
    else:
        evolution_id = evolution['id']

    # Create the initial population for the created evolution
    init_population(evolution_id, name, population_size, gpus)
    best_agent_id = -1

    for _ in range(evolutions):
        # Train the population
        train(evolution_id, gpus)

        # Play all the matches to evaluate strength of population
        play_matches(evolution_id, evaluation_games)

        # Build the ranking to update population
        population, best_agent_id = build_ranking(evolution_id)

        # Update population. Exploitation: Stick to top performer. Exploration: Copy values of top to loosers and variate them by factor X
        evolution_id = update_population(population, evolution_id, gpus)


    agent = models.get_agent(best_agent_id)
    print("Best agent was ", agent["uuid"])



if __name__ == "__main__":
    argparser = argparse.ArgumentParser(
        description='Pipeline to enable PBT for tensorflow training')
    argparser.add_argument('--cfg',
                           type=argparse.FileType('r'),
                           help='yaml configuration with pbt parameters')
    argparser.add_argument('--setup',
                           type=bool,
                           help='setup database')
    argparser.add_argument('--prep_data',
                           type=bool,
                           help='prepare training data')

    # mp.set_start_method('spawn')
    main(argparser.parse_args())
    # mp.freeze_support()
