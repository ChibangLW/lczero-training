import sqlite3
import time


def get_db_connection():
    # try:
    return sqlite3.connect('pbt.db')
    # except Error:
    # print(Error)


def dict_factory(cursor, row):
    d = {}
    for idx, col in enumerate(cursor.description):
        d[col[0]] = row[idx]
    return d


def setup_database():
    c = get_db_connection()
    # c.execute("DROP TABLE networks")
    c.execute(
        "CREATE TABLE IF NOT EXISTS networks (id INTEGER PRIMARY KEY AUTOINCREMENT, name text UNIQUE, filters int, residual_blocks int, se_ratio int, policy_channels int)")
    # c.execute("DROP TABLE evolutions")
    c.execute(
        "CREATE TABLE IF NOT EXISTS evolutions (id INTEGER PRIMARY KEY AUTOINCREMENT, network INTEGER, active int, iteration int, created_at int, completed_at, started_at_train int, finished_at_train int, started_at_games int, finished_at_games int, started_at_update int, finished_at_update int, steps_start int, steps_stop int)"
    )
    # c.execute("DROP TABLE population")
    c.execute(
        "CREATE TABLE IF NOT EXISTS population (id INTEGER PRIMARY KEY AUTOINCREMENT, uuid int, evolution int, name text, active int, lr_values text, lr_boundaries text, gpu int)")
    # c.execute("DROP TABLE matches")
    c.execute(
        "CREATE TABLE IF NOT EXISTS matches (id INTEGER PRIMARY KEY AUTOINCREMENT, opponent1 int, opponent2 int, win_rate REAL, wins1 int, defeats1 int, draws1 int, wins1w int, defeats1w int, draws1w int, wins1b int, defeats1b int, draws1b int, evolution INTEGER )"
    )
    c.commit()


def create_network(name: str, filters: int, residual_blocks: int, se_ratio, policy_channels: int) -> int:
    print("Creating network...")

    conn = get_db_connection()
    cur = conn.cursor()
    # Try to create a new network. Will fail if name is already present in db
    try:
        cur.execute("INSERT INTO networks (name, filters, residual_blocks, se_ratio, policy_channels) VALUES (?, ?, ?, ?, ?)",
                    (name, filters, residual_blocks, se_ratio, policy_channels,))
        conn.commit()
        conn.close()
        return cur.lastrowid

    except sqlite3.IntegrityError:
        print('No new network created')
    conn.commit()
    conn.close()
    return -1

def get_network_by_name(name: str):
    conn = get_db_connection()
    conn.row_factory = dict_factory

    c = conn.cursor()
    population = c.execute(
        "SELECT * FROM networks WHERE name = ?", (name,)).fetchone()
    conn.close()

    return population
    
def get_network(network_id: int):
    conn = get_db_connection()
    conn.row_factory = dict_factory

    c = conn.cursor()
    population = c.execute(
        "SELECT * FROM networks WHERE id = ?", (network_id,)).fetchone()
    conn.close()

    return population

def create_agent(evolution_id: int, uuid: int, lr: int, gpu: int) -> int:
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("INSERT INTO population (evolution, lr_values, lr_boundaries, uuid, gpu) VALUES (?, ?, ?, ?, ?)",
                (evolution_id, ','.join(map(str, [lr])), ','.join(map(str, [])), uuid, gpu,))
    conn.commit()
    conn.close()
    return cur.lastrowid


def get_population_size(evolution_id: int) -> int:
    conn = get_db_connection()
    cur = conn.cursor()
    population_database_size = cur.execute(
        "SELECT count(id) FROM population WHERE evolution = ?", (str(evolution_id),)).fetchone()[0]
    conn.close()
    return population_database_size


def get_population(evolution_id: int):
    conn = get_db_connection()
    conn.row_factory = dict_factory

    c = conn.cursor()
    population = c.execute(
        "SELECT * FROM population WHERE evolution = ?", (str(evolution_id),)).fetchall()
    conn.close()

    return population


def create_match(op_a: int, op_b: int, wins1, defeats1, draws1, wins1w, defeats1w, draws1w, wins1b, defeats1b, draws1b, win_rate, evolution_id: int):
    conn = get_db_connection()
    cur = conn.cursor()
    match_id = -1
    try:
        cur.execute("INSERT INTO matches (opponent1, opponent2, wins1, defeats1, draws1, wins1w, defeats1w, draws1w, wins1b, defeats1b, draws1b, win_rate, evolution) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (op_a, op_b, wins1, defeats1, draws1, wins1w, defeats1w, draws1w, wins1b, defeats1b, draws1b, win_rate, evolution_id,))
        match_id = cur.lastrowid

    except sqlite3.IntegrityError:
        print('Match not saved')

    conn.commit()
    conn.close()
    return match_id


def get_matches(evolution_id: int):
    conn = get_db_connection()
    conn.row_factory = dict_factory
    c = conn.cursor()
    matches = c.execute("SELECT * FROM matches WHERE evolution = ?",
                        (str(evolution_id),)).fetchall()
    conn.close()
    return matches


def create_evolution(network_id: int, iteration: int, steps_start: int, steps_stop: int) -> int:
    print("Creating evolution...")
    conn = get_db_connection()
    cur = conn.cursor()

    evolution = cur.execute(
        "SELECT * FROM evolutions WHERE network = ? AND iteration = ?", (str(network_id), str(iteration))).fetchone()

    if evolution == None:
        # Try to create a new evolution. Will fail if name is already present in db
        try:
            cur.execute("INSERT INTO evolutions (network, active, iteration, created_at, steps_start, steps_stop) VALUES (?, ?, ?, ?,? ,?)",
                        (network_id, 1, iteration, int(time.time()), steps_start, steps_stop))
            conn.commit()
            conn.close()
            return cur.lastrowid

        except sqlite3.IntegrityError:
            print('No new evolution created')
    else:
        print('No new evolution created')

    conn.commit()
    conn.close()
    return evolution[0]

def get_evolution(evolution_id: int):
    conn = get_db_connection()
    conn.row_factory = dict_factory

    evolution = conn.execute(
        "SELECT * FROM evolutions WHERE id = ?", (str(evolution_id),)).fetchone()

    conn.close()
    return evolution

def get_last_evolution(network_id: int):
    conn = get_db_connection()
    conn.row_factory = dict_factory

    evolution = conn.execute(
        "SELECT * FROM evolutions WHERE network = ? ORDER BY iteration DESC LIMIT 1", (str(network_id),)).fetchone()

    conn.close()
    return evolution

def get_agent(agent_id: int):
    conn = get_db_connection()
    conn.row_factory = dict_factory

    agent = conn.execute("SELECT * FROM population WHERE id = ?", (str(agent_id),)).fetchone()

    conn.close()
    return agent

def get_agents_by_gpu(evolution_id: int, gpu: int):
    conn = get_db_connection()
    conn.row_factory = dict_factory

    agent = conn.execute(
        "SELECT * FROM population WHERE evolution = ? AND gpu = ?", (str(evolution_id),str(gpu),)).fetchall()

    conn.close()
    return agent

def finish_evolution(evolution_id):
    conn = get_db_connection()
    conn.row_factory = dict_factory
    c = conn.cursor()

    # Finish old evolution
    c.execute("UPDATE evolutions SET completed_at = ?, active = 0 WHERE id = ?",
              (int(time.time()), str(evolution_id)))

    # Get finished evolution
    old_evolution = c.execute("SELECT * FROM evolutions WHERE id = ?", (str(evolution_id),)).fetchone()

    conn.commit()
    conn.close()
    return old_evolution

def update_evolution_time(evolution_id: int, field):
    conn = get_db_connection()
    conn.row_factory = dict_factory
    c = conn.cursor()

    # Finish old evolution
    c.execute("UPDATE evolutions SET " + str(field) + " = ? WHERE id = ?",
              (int(time.time()), str(evolution_id)))

    # Get finished evolution
    old_evolution = c.execute(
        "SELECT * FROM evolutions WHERE id = ?", (str(evolution_id),)).fetchone()

    conn.commit()
    conn.close()
    return old_evolution
