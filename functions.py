"""
This file may not be shared/redistributed freely. Please read copyright notice in the git repo.
"""
import sys
import collections
import inspect
import types
import pandas as pd
import numpy as np
import shutil
import inspect
import os
import compress_pickle
import itertools
from tqdm import tqdm
from gym.envs.toy_text.discrete import DiscreteEnv
import warnings
from collections import OrderedDict
import glob
import csv
import json
import time
from datetime import datetime

color2num = dict(
    gray=30,
    red=31,
    green=32,
    yellow=33,
    blue=34,
    magenta=35,
    cyan=36,
    white=37,
    crimson=38)

def colorize(string, color, bold=False, highlight=False):
    attr = []
    num = color2num[color]
    if highlight: num += 10
    attr.append(str(num))
    if bold: attr.append('1')
    return '\x1b[%sm%s\x1b[0m' % (';'.join(attr), string)

def plot_data(data, y="accumulated_reward", x="Episode", ci=95, estimator='mean', **kwargs):
    import seaborn as sns
    import matplotlib.pyplot as plt
    if isinstance(data, list): # is this correct even?
        data = pd.concat(data, ignore_index=True,axis=0)
    plt.figure(figsize=(12, 6))
    sns.set(style="darkgrid", font_scale=1.5)
    lp = sns.lineplot(data=data, x=x, y=y, hue="Condition", ci=ci, estimator=estimator, **kwargs)
    plt.legend(loc='best') #.set_draggable(True)

def configure_output_dir(G, d=None):
    """
    Set output directory to d, or to /tmp/somerandomnumber if d is None
    """
    # CDIR = os.path.dirname(os.path.realpath(__file__)).replace('\\', '/')
    G.first_row = True
    G.output_dir = d or "/tmp/experiments/%i" % int(time.time())
    assert not os.path.exists(
        G.output_dir), "Log dir %s already exists! Delete it first or use a different dir" % G.output_dir
    os.makedirs(G.output_dir)
    G.output_file = open(os.path.join(G.output_dir, "log.txt"), 'w')
    print(colorize("Logging data to %s" % G.output_file.name, 'green', bold=True))

class LazyLog(object):
    output_dir = None
    output_file = None
    first_row = True
    log_headers = []
    log_current_row = {}

    def __init__(self, experiment_name, run_name=None, data=None):
        if run_name is None:
            experiment_name += "/"+ datetime.utcnow().strftime("%Y-%m-%d_%H-%M-%S.%f")[:-3]
        else:
            experiment_name += "/" + run_name
        configure_output_dir(self, experiment_name)
        if data is not None:
            self.save_params(data)

    def __enter__(self):
        return self

    def save_params(self, data):
        save_params(self, data)

    def dump_tabular(self, verbose=False):
        dump_tabular(self, verbose)

    def log_tabular(self, key, value):
        log_tabular(self, key, value)

    def __exit__(self, type, value, traceback):
        self.output_file.close()

def log_tabular(G, key, val):
    """
    Log a value of some diagnostic
    Call this once for each diagnostic quantity, each iteration
    """
    if G.first_row:
        G.log_headers.append(key)
    else:
        assert key in G.log_headers, "Trying to introduce a new key %s that you didn't include in the first iteration" % key
    assert key not in G.log_current_row, "You already set %s this iteration. Maybe you forgot to call dump_tabular()" % key
    G.log_current_row[key] = val

def dump_tabular(G, verbose=True):
    """
    Write all of the diagnostics from the current iteration
    """
    vals = []
    key_lens = [len(key) for key in G.log_headers]
    max_key_len = max(15, max(key_lens))
    keystr = '%' + '%d' % max_key_len
    fmt = "| " + keystr + "s | %15s |"
    n_slashes = 22 + max_key_len
    print("-" * n_slashes) if verbose else None
    for key in G.log_headers:
        val = G.log_current_row.get(key, "")
        if hasattr(val, "__float__"):
            valstr = "%8.3g" % val
        else:
            valstr = val
        print(fmt % (key, valstr)) if verbose else None
        vals.append(val)
    print("-" * n_slashes) if verbose else None
    if G.output_file is not None:
        if G.first_row:
            G.output_file.write("\t".join(G.log_headers))
            G.output_file.write("\n")
        G.output_file.write("\t".join(map(str, vals)))
        G.output_file.write("\n")
        G.output_file.flush()
    G.log_current_row.clear()
    G.first_row = False

class defaultdict2(collections.defaultdict):
    def __missing__(self, key):
        if self.default_factory is None:
            raise KeyError((key,))

        if isinstance(self.default_factory, types.FunctionType):
            nargs = len(inspect.getfullargspec(self.default_factory).args)
            self[key] = value = self.default_factory(key) if nargs == 1 else self.default_factory()
            return value
        else:
            return super().__missing__(key)

def main_plot(experiments, legends=None, smoothing_window=10, resample_ticks=None,
              x_key="Episode",
              y_key='Accumulated Reward', **kwargs
              ):
    """
    Plot an experiment. To plot invidual lines (i.e. no averaging) use
    > units="Unit", estimator=None,

    """
    ensure_list = lambda x: x if isinstance(x, list) else [x]
    experiments = ensure_list(experiments)

    if legends is None:
        legends = experiments
    legends = ensure_list(legends)

    data = []
    for logdir, legend_title in zip(experiments, legends):
        resample_key = x_key if resample_ticks is not None else None
        data += get_datasets(logdir, x=x_key, condition=legend_title, smoothing_window=smoothing_window, resample_key=resample_key, resample_ticks=resample_ticks)

    plot_data(data, y=y_key, x=x_key, **kwargs)

def get_datasets(fpath, x, condition=None, smoothing_window=None, resample_key=None, resample_ticks=None):
    unit = 0
    if condition is None:
        condition = fpath

    datasets = []
    for root, dir, files in os.walk(fpath):
        if 'log.txt' in files:
            json = os.path.join(root, 'params.json')
            if os.path.exists(json):
                with open(json) as f:
                    param_path = open(json)
                    params = json.load(param_path)
                    # exp_name = params['exp_name']

            log_path = os.path.join(root, 'log.txt')
            if os.stat(log_path).st_size == 0:
                print("Bad plot file", log_path, "size is zero. Skipping")
                continue
            experiment_data = pd.read_table(log_path)
            # raise Exception("Group by ehre.0")
            if smoothing_window:
                ed_x = experiment_data[x]
                experiment_data = experiment_data.rolling(smoothing_window,min_periods=1).mean()
                experiment_data[x] = ed_x

            experiment_data.insert(
                len(experiment_data.columns),
                'Unit',
                unit
            )
            experiment_data.insert(
                len(experiment_data.columns),
                'Condition',
                condition)

            datasets.append(experiment_data)
            # print(experiment_data.columns)
            # if len(experiment_data.columns) > 7:
            #     a = 234
            unit += 1

    nc = f"({unit}x)"+condition[condition.rfind("/")+1:]
    for i, d in enumerate(datasets):
        datasets[i] = d.assign(Condition=lambda x: nc)
        # d.rename(columns={'Condition': nc}, inplace=True)
        # gapminder.rename(columns={'pop': 'population',
        #                           'lifeExp': 'life_exp',
        #                           'gdpPercap': 'gdp_per_cap'},
        #                  inplace=True)

    if resample_key is not None:
        nmax = 0
        vmax = -np.inf
        vmin = np.inf
        for d in datasets:
            nmax = max( d.shape[0], nmax)
            vmax = max(d[resample_key].max(), vmax)
            vmin = min(d[resample_key].min(), vmin)
        if resample_ticks is not None:
            nmax = min(resample_ticks, nmax)

        new_datasets = []
        tnew = np.linspace(vmin + 1e-6, vmax - 1e-6, nmax)
        for d in datasets:
            nd = {}
            cols = d.columns.tolist()
            for c in cols:
                if c == resample_key:
                    y = tnew
                elif d[c].dtype == 'O':
                    # it is an object. cannot interpolate
                    y = [ d[c][0] ] * len(tnew)
                else:
                    y = np.interp(tnew, d[resample_key].tolist(), d[c], left=np.nan, right=np.nan)
                    y = y.astype(d[c].dtype)
                nd[c] = y

            ndata = pd.DataFrame(nd)
            ndata = ndata.dropna()
            new_datasets.append(ndata)
        datasets = new_datasets

    return datasets

def savepdf(pdf):
    '''
    Save command for generating figures.
    '''
    import matplotlib.pyplot as plt
    plt.savefig(pdf)
    # pdf = pdf.strip()
    # pdf = pdf+".pdf" if not pdf.endswith(".pdf") else pdf

    # frame = inspect.stack()[-1]
    # module = inspect.getmodule(frame[0])
    # filename = module.__file__
    # wd = os.path.dirname(filename)
    # pdf_dir = wd +"/pdf"
    # # print(inspect.stack())
    # # print("FILENAME: ", filename)
    # if filename.endswith("_RUN_OUTPUT_CAPTURE.py"):
    #     return
    # if not os.path.isdir(pdf_dir):
    #     os.mkdir(pdf_dir)
    # # print("PDF SAVE> ", wd)
    # if os.path.exists(os.getcwd()+ "/../../../Exercises") and os.path.exists(os.getcwd()+ "/../../../pdf_out"):
    #     # figs = [os.path.join(wd, f"../../../Exercises/ExercisesPython/Exercise{i}/latex/output") for i in range(12)]
    #     lecs = [os.path.join(wd, "../../../shared/output")]
    #     od = lecs+[pdf_dir]
    #     for f in od:
    #         if not os.path.isdir(f):
    #             os.makedirs(f)

    #     on = od[0] + "/" + pdf
    #     plt.savefig(fname=on)
    #     from thtools.slider import convert
    #     convert.pdfcrop(on, fout=on)
    #     for f in od[1:]:
    #         shutil.copy(on, f +"/"+pdf)
    # else:
    #     plt.savefig(fname=wd+"/"+pdf)
    # print(">", pdf)

def log_time_series(experiment, list_obs, max_xticks_to_log=None, run_name=None):
    logdir = f"{experiment}/"

    if max_xticks_to_log is not None and len(list_obs) > max_xticks_to_log:
        I = np.round(np.linspace(0, len(list_obs) - 1, max_xticks_to_log))
        list_obs = [o for i, o in enumerate(list_obs) if i in I.astype(np.int).tolist()]

    with LazyLog(logdir) as logz:
        for n,l in enumerate(list_obs):
            for k,v in l.items():
                logz.log_tabular(k,v)
            if "Steps" not in l:
                logz.log_tabular("Steps", n)
            if "Episode" not in l:
                logz.log_tabular("Episode",n)
            logz.dump_tabular(verbose=False)

def existing_runs(experiment):
    nex = 0
    for root, dir, files in os.walk(experiment):
        if 'log.txt' in files:
            nex += 1
    return nex

def train(env, agent, experiment_name=None, num_episodes=None, verbose=True, reset=True, max_steps=1e10,
          max_runs=None, saveload_model=False):

    if max_runs is not None and existing_runs(experiment_name) >= max_runs:
            return experiment_name, None, True
    stats = []
    steps = 0
    ep_start = 0
    if saveload_model:  # Code for loading/saving models
        did_load = agent.load(os.path.join(experiment_name))
        if did_load:
            stats, recent = load_time_series(experiment_name=experiment_name)
            ep_start, steps = stats[-1]['Episode']+1, stats[-1]['Steps']

    done = False
    with tqdm(total=num_episodes, disable=not verbose) as tq:
        for i_episode in range(num_episodes): 
            s = env.reset() if reset else (env.s if hasattr(env, "s") else env.env.s) 
            reward = []
            for _ in itertools.count():
                a = agent.pi(s)
                sp, r, done, _ = env.step(a)
                agent.train(s, a, r, sp, done)
                reward.append(r)
                steps += 1
                if done or steps > max_steps:
                    break
                s = sp 

            stats.append({"Episode": i_episode + ep_start,
                          "Accumulated Reward": sum(reward),
                          "Average Reward": np.mean(reward),
                          "Length": len(reward),
                          "Steps": steps})
            tq.set_postfix(ordered_dict=OrderedDict(stats[-1]))
            tq.update()
    sys.stderr.flush()
    if saveload_model:
        agent.save(experiment_name)
        if did_load:
            os.rename(recent+"/log.txt", recent+"/log2.txt")  # Shuffle old logs

    if experiment_name is not None:
        log_time_series(experiment=experiment_name, list_obs=stats)
        print(f"Training completed. Logging: '{', '.join( stats[0].keys()) }' to {experiment_name}")
    return experiment_name, stats, done

def cache_write(object, file_name, only_on_professors_computer=False):
    if only_on_professors_computer and not is_this_my_computer():
        """ Probably for your own good :-). """
        return
    # file_name = cn_(file_name) if cache_prefix else file_name
    dn = os.path.dirname(file_name)
    if not os.path.exists(dn):
        os.mkdir(dn)
    print("Writing cache...", file_name)
    with open(file_name, 'wb', ) as f:
        compress_pickle.dump(object, f, compression="lzma")
    print("Done!")


def cache_exists(file_name, cache_prefix=True):
    # file_name = cn_(file_name) if cache_prefix else file_name
    return os.path.exists(file_name)


def cache_read(file_name, cache_prefix=True):
    # file_name = cn_(file_name) if cache_prefix else file_name
    if os.path.exists(file_name):
        with open(file_name, 'rb') as f:
            return compress_pickle.load(f, compression="lzma")
            # return pickle.load(f)
    else:
        return None



