import json
import traceback

import click
from tqdm import tqdm

from estimators.division_filter import GreedyFilter, FullFilter
from estimators.meta import MetaEstimator
from estimators.splitter import ShorterSplitter
from solution import Instance
from utils.lazy_class import ForceClass
from utils.regressor import ModelRegressor, ZeroRegressor


@click.command()
@click.option('--nn', help='Path to neural network.')
@click.option('--instances_file', help='Json file with list of instances.')
@click.option('--instance', help='')
@click.option('--output_file', help='Path to file with results.')
def evaluate_instance(nn=None, instances_file=None, instance=None, output_file=None):
    instances = []
    if instance:
        print(instance)
        instances.append(json.loads(instance))
    if instances_file:
        instances += json.load(open(instances_file))

    if len(instances) == 0:
        while True:
            try:
                print('Give me instances processing time separated by space (e.g. 3 10 11 12)')
                proc = list(map(int, input().split()))
                print('Give me instances due dates separated by space (e.g. 3 10 11 12)')
                due = list(map(int, input().split()))
                instances.append({'proc': proc, 'due': due})
                print('-1 .. exit input of instance, else continue.')
                if input() == '-1':
                    break
            except Exception as e:
                print('Error during parsing of the instance.')
                print(e)
                print(traceback.format_exc())


    print(f'Loaded instances {len(instances)}.')
    print('Try to build instances.')
    instances_obj = {}
    for i, inst in enumerate(instances):
        try:
            instances_obj[i] = Instance.from_lists(inst['proc'], inst['due'])
        except Exception as e:
            print(f'During loading {i} instance to object errors occur.')
            print(f'Instance data: {inst}.')
            print(e)
            print(traceback.format_exc())

    if len(instances_obj) == 0:
        print('There is no instances for evaluation, exit.')
        return 1
    print(f'Loaded instances to object {len(instances_obj)}.')
    print('Try to load NN and build estimator.')
    if not nn:
        print('Give me path to NN model.')
        nn = input()
    reg = ModelRegressor.lazy_init(nn, '', 5)
    print('NN loaded successfully.')
    print('Try to build estimator.')
    est: MetaEstimator = MetaEstimator.from_lazy(ForceClass(ShorterSplitter), reg, ForceClass(GreedyFilter)).force()
    print('Estimator build successfully.')
    print('Try to estimate first instance and fully load the NN.')
    est.estimate(instances_obj[0])

    print('Start evaluation of the instances.')
    results = {}
    for k, inst in tqdm(instances_obj.items()):
        results[k] = {}
        results[k]['HORDA'] = est.estimate(inst)[0].to_human()

    full = MetaEstimator.from_lazy(ForceClass(ShorterSplitter), ForceClass(ZeroRegressor), ForceClass(FullFilter)).force()
    for k, inst in tqdm(instances_obj.items()):
        results[k]['OPT'] = full.estimate(inst)[0].to_human()

    print('Results:')
    print(results)

    if output_file:
        json.dump(results, open(output_file, 'w'))

    return 0

if __name__ == '__main__':
    evaluate_instance()