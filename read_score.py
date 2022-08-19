
import argparse
import re
import os

parser = argparse.ArgumentParser()
parser.add_argument('--name', type=str, default='models_normal_87')

args = parser.parse_args()


model_path = 'models_classifier_zip'
model_path = 'models_iterative_zip'
model_path = 'models_normal'
model_path = 'models_iterative'
model_path = 'models_iterative_lm'
model_path = 'models_iterative_1234'
model_path = 'models_iterative'
model_path = 'models_circular'
model_path = 'models_normal_87'


model_path = args.name

# model_path = 'models_classifier'

res = {}
def read_score(file_path):
    f1, precision, recall = None, None, None
    f_in = open(file_path, 'r').readlines()[-15:]
    for line in f_in:
        if '=== F MEASURE' in line:
            f1 = line.split()[-1]
        if '=== TOTAL TEST WORDS PRECISION' in line:
            precision = line.split()[-1]
        if '=== TOTAL TRUE WORDS RECALL' in line:
            recall = line.split()[-1]
    try:
        return {
            'f1': float(f1)*100,
            'precision': float(precision)*100,
            'recall': float(recall)*100,
        }
    except:
        return {
            'f1': .0,
            'precision': .0,
            'recall': .0,
        }


dirs = os.listdir(model_path)
for exp in dirs:
    if os.path.isfile(os.path.join(model_path, exp)):
        continue
    for file in os.listdir(os.path.join(model_path, exp)):
        file_path = os.path.join(model_path, exp, file)
        if file == 'train.log':
            line = open(file_path, 'r').readlines()[0]
            seed = re.search('seed=\d+', line).group()
            batch_size = re.search('unsupervised_batch_size=\d+', line).group()
            learning_rate = re.search('adam_learning_rate=\d+.?\d+', line).group()
            warm_up_steps = re.search('warm_up_steps=\d+', line).group()
            print(f'{seed}, {batch_size}, {learning_rate}, {warm_up_steps}, {exp}')
        if file not in ['score.txt', 'score_cls.txt']:
            continue
        name = 'cls' if 'cls' in file_path else 'seg'
        res[f'{exp}-{name}'] = read_score(file_path)



## Display
def display(res, name):
    out_str = f'--- {name} ---\n'
    out_str += f'|name|f1|precision|recall|\n'
    out_str += f'|-|-|-|-|\n'
    for exp, info in res:
        f1, precision, recall = info['f1'], info['precision'], info['recall']
        if name not in exp:
            continue
        exp = exp.replace('unsupervised-', '')
        exp = exp.replace(f'-{name}', '')
        out_str += '|{}|{:.1f}|{:.1f}|{:.1f}|\n'.format(exp.ljust(7, " "), f1, precision, recall)

    out_str += '\n\n'

    return out_str



res = sorted(res.items(), key=lambda x: x[0])


file_str = f'\n=== {model_path} ===\n'
file_str += display(res, 'seg')
file_str += display(res, 'cls')

print(file_str)
with open(f'{model_path}/read_result.txt', 'w') as f_out:
    f_out.write(file_str)

