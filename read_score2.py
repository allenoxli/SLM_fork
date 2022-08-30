
import argparse
import re
import os

parser = argparse.ArgumentParser()
parser.add_argument('--name', type=str, default='models_normal_87')

args = parser.parse_args()

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
        if 'score' in file:
            name = file.split('_score')[0]
            print(f'{exp}_{name}')
            res[f'{exp}_{name}'] = read_score(file_path)

## Display
def display(res):
    out_str = f'|name|bound|f1|precision|recall|\n'
    out_str += f'|-|-|-|-|-|\n'
    for exp, info in res:
        bound, name = exp.split('_')
        bound = bound.replace('bound', '')
        f1, precision, recall = info['f1'], info['precision'], info['recall']
        out_str += '|{}|{}|{:.1f}|{:.1f}|{:.1f}|\n'.format(
            name.ljust(5, " "), bound.ljust(3, " "), f1, precision, recall
        )
    out_str += '\n\n'

    return out_str



res = sorted(res.items(), key=lambda x: x[0])


file_str = f'\n=== {model_path} ===\n'
file_str += display(res)

print(file_str)
with open(f'{model_path}/read_result.txt', 'w') as f_out:
    f_out.write(file_str)

