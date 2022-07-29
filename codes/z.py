import os
import subprocess

'/home/P76094436/project/SLM/data/score.pl'



SCRIPT='/home/P76094436/project/SLM/data/score.pl'
TRAINING_WORDS='/home/P76094436/project/SLM/data/cityu/words.txt'
VALID_OUTPUT='/home/P76094436/project/SLM/models_classifier/unsupervised-cityu-3/valid_prediction.txt'
VALID_SCORE='/home/P76094436/project/SLM/models_classifier/zzzz_score.txt'

GOLD_TEST = '/home/P76094436/project/SLM/data/cityu/test_gold.txt'

eval_command_slm = f'perl {SCRIPT} {TRAINING_WORDS} {GOLD_TEST} {VALID_OUTPUT}'
eval_command = f'perl {SCRIPT} {TRAINING_WORDS} {GOLD_TEST} {VALID_OUTPUT}'
# out = os.popen(eval_command_slm)
# a = out.readlines()
# out = os.popen('perl --help > zzz.txt')

out = subprocess.Popen(eval_command.split(' '),
                        stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT)
stdout, stderr = out.communicate()
stdout = stdout.decode("utf-8")
print('Test evaluation results:\n%s' % stdout)

a = stdout.split('\n')[-15:]
print('\n'.join(a))

with open(VALID_SCORE, 'w') as f:
    f.write(stdout)