import os
import subprocess

output = subprocess.check_output('perl multi-bleu.pl prediction/eps-40k-ml10-3trans/reference.txt < prediction/experiments/viterbi-predictions-{0}.txt'.format(0+1),
								 shell=True)

# print(str(output).split()[2][0:-1])
print((str(output).split()[3]).split('/')[0])
print((str(output).split()[3]).split('/')[1])

