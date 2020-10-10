import os

local_ep_epoch = (1, 100)
datasets = ['all_data_1_digits_1_niid', 'all_data_1_digits_10_niid',
           'all_data_1_digits_5_niid', 'all_data_1_digits_2_niid']

num_users_sel = 100
compressors = ['none', 'uniform_drop']
rates = [0.1, 0.3, 0.5, 0.7, 0.9]
models = 'cnn'
data = 'mnist'

attacks = ['none', 'byzantine']
defenses = ['none', 'krum']
percentage = '0.2'

for compressor in compressors:
	for i, rate in enumerate(rates):
		if compressor == 'none' and i != 0:
			break
		for attack in attacks:
			for defense in defenses:
				for dataset in datasets:
					suffix = "python3 run_fl.py" + " --dataset=" + data + " --dataset_name=" + str(dataset) \
					         + " --model=" + str(models) \
					         + " --num_round=" + str(local_ep_epoch[1]) \
					         + " --clients_per_round=" + str(num_users_sel) \
					         + " --num_epochs=" + str(local_ep_epoch[0]) \
							 + " --attack=" + attack + " --defense=" + defense \
							 + " --attack_percentage=" + percentage \
							 + " --error_feedback=True"
					os.system(suffix)

					suffix = "python3 run_fl.py" + " --dataset=" + data + " --dataset_name=" + str(dataset) \
					         + " --model=" + str(models) \
					         + " --num_round=" + str(local_ep_epoch[1]) \
					         + " --clients_per_round=" + str(num_users_sel) \
					         + " --num_epochs=" + str(local_ep_epoch[0]) \
					         + " --attack=" + attack + " --defense=" + defense \
					         + " --attack_percentage=" + percentage \
					         + " --error_feedback=False"
					os.system(suffix)


