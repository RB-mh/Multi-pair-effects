from cipher import speck
import numpy as np
from nets import dbitnet as dbnet
from nets import train_nets
from cipher import simon


def integer_to_binary_array(int_val, num_bits):
    return np.array([int(i) for i in bin(int_val)[2:].zfill(num_bits)], dtype = np.uint8).reshape(1, num_bits)



def baseline_test (input_differences, cipher = 'speck', evaluation = 0):
	if cipher == 'simon':
		plain_bits = simon.plain_bits
		key_bits = simon.key_bits
		word_size = simon.word_size
		starting_round = 7
		max_rounds = 11
	elif cipher == 'speck':
		plain_bits = speck.plain_bits
		key_bits = speck.key_bits
		word_size = speck.word_size
		starting_round = 5
		max_rounds = 8
	else:
		return

	results = {}


	delta_key = 0
	input_size = plain_bits
	epochs = 10
	num_samples = 10**7
	output_dir = f'results_{cipher}'
	s = 'baseline'
	number_pairs = 1
	net = 'dbitnet'

	for input_difference in input_differences:
		delta = integer_to_binary_array(input_difference, plain_bits)
		diff_str = hex(input_difference)
		results = {}

		starting_round = 7
		delta_plain = delta[:, :plain_bits]


		print(f'Training {net} for input difference {diff_str}, starting from round {starting_round}...')
		results[net] = {}
		reshape = True if net.startswith('2-dim') else False
		if cipher == 'speck':
			data_generator = lambda num_samples, nr : speck.make_train_data(plain_bits, key_bits, num_samples, nr, delta_plain, delta_key, number_pairs, reshape)
		else : 
			data_generator = lambda num_samples, nr : simon.make_train_data(plain_bits, key_bits, num_samples, nr, delta_plain, delta_key, number_pairs, reshape)
		best_round, best_val_acc, message = train_nets.train_neural_distinguisher(
        	starting_round = starting_round,
            data_generator = data_generator,
            model_name = net,
            input_size = input_size,
            word_size = word_size,
            log_prefix = f'{output_dir}/{s}_{hex(input_difference)}',
            _epochs = epochs,
            _num_samples = num_samples,
            num_pairs = number_pairs,
            max_round= max_rounds,
            cipher= cipher,
            evaluate = evaluation)
		results[net]['Best round'] = best_round
		results[net]['Validation accuracy'] = best_val_acc



		with open(f'{output_dir}/{s}_{hex(input_difference)}_final_result', 'a') as f:
			f.write(diff_str)
			f.write(f'{net} {number_pairs} pairs : {results[net]["Best round"]}, {results[net]["Validation accuracy"]}\n')
			f.write(message)
		print(results)





if __name__ == '__main__':
	input_differences = [0x00400000, 0x01408000, 0x28000010, 0x00081000, 0x00102000] #speck
	#input_differences = [0x40, 0x1, 0x100040, 0x40000041] #simon

	cipher = 'speck' #either 'speck' or 'simon'
	evaluation = 2 # number of pairs combined by the cinfidence scores

	baseline_test(input_differences, cipher, evaluation)