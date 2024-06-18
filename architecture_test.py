from cipher import speck
import numpy as np
from nets import train_nets
from cipher import simon


def integer_to_binary_array(int_val, num_bits):
    return np.array([int(i) for i in bin(int_val)[2:].zfill(num_bits)], dtype = np.uint8).reshape(1, num_bits)

def architecture_test (cipher = 'speck', delta_state = None, nets = ['2-dim-PP'], number_pairs = 2):
    if cipher == 'simon':
        plain_bits = simon.plain_bits
        key_bits = simon.key_bits
        word_size = simon.word_size
        input_difference = 0x40 if delta_state is None else delta_state 
        starting_round = 7
        max_rounds = 11
    elif cipher == 'speck':
        plain_bits = speck.plain_bits
        key_bits = speck.key_bits
        word_size = speck.word_size
        input_difference = 0x00400000 if delta_state is None else delta_state 
        starting_round = 5
        max_rounds = 8
    else:
        return



    delta = integer_to_binary_array(input_difference, plain_bits)



    diff_str = hex(input_difference)
    results = {}

    delta_plain = delta[:, :plain_bits]

    delta_key = 0
    input_size = plain_bits
    epochs = 10
    num_samples = 10**7
    output_dir = f'results_{cipher}'
    s = 'architecture'

    for net in nets:
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
            cipher= cipher)
        results[net]['Best round'] = best_round
        results[net]['Validation accuracy'] = best_val_acc



        with open(f'{output_dir}/{s}_{hex(input_difference)}_final_result', 'a') as f:
            f.write(diff_str)
            f.write(f'{net} {number_pairs} pairs : {results[net]["Best round"]}, {results[net]["Validation accuracy"]}\n')
            f.write(message)
        print(results)


if __name__ == '__main__':
    
    nets = ['2-dim-PP'] # can either be ['1-dim-AKS', '1-dim-FKS', '2-dim-PP', '2-dim-NPP']
    cipher = 'speck' # either 'simon' or 'speck'
    number_pairs = 2 # number of pairs per sample
    delta_state = None # used to define differences, if equal None the basic differences from literature are used
    architecture_test(cipher, delta_state=delta_state, nets = nets, number_pairs= number_pairs)