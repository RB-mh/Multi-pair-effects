from cipher import speck
import numpy as np
from nets import dbitnet as dbnet
from nets import train_nets
from cipher import simon

def integer_to_binary_array(int_val, num_bits):
    return np.array([int(i) for i in bin(int_val)[2:].zfill(num_bits)], dtype = np.uint8).reshape(1, num_bits)



def combination_test (input_differences, cipher = 'speck', number_pairs = 2, mixed = False):
    if cipher == 'simon':
        plain_bits = simon.plain_bits
        key_bits = simon.key_bits
        word_size = simon.word_size
        input_difference = 0x40
        starting_round = 7
        max_rounds = 11
    elif cipher == 'speck':
        plain_bits = speck.plain_bits
        key_bits = speck.key_bits
        word_size = speck.word_size
        input_difference = 0x00400000
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
    net = '2-dim-NPP'
    s = 'combination'

    for i in range(len(input_differences)):
        for j in range(len(input_differences)):

            if i == j:
                continue

            if mixed:
                diff1 = integer_to_binary_array(input_differences[i] ^ input_differences[j], plain_bits)
                diff1_plain = diff1[:, :plain_bits]
                diff2 = integer_to_binary_array(input_differences[i], plain_bits)
                diff2_plain = diff2[:, :plain_bits]
            else:
                diff1 = integer_to_binary_array(input_differences[i], plain_bits)
                diff1_plain = diff1[:, :plain_bits]
                diff2 = integer_to_binary_array(input_differences[j], plain_bits)
                diff2_plain = diff2[:, :plain_bits]




            print(f'Training {net} for input difference {hex(input_differences[i])}, {hex(input_differences[j])}, starting from round {starting_round}...')
            results[f'{diff1}, {diff2}'] = {}
            reshape = True if net == 'neben' else False
            if cipher == 'speck':
                data_generator = lambda num_samples, nr : speck.make_train_data(plain_bits, key_bits, num_samples, nr, diff1_plain, delta_key, number_pairs, reshape, diff2_plain)
            else : 
                data_generator = lambda num_samples, nr : simon.make_train_data(plain_bits, key_bits, num_samples, nr, diff1_plain, delta_key, number_pairs, reshape, diff2_plain)
            best_round, best_val_acc, message = train_nets.train_neural_distinguisher(
                starting_round = starting_round,
                data_generator = data_generator,
                model_name = net,
                input_size = input_size,
                word_size = word_size,
                log_prefix = f'{output_dir}/{s}',
                _epochs = epochs,
                _num_samples = num_samples,
                num_pairs = number_pairs,
                max_round= max_rounds,
                cipher = cipher)
            results[f'{diff1}, {diff2}']['Best round'] = best_round
            results[f'{diff1}, {diff2}']['Validation accuracy'] = best_val_acc


            suffix = '_mixed' if mixed else ''
            with open(f'{output_dir}/{s}_final_results{suffix}', 'a') as f:
                f.write(f'{diff_str}\n')
                f.write(f'{net} ({input_differences[i]}, {input_differences[j]}) {number_pairs} pairs : {results[f"{diff1}, {diff2}"]["Best round"]}, {results[f"{diff1}, {diff2}"]["Validation accuracy"]}\n')
                f.write(message)
            print(results)





if __name__ == '__main__':
    cipher = 'speck' # either 'speck' or 'simon'
    input_differences = [0x400000, 0x1408000, 0x28000010, 0x81000, 0x102000] # speck
    #input_differences = [0x40, 0x1, 0x100040, 0x40000041] # simon

    mixed = False # set this to true if pairs of (x, x+a+b) (x+a, x+b) should be considered, else (x,x+a)(x+b,x+a+b) is considered

    combination_test(input_differences, cipher, mixed = mixed)