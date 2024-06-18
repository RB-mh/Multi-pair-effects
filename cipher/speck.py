import numpy as np
from os import urandom

plain_bits = 32
key_bits = 64
word_size = 16

def WORD_SIZE():
    return(16);

def ALPHA():
    return(7);

def BETA():
    return(2);

MASK_VAL = 2 ** WORD_SIZE() - 1;

def shuffle_together(l):
    state = np.random.get_state();
    for x in l:
        np.random.set_state(state);
        np.random.shuffle(x);

def rol(x,k):
    return(((x << k) & MASK_VAL) | (x >> (WORD_SIZE() - k)));

def ror(x,k):
    return((x >> k) | ((x << (WORD_SIZE() - k)) & MASK_VAL));

def enc_one_round(p, k):
    c0, c1 = p[0], p[1];
    c0 = ror(c0, ALPHA());
    c0 = (c0 + c1) & MASK_VAL;
    c0 = c0 ^ k;
    c1 = rol(c1, BETA());
    c1 = c1 ^ c0;
    return(c0,c1);

def dec_one_round(c,k):
    c0, c1 = c[0], c[1];
    c1 = c1 ^ c0;
    c1 = ror(c1, BETA());
    c0 = c0 ^ k;
    c0 = (c0 - c1) & MASK_VAL;
    c0 = rol(c0, ALPHA());
    return(c0, c1);

def expand_key(k, t):
    ks = [0 for i in range(t)];
    ks[0] = k[len(k)-1];
    l = list(reversed(k[:len(k)-1]));
    for i in range(t-1):
        l[i%3], ks[i+1] = enc_one_round((l[i%3], ks[i]), i);
    return(ks);

# The encrypt function must adhere to this format, with p and k being binary matrices representing the plaintexts and the key, and the return value being a binary matrix as well.
def encrypt(p, k, r):
    P = convert_from_binary(p) 
    K = convert_from_binary(k).transpose()
    ks = expand_key(K, r)
    x, y = P[:, 0], P[:, 1];
    for i in range(r):
        rk = ks[i]
        x,y = enc_one_round((x,y), rk);
    return convert_to_binary([x, y]);



#convert_to_binary takes as input an array of ciphertext pairs
#where the first row of the array contains the lefthand side of the ciphertexts,
#the second row contains the righthand side of the ciphertexts,
#the third row contains the lefthand side of the second ciphertexts,
#and so on
#it returns an array of bit vectors containing the same data
def convert_to_binary(arr):
  X = np.zeros((len(arr) * WORD_SIZE(),len(arr[0])),dtype=np.uint8);
  for i in range(len(arr) * WORD_SIZE()):
    index = i // WORD_SIZE();
    offset = WORD_SIZE() - (i % WORD_SIZE()) - 1;
    X[i] = (arr[index] >> offset) & 1;
  X = X.transpose();
  return(X);

# Convert_from_binary takes as input an n by num_bits binary matrix of type np.uint8, for n samples, 
# and converts it to an n by num_words array of type dtype.
def convert_from_binary(arr, _dtype=np.uint16):
  num_words = arr.shape[1]//WORD_SIZE()
  X = np.zeros((len(arr), num_words),dtype=_dtype);
  for i in range(num_words):
    for j in range(WORD_SIZE()):
        pos = WORD_SIZE()*i+j
        X[:, i] += 2**(WORD_SIZE()-1-j)*arr[:, pos]
  return(X);


def check_testvectors():
  p = np.uint16([0x6574, 0x694c]).reshape(-1, 1)
  k = np.uint16([0x1918, 0x1110, 0x0908, 0x0100]).reshape(-1, 1)
  pb = convert_to_binary(p)
  kb = convert_to_binary(k)
  c = convert_from_binary(encrypt(pb, kb, 22))
  assert np.all(c[0] == [0xa868, 0x42f2])

check_testvectors()


def make_train_data(plain_bits, key_bits, n, nr, delta_state=0, delta_key=0, nr_pairs = 1, reshape = False, delta_state_2 = None, key_addition_dimension = 'first'):
    """
    plain_bits = number of plain bits generated
    key_bits = number of key bits used for encryption
    n = number of generated samples
    nr = number of encrypted rounds
    delta_state = difference between plaintexts
    delta_key = defines difference between used keys, if equal to 'random' a completely random key will be used
    nr_pairs = generated number of pairs per sample
    reshape = if true: reshapes the training data to a 2 dimensional vector
    delta_state_2 = second difference for data generation
    key_addition_dimension = either 'first' or 'second', defines the way a different key is used for exxperiments with different keys
    """
    keys0 = (np.frombuffer(urandom(n*key_bits),dtype=np.uint8)&1)
    keys0 = keys0.reshape(n, key_bits);
    pt0 = (np.frombuffer(urandom(n*plain_bits),dtype=np.uint8)&1).reshape(n, plain_bits);
    if isinstance(delta_key,str) and  delta_key == 'random':
        keys1 = (np.frombuffer(urandom(n*key_bits),dtype=np.uint8)&1)
        keys1 = keys1.reshape(n, key_bits);
    else:
        keys1 = keys0^delta_key

    pt1 = pt0^delta_state
    if key_addition_dimension == 'first':
        C0 = encrypt(pt0, keys0, nr)
        C1 = encrypt(pt1, keys0, nr)
    else:
        C0 = encrypt(pt0, keys0, nr)
        C1 = encrypt(pt1, keys1, nr)
    C = np.hstack([C0, C1])

    if nr_pairs == 2:
        if delta_state_2 is None:
            pt2 = (np.frombuffer(urandom(n*plain_bits),dtype=np.uint8)&1).reshape(n, plain_bits);
            pt3 = pt2^delta_state
        else:
            pt2 = pt1^delta_state_2
            pt3 = pt2^delta_state

        if key_addition_dimension == 'first':
            C2 = encrypt(pt2, keys1, nr)
            C3 = encrypt(pt3, keys1, nr)
        else:
            C0 = encrypt(pt0, keys0, nr)
            C1 = encrypt(pt1, keys1, nr)
        C = np.hstack([C0, C1, C2, C3])
    elif nr_pairs > 2:
        for _ in range(1, nr_pairs):
            pt2 = (np.frombuffer(urandom(n*plain_bits),dtype=np.uint8)&1).reshape(n, plain_bits);
            pt3 = pt2^delta_state

            C2 = encrypt(pt2, keys0, nr)
            C3 = encrypt(pt3, keys1, nr)
            C = np.hstack([C, C2, C3])

    Y = np.frombuffer(urandom(n), dtype=np.uint8); Y = Y & 1;
    num_rand_samples = np.sum(Y==0);
    C[Y==0] = (np.frombuffer(urandom(num_rand_samples*C0.shape[1]*2* nr_pairs),dtype=np.uint8)&1).reshape(num_rand_samples, -1)

    if reshape:
        C = C.reshape((n, nr_pairs, -1)).swapaxes(1,2)
    return C, Y

if __name__ == '__main__':
    make_train_data(32, 64, 10, 5)

