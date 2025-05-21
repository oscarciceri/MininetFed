
import numpy as np
from Pyfhel import Pyfhel

# l = 128 #quantidade de linhas concat_actv
# # TODO: achar valor real da quantidade de linhas da última camada


# We will define a pair of 1D integer vectors of size l.
#  For the purpose of this demo, we can study two cases:
#  #1: small l (l <= n).     --> encoding with trailing zeros
#  #3: large l (l > n).      --> several ciphertexts per vector

# @User: You can modify the case selection below
CASE_SELECTOR = 1          # 1 or 2

case_params = {
    1: {'l': 256},         # small l
    2: {'l': 65536},       # large l
}[CASE_SELECTOR]
l = case_params['l']


def bitsize(x): return np.ceil(np.log2(x))
def get_closest_power_of_two(x): return int(2**(bitsize(x)))


def get_CKKS_context_scalar_prod(
    l: int, sec: int = 128,
    use_n_min: bool = True
) -> Pyfhel:
    """
    Returns the best context parameters to compute scalar product in CKKS scheme.

    The important context parameters to set are:
    - n: the polynomial modulus degree (= 2*n_slots)

    *Optimal n*: Chosen among {2**12, 2**13, 2**14, 2**15}.
        The bigger n, the more secure the scheme, but the slower the computations.
        It might be faster to use n<l and have multiple ciphertexts pervector.

    Arguments:
        l: vector length
        v_max: max element value

    Returns:
        Pyfhel: context to perform homomorphic encryption
    """
    # > OPTIMAL n
    n_min = 2**14
    # n_min = 2**15
    if use_n_min:
        n = n_min    # use n_min regardless of l
    elif 2*l < n_min:
        n = n_min    # Smallest
    elif 2*l > 2**15:
        n = 2**15    # Largest
    else:
        n = get_closest_power_of_two(2*l)

    context_params = {
        'scheme': 'CKKS',
        'n': n,          # Poly modulus degree. BFV ptxt is a n//2 by 2 matrix.
        'sec': sec,      # Security level.
        'scale': 2**30,
        # 'qi_sizes': [60] + 10 * [30] + [60],   # Max number of multiplications = 1
        # Max number of multiplications = 1
        'qi_sizes': [60] + 5 * [30] + [60],
    }
    HE = Pyfhel(context_params)
    return HE


HE = get_CKKS_context_scalar_prod(l, sec=128, use_n_min=True)
HE.keyGen()
HE.relinKeyGen()
HE.rotateKeyGen()

#dir_path = "/home/johann/Documents/MininetFed/temp/ckksfed_fhe/pasta"
# dir_path = "/home/user/INESC_TEC/MininetFed/temp/ckksfed_fhe/pasta"
dir_path = "/pasta"


# Now we save all objects into files
HE.save_context(dir_path + "/context")
HE.save_public_key(dir_path + "/pub.key")
HE.save_secret_key(dir_path + "/sec.key")
HE.save_relin_key(dir_path + "/relin.key")
HE.save_rotate_key(dir_path + "/rotate.key")

# print("2a. Saving everything into files. Let's check the temporary dir:")
# print("\n\t".join(os.listdir(dir_path)))

# Now we restore everything and quickly check if it works.
#  Note! make sure to set the `pyfhel` parameter in PyCtxt/PyPtxt creation!


# CLIENTE:
HE_f = Pyfhel()  # Empty creation
HE_f.load_context(dir_path + "/context")
HE_f.load_public_key(dir_path + "/pub.key")
HE_f.load_secret_key(dir_path + "/sec.key")
HE_f.load_relin_key(dir_path + "/relin.key")
HE_f.load_rotate_key(dir_path + "/rotate.key")

print("2b. Loading everything from files into a new environment.")
