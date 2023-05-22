import torch
import math

HOST = "localhost"
PORT = 20010
# plain, ckks, gpu, gpu_cheetah
CRYPTO = "gpu_cheetah"
CUDA = torch.cuda.is_available()
DEVICE = "cuda" if CUDA else "cpu"

# NETWORK_BANDWIDTH = 384 * (1024 ** 2) # bytes per second
NETWORK_BANDWIDTH = 0

DELPHI_BATCH = True
DELPHI_PROTECT_DIFF = True

assert(DELPHI_BATCH and DELPHI_PROTECT_DIFF)

RANDOM_TENSOR_BOUND = 32

if DELPHI_PROTECT_DIFF:
  assert(DELPHI_BATCH)

FORCE_REGENERATE_PREPARE = False

PAILLIER_N_LENGTH = 192
CKKS_PARAMETER_DICT = {
  4096: ((30, 20, 20, 30), 20),
  8192: ((60, 40, 40, 60), 40),
  16384: ((60, 50, 50, 60), 50),
}
CKKS_POLY_MODULUS_DEGREE = 8192
CKKS_COEFF_MODULUS = CKKS_PARAMETER_DICT[CKKS_POLY_MODULUS_DEGREE][0]

if CRYPTO == "gpu_cheetah":
  CONV2D_MAX_IMAGE_SIZE = int(math.sqrt(CKKS_POLY_MODULUS_DEGREE))
  LINEAR_MAX_COLUMN_SIZE = CKKS_POLY_MODULUS_DEGREE
else:
  CONV2D_MAX_IMAGE_SIZE = int(math.sqrt(CKKS_POLY_MODULUS_DEGREE // 2))
  LINEAR_MAX_COLUMN_SIZE = CKKS_POLY_MODULUS_DEGREE // 2

# CONV2D_MAX_IMAGE_SIZE = 16
# LINEAR_MAX_COLUMN_SIZE = 32

SCALE_EXPONENT = CKKS_PARAMETER_DICT[CKKS_POLY_MODULUS_DEGREE][1] if CRYPTO != 'paillier' else 40
PS_FULL = math.pow(2, SCALE_EXPONENT)
PS_HALF = math.pow(2, SCALE_EXPONENT/2)
PS_TWO_THIRD = math.pow(2, SCALE_EXPONENT/4*3)
PRECISION = 1/PS_FULL
