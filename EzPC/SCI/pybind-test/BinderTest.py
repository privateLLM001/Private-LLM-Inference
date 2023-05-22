import sys
sys.path.append("../build/lib")

import argparse
import numpy as np

import socket
import pickle
import os
import numpy as np
import time
import struct

HOST = "localhost"
PORT = 30010

def send_msg(sock, msg):
    # Prefix each message with a 4-byte length (network byte order)
    msg = struct.pack('>I', len(msg)) + msg
    sock.sendall(msg)

def recvall(sock, n):
    # Helper function to recv n bytes or return None if EOF is hit
    data = bytearray()
    while len(data) < n:
        packet = sock.recv(n - len(data))
        if not packet:
            return None
        data.extend(packet)
    return data

def recv_msg(sock):
    # Read message length and unpack it into an integer
    raw_msglen = recvall(sock, 4)
    if not raw_msglen:
        return None
    msglen = struct.unpack('>I', raw_msglen)[0]
    # Read the message data
    return recvall(sock, msglen)

class ClientCommunication:
  
  def __init__(self):
    pass

  def connect(self):
    self.connection = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
      self.connection.connect((HOST, PORT))
    except:
      self.connection.connect((HOST, PORT + 1))
  
  def close_connection(self):
    self.connection.close()

  def send(self, obj):
    obj_bytes = pickle.dumps(obj)
    length = len(obj_bytes)
    send_msg(self.connection, obj_bytes)

  def recv(self):
    obj_bytes = recv_msg(self.connection)
    return pickle.loads(obj_bytes)


class ServerCommunication:

  def __init__(self):
    pass

  def listen(self):
    self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
      self.socket.bind((HOST, PORT))
    except:
      self.socket.bind((HOST, PORT+1))
    self.socket.listen()
  
  def accept_connection(self):
    self.connection, self.client_address = self.socket.accept()

  def close_connection(self):
    self.connection.close()

  def send(self, obj):
    obj_bytes = pickle.dumps(obj)
    length = len(obj_bytes)
    send_msg(self.connection, obj_bytes)

  def recv(self):
    obj_bytes = recv_msg(self.connection)
    return pickle.loads(obj_bytes)


BFV_POLY_DEGREE = 8192
BFV_Q_BITS = (60, 60, 60)
PLAIN_MODULUS = 1 << 41
SCALE_BITS = 12
SCALE = 1<<SCALE_BITS

def to_field(a: np.ndarray, scale=SCALE):
    a = a.flatten() * scale
    a = np.where(a < 0, PLAIN_MODULUS + a, a).astype(np.uint64)
    return a


def to_decimal(a: np.ndarray, scale=SCALE, shape=None):
    a = a.astype(np.float64)
    a = np.where(a > PLAIN_MODULUS // 2, a - PLAIN_MODULUS, a) / scale
    if shape is not None:
        a = np.reshape(a, shape)
    return a

def field_random_mask(size):
    return np.random.randint(0, PLAIN_MODULUS, size, dtype=np.uint64)

def field_negate(x, field_modulus = PLAIN_MODULUS):
    return np.mod(field_modulus - x, field_modulus)

def field_mod(x, field_modulus = PLAIN_MODULUS):
    return np.mod(x, field_modulus)

def field_add(x, y, field_modulus = PLAIN_MODULUS):
    return np.mod(x + y, field_modulus)

def get_shares(x):
    x1 = field_random_mask(x.size)
    x2 = field_add(x, field_negate(x1))
    return x1, x2

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", type=int)
    args = parser.parse_args()
    party = args.p
    assert(party == 1 or party == 2)
    if party == 1:
        import sci_provider_alice_41 as sci_provider
        comm = ServerCommunication()
        comm.listen()
        comm.accept_connection()
    else:
        import sci_provider_bob_41 as sci_provider
        comm = ClientCommunication()
        comm.connect()

    np.random.seed(12345)

    def reconstruct(x, field_mod = PLAIN_MODULUS):
        comm.send(x)
        return field_add(x, comm.recv(), field_mod)

    provider = sci_provider.SCIProvider(SCALE_BITS)
    provider.startComputation()
    n = 5
    
    print("[Sqrt test]")
    n = 5
    r = np.random.rand(n) * 0.9 + 0.1
    print("x    =", r)
    r_field = to_field(r, SCALE * SCALE)
    r_shares = get_shares(r_field)
    result = provider.sqrt(r_shares[party-1], SCALE * SCALE, SCALE, False)
    result = reconstruct(result)
    result = to_decimal(result)
    print("sqrt =", result)
    
    print("[Exp test]")
    n = 5
    r = np.random.rand(n) * (-0.9) - 0.1
    print("x    =", r)
    r_field = to_field(r, SCALE)
    r_shares = get_shares(r_field)
    result = provider.exp(r_shares[party-1])
    result = reconstruct(result)
    result = to_decimal(result)
    print("exp =", result)
    print("correct = ", np.exp(r))
    
    print("[Exp reduce test]")
    n = 5
    r = np.random.rand(n) * 2 - 1;
    print("x    =", r)
    r_field = to_field(r, SCALE)
    r_shares = get_shares(r_field)
    result = provider.exp_reduce(r_shares[party-1])
    result = reconstruct(result, field_mod = PLAIN_MODULUS >> SCALE_BITS)
    result = to_decimal(result)
    print("exp =", result)
    print("correct = ", np.exp(r))
    
    print("[Softmax test]")
    n = 5
    m = 128
    r = np.random.rand(n * m) * 10 - 5
    r_field = to_field(r, SCALE)
    r_field = (r_field + PLAIN_MODULUS - (3 << SCALE_BITS)) % PLAIN_MODULUS
    r_shares = get_shares(r_field)
    result = provider.softmax(r_shares[party-1], m)
    result = reconstruct(result)
    result = to_decimal(result, shape=(n, m), scale=SCALE*SCALE)
    truth = np.exp(r.reshape((n, m))) / np.sum(np.exp(r.reshape((n, m))), axis=1).reshape((n, 1))
    print("diff = ", np.max(np.abs(truth - result)), "magnitude = ", np.max(np.abs(truth)))

    print("[Elementwise multiply]")
    n = 5
    r1 = np.random.rand(n) * 2 - 1
    r2 = np.random.rand(n) * 2 - 1
    print("x1    =", r1)
    print("x2    =", r2)
    r1_field = to_field(r1, SCALE)
    r1_shares = get_shares(r1_field)
    r2_field = to_field(r2, SCALE)
    r2_shares = get_shares(r2_field)
    result = provider.elementwise_multiply(r1_shares[party-1], r2_shares[party-1])
    result = reconstruct(result)
    result = to_decimal(result, SCALE*SCALE)
    print("mult =", result)

    print("[Tanh test]")
    n = 5
    r = np.random.rand(n) * 2 - 1;
    print("x    =", r)
    r_field = to_field(r, SCALE)
    r_shares = get_shares(r_field)
    result = provider.tanh(r_shares[party-1])
    result = reconstruct(result)
    result = to_decimal(result)
    print("exp =", result)
    print("correct = ", np.tanh(r))

    print("[Div test]")
    n = 5
    r = np.array([1,2,3,0.4,5])
    d = np.array([2,3,4,5,6])
    print("x    =", r)
    r_field = to_field(r, SCALE)
    d_field = to_field(d, SCALE)
    r_shares = get_shares(r_field)
    d_shares = get_shares(d_field)
    result = provider.div(r_shares[party-1], d_shares[party-1])
    result = reconstruct(result)
    result = to_decimal(result, scale=SCALE*SCALE)
    print("div =", result)
    print("correct = ", r / d)

    provider.endComputation()
    comm.close_connection()