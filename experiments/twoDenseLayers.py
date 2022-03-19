import numpy as np
from numpy import random, linalg as LA
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
import tensorflow as tf
import gc

with tf.device('/device:GPU:0'):
  VECTOR_SIZE = 8

  results = []

  def projected_vector(basis, vector):
    # our basis vector here is row vector so (1, 10) => we want something like (10,1)
    return np.matmul(basis.T, vector)

  def projection(basis, vector):
    return LA.norm(projected_vector(basis, vector))

  def lifting(initial_vector, basis_matrix, energy):
  # projection of vector on subspace
    p1 = np.matmul(np.matmul(basis_matrix.T, basis_matrix), initial_vector)
    p2 = initial_vector - p1
    py = energy * p1 / (LA.norm(p1) + 0.001) + (1 - energy**2)**0.5 * p2 / (LA.norm(p2) + 0.001)
    py = py/LA.norm(py)
    return py

  def get_data(vector_size, no_of_rows, no_of_columns):
    # starting with creating bunch of subspaces (no_of_columns)
    random_vectors = random.rand(256, vector_size)
    subspaces_basis = []

    # adding N number of subspaces in a list(then we can count energy for random signals)
    for __ in range(no_of_columns):
      subspace_size = random.randint(1, vector_size + 1)
      random_indices = random.choice(256, size = subspace_size)
      while True:
        a = random_vectors[random_indices].T
        basis, _ = LA.qr(a)
        if LA.matrix_rank(basis) == subspace_size:
          break
      subspaces_basis.append(basis)
    
    Y = []
    X = []

    for _ in range(no_of_rows):
      random_signal = random.rand(vector_size)
      random_signal = random_signal/np.sum(random_signal)
      Y.append(random_signal)
      X.append([projection(basis, random_signal) for basis in subspaces_basis])
    X = np.array(X)
    return X, np.array(Y)

  def avg_dist(y_true, y_pred):
    return tf.reduce_sum(tf.abs(y_true-y_pred), axis=1)

  def avg_norm(y_true, y_pred):
    return tf.norm(y_true-y_pred, axis=1)

  def success(y_true, y_pred):
    return tf.reduce_sum(tf.abs(y_true-y_pred), axis=1) < 0.05

  def success_norm(y_true, y_pred):
    return tf.norm(y_true-y_pred, axis=1) < 0.05


  # fixing the input data
  random.seed(42)
  NO_SUBSPACES = 50
  NO_SIGNALS = 10_000
  X, Y = get_data(VECTOR_SIZE, NO_SIGNALS, NO_SUBSPACES)
  # X.shape, Y.shape

  for i in range(2, 50, 2):
    X_train = X[:7000, :i] # first i columns
    X_test = X[7000:, :i] # first i columns
    y_train = Y[:7000] # no need to change output signal
    y_test = Y[7000:]
  #   print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
    model = Sequential()
    # our dense layer will have say i neurons for i subspaces
    model.add(Dense(i, input_dim=i, activation='sigmoid'))
    model.add(Dense(8, input_dim=50, activation='sigmoid'))
    # will return average distance
    model.compile(
        optimizer='adam', 
        loss='mean_squared_error',
        metrics=[
            avg_dist,
            avg_norm,
            success,
            success_norm
        ])
    history = model.fit(X_train, y_train, epochs=150, batch_size=8)
    _, avg_norm_of_signal, avg_distance, success_, success_norm_ = model.evaluate(X_test, y_test, batch_size=8)

    print(f'manhattan dist:: {avg_distance}, euclidean dist:: {avg_norm_of_signal}, manhattan succ: {success_}, euclidean succ: {success_norm_}')
    results.append({
        'manh': avg_distance,
        'eucl': avg_norm_of_signal,
        'manh_succ': success_,
        'eucl_succ': success_norm_
    })
    del history
    del _, avg_norm_of_signal, avg_distance, success_, success_norm_
    gc.collect()
  print(results)
  y = list(map(lambda _: _['eucl_succ']*100, results))
  x = (np.arange(25) + 2)*2
  plt.plot(x, y)
  plt.xlabel("Number of subspaces")
  plt.ylabel("Success(%)")
  # plt.legend(loc='upper left')
  plt.show()

  """
  [
      {'manh': 0.17960424721240997, 'eucl': 0.43445783853530884, 'manh_succ': 0.0, 'eucl_succ': 0.0023333332501351833}, 
      {'manh': 0.17052434384822845, 'eucl': 0.40939071774482727, 'manh_succ': 0.0, 'eucl_succ': 0.0033333334140479565}, 
      {'manh': 0.16921262443065643, 'eucl': 0.40584996342658997, 'manh_succ': 0.0, 'eucl_succ': 0.0033333334140479565}, 
      {'manh': 0.16737642884254456, 'eucl': 0.40109485387802124, 'manh_succ': 0.0, 'eucl_succ': 0.004333333112299442}, 
      {'manh': 0.15668168663978577, 'eucl': 0.37439844012260437, 'manh_succ': 0.0, 'eucl_succ': 0.007666666526347399}, 
      {'manh': 0.13741815090179443, 'eucl': 0.3228459358215332, 'manh_succ': 0.0, 'eucl_succ': 0.02033333294093609}, 
      {'manh': 0.12893956899642944, 'eucl': 0.3023936450481415, 'manh_succ': 0.0, 'eucl_succ': 0.03433333337306976}, 
      {'manh': 0.11152348667383194, 'eucl': 0.2646232545375824, 'manh_succ': 0.0020000000949949026, 'eucl_succ': 0.06433333456516266}, 
      {'manh': 0.11188417673110962, 'eucl': 0.2653805911540985, 'manh_succ': 0.0016666667070239782, 'eucl_succ': 0.0573333315551281}, 
      {'manh': 0.093061164021492, 'eucl': 0.22004854679107666, 'manh_succ': 0.011666666716337204, 'eucl_succ': 0.13633333146572113}, 
      {'manh': 0.08718611299991608, 'eucl': 0.20638507604599, 'manh_succ': 0.00800000037997961, 'eucl_succ': 0.16500000655651093}, 
      {'manh': 0.08587413281202316, 'eucl': 0.20288315415382385, 'manh_succ': 0.01133333332836628, 'eucl_succ': 0.17900000512599945}, 
      {'manh': 0.08401232212781906, 'eucl': 0.19968833029270172, 'manh_succ': 0.008999999612569809, 'eucl_succ': 0.19300000369548798}, 
      {'manh': 0.08149350434541702, 'eucl': 0.19332118332386017, 'manh_succ': 0.015333333052694798, 'eucl_succ': 0.21066667139530182}, 
      {'manh': 0.06446227431297302, 'eucl': 0.15233908593654633, 'manh_succ': 0.044333335012197495, 'eucl_succ': 0.37299999594688416}, 
      {'manh': 0.05204534903168678, 'eucl': 0.12160715460777283, 'manh_succ': 0.06866666674613953, 'eucl_succ': 0.5306666493415833}, 
      {'manh': 0.0491911880671978, 'eucl': 0.1146136149764061, 'manh_succ': 0.08100000023841858, 'eucl_succ': 0.5709999799728394}, 
      {'manh': 0.050162557512521744, 'eucl': 0.11780693382024765, 'manh_succ': 0.05166666582226753, 'eucl_succ': 0.5609999895095825}, 
      {'manh': 0.04264616221189499, 'eucl': 0.09898287802934647, 'manh_succ': 0.10366666316986084, 'eucl_succ': 0.7210000157356262}, 
      {'manh': 0.040579039603471756, 'eucl': 0.09378597885370255, 'manh_succ': 0.1340000033378601, 'eucl_succ': 0.7630000114440918}, 
      {'manh': 0.03960931673645973, 'eucl': 0.09224795550107956, 'manh_succ': 0.14666666090488434, 'eucl_succ': 0.7739999890327454}, 
      {'manh': 0.03790315240621567, 'eucl': 0.0878317803144455, 'manh_succ': 0.15199999511241913, 'eucl_succ': 0.8103333115577698}, 
      {'manh': 0.039230167865753174, 'eucl': 0.09202012419700623, 'manh_succ': 0.10833333432674408, 'eucl_succ': 0.7950000166893005}, 
      {'manh': 0.03710898756980896, 'eucl': 0.084768146276474, 'manh_succ': 0.1599999964237213, 'eucl_succ': 0.8343333601951599}, 
      {'manh': 0.032601144164800644, 'eucl': 0.07496318221092224, 'manh_succ': 0.2370000034570694, 'eucl_succ': 0.9006666541099548}]
  """
