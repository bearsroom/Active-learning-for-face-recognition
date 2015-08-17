
import caffe
import numpy as np
import h5py
import datetime, time
import google.protobuf as pb2

def load_data_hdf5():
# load data, data in format bc01, targets as 1D vector giving the correct class index par data point
    train_db = h5py.File('/home/liyh/DeepFace2Aug/train/siamese_db_train.h5', 'r')
    train_data_left = train_db['data_full_1.00_left']
    train_data_right = train_db['data_full_1.00_right']
    train_targets = train_db['label_full_1.00_caffe']

    test_db = h5py.File('/home/liyh/lfwAug/test/db_test.h5', 'r')
    test_data_left = test_db['data_full_1.00_left']
    test_data_right = test_db['data_full_1.00_right']
    test_targets = test_db['label_full_1.00_caffe']
    return train_data_left, train_data_right, train_targets, test_data_left, test_data_right, test_targets

def iterate_minibatches(inputs_left, inputs_right, targets, batchsize, shuffle=False):
    assert inputs_left.shape[0] == targets.shape[0]
    assert inputs_left.shape == inputs_right.shape
    if shuffle:
        indices = np.arange(inputs_left.shape[0])
        np.random.shuffle(indices)
    for start_idx in range(0, inputs_left.shape[0] - batchsize + 1, batchsize):
        if shuffle:
            excerpt = np.sort(indices[start_idx:start_idx + batchsize]).tolist()
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        yield inputs_left[excerpt], inputs_right[excerpt], targets[excerpt]

def trim_batches_data(X, X_p, y, batch_size):
    assert X.shape == X_p.shape
    assert X.shape[0] == y.shape[0]
    num_data = X.shape[0] - np.remainder(X.shape[0], batch_size)
    return X[:num_data], X_p[:num_data], y[:num_data]

def calculate_accuracy_siamese(a, b, y, threshold):
    assert a.shape == b.shape
    distance = np.linalg.norm((a-b), axis=1)
    assert distance.shape[0] == y.shape[0]
    predict = np.zeros(len(distance))
    predict[distance <= threshold] = 1
    y = np.squeeze(y)
    return np.mean(predict == y)

# set gpu
caffe.set_mode_gpu()
caffe.set_device(0)

# init SGD solver
print("Initialize the solver...")
solver_prototxt = '/home/liyh/people_200_models/siamese_solver_memory.prototxt'
solver = caffe.SGDSolver(solver_prototxt)
solver_param = caffe.proto.caffe_pb2.SolverParameter()
with open(solver_prototxt, 'rt') as fd:
    pb2.text_format.Merge(fd.read(), solver_param)

# load caffe model
print("Load pretrained model...")
caffemodel = '/home/liyh/people_200_models/train_model_full_1.00_no_group_iter_50000.caffemodel'
solver.net.copy_from(caffemodel)

# load data
print("Load data...")
train_batch_size = solver.net.blobs['data_full_1.00'].data.shape[0]
test_batch_size = solver.test_nets[0].blobs['data_full_1.00'].data.shape[0]
X_train, X_train_p, y_train, X_test, X_test_p, y_test = load_data_hdf5()

# fisrt test before training
test_loss = 0
test_acc = 0
test_batches = 0
for batch in iterate_minibatches(X_test, X_test_p, y_test, test_batch_size, shuffle=True):
    X, X_p, y = batch
    solver.test_nets[0].set_input_arrays(np.concatenate((X, X_p), axis=0), np.concatenate((y, y), axis=0))
    solver.test_nets[0].forward()
    test_loss += solver.test_nets[0].blobs['loss'].data
    test_acc += solver.test_nets[0].blobs['predict_error'].data
    test_batches += 1
test_loss = test_loss / test_batches
test_acc = 1 - 2 * test_acc / test_batches
print("["+str(datetime.datetime.now().time())+"]  First test before training")
print("    test loss:\t\t{:.6f}".format(test_loss))
print("    test acc:\t\t{:.6f}".format(test_acc))

# launch the training loop
print("["+str(datetime.datetime.now().time())+"]  Starting training...")
start = time.time()

# iterate over epochs:
num_epochs = 10
for epoch in range(num_epochs):
    train_loss = 0
    train_acc = 0
    train_batches = 0
    print("["+str(datetime.datetime.now().time())+"]  Epoch {} starts. caffe iter {}".format(epoch+1, solver.iter))
    start_time = time.time()
    for batch in iterate_minibatches(X_train[:11000], X_train_p[:11000], y_train[:11000], train_batch_size, shuffle=True):
        X, X_p, y = batch
        solver.net.set_input_arrays(np.concatenate((X, X_p), axis=0), np.concatenate((y, y), axis=0))
        solver.step(1)
        train_loss += solver.net.blobs['loss'].data
        train_acc += solver.net.blobs['predict_error'].data
        train_batches += 1
    train_loss = train_loss / train_batches
    train_acc = 1 - 2 * train_acc /train_batches    

    test_loss = 0
    test_acc = 0
    test_batches = 0
    for batch in iterate_minibatches(X_test, X_test_p, y_test, test_batch_size, shuffle=True):
        X, X_p, y = batch
        solver.test_nets[0].set_input_arrays(np.concatenate((X, X_p), axis=0), np.concatenate((y, y), axis=0))
        solver.test_nets[0].forward()
        test_loss += solver.test_nets[0].blobs['loss'].data
        test_acc += solver.test_nets[0].blobs['predict_error'].data
        test_batches += 1
    test_loss = test_loss / test_batches
    test_acc = 1 - 2 * test_acc /test_batches    

    # print the results for this epoch
    print("["+str(datetime.datetime.now().time())+"]  Epoch {} of {} took {:.3f}s - base learning rate: {:.7f}".format(epoch+1, num_epochs, time.time()-start_time, solver_param.base_lr))
    print("    training loss:\t\t{:.6f}".format(train_loss))
    print("    training acc:\t\t{:.6f}".format(train_acc))
    print("    test loss:\t\t{:.6f}".format(test_loss))
    print("    test acc:\t\t{:.6f}".format(test_acc))


print("["+str(datetime.datetime.now().time())+"]  Optimization done. Took {:.3f}s".format(time.time()-start))
