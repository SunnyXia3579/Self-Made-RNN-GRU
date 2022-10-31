import numpy as np
import matplotlib.pyplot as plt


class Rnn:
    # constructor: instance value for Rnn instance
    def __init__(self):
        # constant value--dimension
        self.m = 1                                                             # dimension of output vector
        self.n = 30                                                            # dimension of vector of hidden state
        self.p = 1                                                             # dimension of input vector
        self.step = 10                                                         # maximum length of sequence
        self.batch_size = 32
        self.pre_step = 1                                                      # maximum length of prediction sequence
        self.initial = 1.52
        # other value--parameter matrix
        self.v = np.random.uniform(-self.initial, self.initial, (self.m, self.n))                #
        self.w = np.random.uniform(-self.initial/self.n, self.initial/self.n, (self.n, self.n))  #
        self.u = np.random.uniform(-self.initial, self.initial, (self.n, self.p))                #
        self.rnn_cells = [Rnn.RnnCell(self)]                                   # Rnn cell array of Rnn instance
        self.losses = np.empty(0)                                              # loss of each prediction of Rnn instance
        self.input_mat = np.zeros((0, 0))                                      # matrix containing input vectors
        self.output_mat = np.zeros((0, 0))                                     # matrix containing output vectors
        self.input_length = 0                                                  #
        self.output_length = 0                                                 # defined by user
        self.output_start = 0                                                  # defined by user, default 1
        self.train_time = 3000                                                 #
        self.learning_rate = 0.001                                             #

    class RnnCell:
        def __init__(self, rnn_obj):
            self.ot = np.zeros((rnn_obj.m, 1))                                 # output vector
            self.st = np.zeros((rnn_obj.n, 1))                                 # hidden vector
            self.xt = np.zeros((rnn_obj.p, 1))                                 # input vector
            self.yt = self.ot                                                  # actual result
            self.ot0 = self.ot                                                 # output vector before softmax activation
            self.st0 = self.st                                                 # hidden vector before sigmoid activation
            self.lt_partial_u = np.zeros_like(rnn_obj.u)                       # contribution to gradient of u
            self.lt_partial_w = np.zeros_like(rnn_obj.w)                       # contribution to gradient of w
            self.lt_partial_v = np.zeros_like(rnn_obj.v)                       # contribution to gradient of v
            self.st0_partial_u = np.zeros((rnn_obj.n, rnn_obj.n * rnn_obj.p))  # iteration
            self.st0_partial_w = np.zeros((rnn_obj.n, rnn_obj.n * rnn_obj.n))  # iteration

        # calculation method for RnnCell
        def cross_entropy_loss(self):
            return

        def mse_loss(self):                                     # input is a vector, return is a vector
            return np.sum((self.ot - self.yt) ** 2)

        def partial_mse(self):                                  # input is a vector, return is a vector
            return 2*(self.ot-self.yt)

        def sigmoid(self):                                      # input is a vector, return is a vector
            return np.ones_like(self.st0) / (np.ones_like(self.st0) + np.exp(-self.st0))

        def partial_sigmoid(self):                              # input is a vector, return is a vector
            return self.st * (np.ones_like(self.st) - self.st)

        def tanh(self):
            return (np.exp(self.st0) - np.exp(-self.st0)) / (np.exp(self.st0) + np.exp(-self.st0))

        def partial_tanh(self):
            return np.ones_like(self.st) - self.st ** 2

        def softmax(self):                                      # input is a vector, return is a vector
            #   x=x-np.max(x)                                   # compute in a stable way # hard to differentiate
            return np.exp(self.ot0) / np.sum(np.exp(self.ot0))

        def partial_softmax(self):                              # input is a vector, return is a matrix
            return np.diag(self.ot) - self.ot @ self.ot.transpose()

    # training method for Rnn instance
    def get_data(self, data):
        self.input_mat = np.reshape(data, (self.p, np.size(data)))
        return np.size(data)

    def forward_propagate(self):
        for i in np.arange(1, self.batch_size+1, 1):
            self.rnn_cells[i].st0 = self.u @ self.rnn_cells[i].xt + self.w @ self.rnn_cells[i-1].st
            self.rnn_cells[i].st = self.rnn_cells[i].tanh()

        # rnn_cell[self.step] need to be specially tackled
        self.rnn_cells[self.batch_size].ot0 = self.v @ self.rnn_cells[self.batch_size].st
        self.rnn_cells[self.batch_size].ot = self.rnn_cells[self.batch_size].ot0
        lose = self.rnn_cells[self.batch_size].mse_loss()

        for i in np.arange(self.batch_size+1, self.batch_size+self.pre_step, 1):
            self.rnn_cells[i].xt = self.rnn_cells[i-1].ot
            self.rnn_cells[i].st0 = self.u @ self.rnn_cells[i].xt + self.w @ self.rnn_cells[i-1].st
            self.rnn_cells[i].st = self.rnn_cells[i].tanh()
            self.rnn_cells[i].ot0 = self.v @ self.rnn_cells[i].st
            self.rnn_cells[i].ot = self.rnn_cells[i].ot0      # no activation in output layer
            lose = lose + self.rnn_cells[i].mse_loss()
        return lose

    def back_propagate(self):
        du = np.zeros_like(self.u)  # initialize du, dw, dv
        dw = np.zeros_like(self.w)
        dv = np.zeros_like(self.v)
        for i in np.arange(self.batch_size-self.step, self.batch_size+self.pre_step, 1):
            lt_partial_ot = self.rnn_cells[i].partial_mse()
            lt_partial_st0 = (self.v.transpose() @ lt_partial_ot) * self.rnn_cells[i].partial_tanh()
            self.rnn_cells[i].st0_partial_u = \
                np.kron(np.eye(self.n), self.rnn_cells[i].xt.transpose()) + self.w @ \
                (np.kron(np.ones((1, self.n * self.p)), self.rnn_cells[i].partial_tanh()) *
                 self.rnn_cells[i-1].st0_partial_u)
            self.rnn_cells[i].st0_partial_w = \
                np.kron(np.eye(self.n), self.rnn_cells[i-1].st.transpose()) + self.w @ \
                (np.kron(np.ones((1, self.n * self.n)), self.rnn_cells[i-1].partial_tanh()) *
                 self.rnn_cells[i-1].st0_partial_w)
            self.rnn_cells[i].lt_partial_u = np.reshape(lt_partial_st0.transpose() @ self.rnn_cells[i].st0_partial_u,
                                                        (self.n, self.p))
            self.rnn_cells[i].lt_partial_w = np.reshape(lt_partial_st0.transpose() @ self.rnn_cells[i].st0_partial_w,
                                                        (self.n, self.n))
            self.rnn_cells[i].lt_partial_v = lt_partial_ot @ self.rnn_cells[i].st.transpose()
            du = du + self.rnn_cells[i].lt_partial_u
            dw = dw + self.rnn_cells[i].lt_partial_w
            dv = dv + self.rnn_cells[i].lt_partial_v

        self.u = self.u - self.learning_rate * self.rnn_cells[self.batch_size].lt_partial_u
        self.w = self.w - self.learning_rate * self.rnn_cells[self.batch_size].lt_partial_w
        self.v = self.v - self.learning_rate * self.rnn_cells[self.batch_size].lt_partial_v
        return

    def train(self, data):
        # process input data and initialize array of Rnn cells as well as array of losses
        self.input_length = self.get_data(data)
        self.losses = np.empty(self.train_time)

        # self.train_time is times of training
        # in each time of training, randomly choose a batch of self.step samples
        # create an array of self.step Rnn cells
        # forward propagate and back_propagate respectively
        for i in np.arange(0, self.train_time, 1):
            self.rnn_cells = [Rnn.RnnCell(self)] * (self.batch_size+self.pre_step)       # create an array of Rnn cell
            begin = np.random.randint(1, self.input_length-self.batch_size-self.pre_step+1)  # start of training step

            print(i, ": begin: ", begin)

            for j in np.arange(begin, begin+self.batch_size, 1):               # initialize input for training network
                self.rnn_cells[j-begin+1].xt = np.reshape(self.input_mat[:, j-1], (self.p, 1))
            for j in np.arange(begin, begin+self.batch_size+self.pre_step-1, 1):
                self.rnn_cells[j-begin+1].yt = np.reshape(self.input_mat[:, j], (self.p, 1))
            self.losses[i] = self.forward_propagate()
            self.back_propagate()
        return

    # predicting method for Rnn instance
    def predict(self, output_start, output_length):
        self.output_start = output_start
        self.output_length = output_length
        if self.output_start > self.input_length:
            Exception("Start of prediction > input_length!")
            return
        self.output_mat = np.zeros((self.m, self.output_length))
        self.rnn_cells = [Rnn.RnnCell(self)] * (self.output_start+self.output_length)
        for i in np.arange(1, self.output_start+1, 1):                     # initialize input for training network
            self.rnn_cells[i].xt = np.reshape(self.input_mat[:, i-1], (self.p, 1))
            self.rnn_cells[i].st0 = self.u @ self.rnn_cells[i].xt + self.w @ self.rnn_cells[i-1].st
            self.rnn_cells[i].st = self.rnn_cells[i].tanh()

            # if i==2:
            #     print(self.u, "\n")
            #     print(self.w, "\n")
            #     print(self.rnn_cells[i].xt, "\n")
            #     print(self.rnn_cells[i].st0)

        self.rnn_cells[self.output_start].ot0 = self.v @ self.rnn_cells[self.output_start].st
        self.rnn_cells[self.output_start].ot = self.rnn_cells[self.output_start].ot0
        self.output_mat[:, 0] = np.reshape(self.rnn_cells[self.output_start].ot, self.m)

        # print(self.rnn_cells[self.output_start].st)
        # print(self.rnn_cells[self.output_start].ot)

        for i in np.arange(self.output_start+1, self.output_start+self.output_length, 1):
            self.rnn_cells[i].xt = self.rnn_cells[i-1].ot

            # print(self.rnn_cells[i].st)

            self.rnn_cells[i].st0 = self.u @ self.rnn_cells[i].xt + self.w @ self.rnn_cells[i-1].st
            self.rnn_cells[i].st = self.rnn_cells[i].tanh()
            self.rnn_cells[i].ot0 = self.v @ self.rnn_cells[i].st
            self.rnn_cells[i].ot = self.rnn_cells[i].ot0

            # print(self.rnn_cells[i].ot)

            self.output_mat[:, i-self.output_start] = np.reshape(self.rnn_cells[i].ot, self.m)
        return self.output_mat


if __name__ == '__main__':
    sin_data = np.sin(np.arange(0, 101, 0.8))
    cos_data = np.cos(np.arange(0, 101, 1))
    pow_data = np.arange(0, 101, 1) ** 2
    line_data = np.arange(0, 101, 1)
    pow_3_4_data = np.arange(0, 101, 1) ** 0.75
    sqrt_data = np.sqrt(np.arange(0, 101, 1))
    log_data = np.log(np.arange(1, 102, 1))
    con_data = np.ones(101)
    # dec_data = np.exp(-np.ones_like(np.arange(0, 101, 1))) * sin_data
    rnn = Rnn()
    rnn.train(line_data)
    # plt.plot(np.arange(0, np.size(sin_data), 1), sin_data)
    # fig1 = plt.plot(np.arange(0, np.size(rnn.losses), 1), rnn.losses)
    # print(rnn.u)
    # print(rnn.w)
    # print(rnn.v)
    # plt.show()
    output = np.reshape(rnn.predict(100, 3), 3)
    fig2 = plt.plot(np.arange(0, np.size(output), 1), output)
    print(np.sin(100))
    plt.show()
    pass
