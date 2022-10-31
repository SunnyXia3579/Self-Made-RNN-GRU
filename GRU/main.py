import numpy as np
import matplotlib.pyplot as plt
from numba import jit


class Gru:
    # constructor: instance value for Rnn instance
    def __init__(self):
        # constant value--dimension
        self.m = 1  # dimension of output vector
        self.n = 32  # dimension of vector of hidden state
        self.p = 1  # dimension of input vector
        self.initial = 3.8
        # other value--parameter matrix
        self.v = np.random.uniform(-self.initial, self.initial, (self.m, self.n))  #
        self.wh = np.random.uniform(-self.initial / self.n, self.initial / self.n, (self.n, self.n))  #
        self.wr = np.random.uniform(-self.initial / self.n, self.initial / self.n, (self.n, self.n))  #
        self.wz = np.random.uniform(-self.initial / self.n, self.initial / self.n, (self.n, self.n))  #
        self.uh = np.random.uniform(-self.initial, self.initial, (self.n, self.p))  #
        self.ur = np.random.uniform(-self.initial, self.initial, (self.n, self.p))  #
        self.uz = np.random.uniform(-self.initial, self.initial, (self.n, self.p))  #
        self.bh = np.random.uniform(-self.initial, self.initial, (self.n, 1))
        self.br = np.random.uniform(-self.initial, self.initial, (self.n, 1))
        self.bz = np.random.uniform(-self.initial, self.initial, (self.n, 1))
        self.bo = np.random.uniform(-self.initial, self.initial, (self.m, 1))
        self.input_mat = np.zeros((self.p, 1))

    class GruCell:
        def __init__(self, gru_obj):
            self.ot = np.zeros((gru_obj.m, 1))                                 # output vector
            self.yt = np.zeros((gru_obj.m, 1))                                 # actual result
            self.st = np.zeros((gru_obj.n, 1))                                 # hidden vector
            self.ht = np.zeros((gru_obj.n, 1))
            self.zt = np.zeros((gru_obj.n, 1))
            self.rt = np.zeros((gru_obj.n, 1))
            self.xt = np.zeros((gru_obj.p, 1))                                 # input vector

        # calculation method for RnnCell
        def cross_entropy_loss(self):
            return

        # @jit
        def mse_loss(self):                                     # input is a vector, return is a vector
            return np.sum((self.ot - self.yt) ** 2)

        # @jit
        def p_mse(self):                                  # input is a vector, return is a vector
            return 2*(self.ot-self.yt).transpose()

        @staticmethod
        @jit(nopython=True)
        def sigmoid(x):                                   # input is a vector, return is a vector
            return np.ones_like(x) / (np.ones_like(x) + np.exp(-x))

        @staticmethod
        @jit(nopython=True)
        def p_sigmoid(y):                           # input is a vector, return is a vector
            return y * (np.ones_like(y) - y)

        @staticmethod
        @jit(nopython=True)
        def tanh(x):
            return np.tanh(x)

        @staticmethod
        @jit(nopython=True)
        def p_tanh(y):
            return np.ones_like(y) - y ** 2

    # training method for Rnn instance
    # @jit
    def get_data(self, data):
        return np.reshape(data, (self.p, np.size(data)))

    # @jit
    def forward_propagate(self, time_step, begin, pre_step, op):
        gru_cell = [Gru.GruCell(self)] * (time_step + pre_step)
        for i in np.arange(1, time_step+1, 1):
            gru_cell[i].xt = np.reshape(self.input_mat[:,begin+i-2], (self.p, 1))
            gru_cell[i].yt = np.reshape(self.input_mat[:,begin+i-1], (self.m, 1))
            gru_cell[i].ht = Gru.GruCell.tanh(self.uh @ gru_cell[i].xt + self.wh @ gru_cell[i-1].st + self.bh)
            gru_cell[i].rt = Gru.GruCell.sigmoid(self.ur @ gru_cell[i].xt + self.wr @ gru_cell[i-1].st + self.br)
            gru_cell[i].zt = Gru.GruCell.sigmoid(self.uz @ gru_cell[i].xt + self.wz @ gru_cell[i-1].st + self.bz)
            gru_cell[i].st = (1-gru_cell[i].zt) * gru_cell[i].ht + gru_cell[i].zt * gru_cell[i-1].st
        gru_cell[time_step].ot = self.v @ gru_cell[time_step].st + self.bo
        if op == "train":
            return [gru_cell[time_step].mse_loss(), gru_cell]
        output_mat = np.zeros((self.m, pre_step))
        output_mat[:,0] = np.reshape(gru_cell[time_step].ot, self.m)

        print(gru_cell[time_step].ot)

        for i in np.arange(time_step+1, time_step+pre_step, 1):
            gru_cell[i].xt = gru_cell[i-1].ot
            gru_cell[i].ht = Gru.GruCell.tanh(self.uh @ gru_cell[i].xt + self.wh @ gru_cell[i-1].st + self.bh)
            gru_cell[i].rt = Gru.GruCell.sigmoid(self.ur @ gru_cell[i].xt + self.wr @ gru_cell[i-1].st + self.br)
            gru_cell[i].zt = Gru.GruCell.sigmoid(self.uz @ gru_cell[i].xt + self.wz @ gru_cell[i-1].st + self.bz)
            gru_cell[i].st = (1 - gru_cell[i].zt) * gru_cell[i].ht + gru_cell[i].zt * gru_cell[i-1].st
            gru_cell[i].ot = self.v @ gru_cell[i].st + self.bo

            # print(self.input_mat)
            # print(self.uh @ gru_cell[i].xt + self.wh @ gru_cell[i-1].st + self.bh)

            output_mat[:,i-time_step] = np.reshape(gru_cell[i].ot, self.m)
        return output_mat

    # @jit
    def back_propagate(self, gru_cell, pre_step, learning_rate, precision):
        length = np.size(gru_cell)
        time_step = length-pre_step
        st_p_uh = [np.zeros((self.n, self.n*self.p))] * length
        st_p_ur = [np.zeros((self.n, self.n*self.p))] * length
        st_p_uz = [np.zeros((self.n, self.n*self.p))] * length
        st_p_wh = [np.zeros((self.n, self.n*self.n))] * length
        st_p_wr = [np.zeros((self.n, self.n*self.n))] * length
        st_p_wz = [np.zeros((self.n, self.n*self.n))] * length
        st_p_bh = [np.zeros((self.n, self.n))] * length
        st_p_br = [np.zeros((self.n, self.n))] * length
        st_p_bz = [np.zeros((self.n, self.n))] * length
        duh = np.zeros_like(self.uh)
        dur = np.zeros_like(self.ur)
        duz = np.zeros_like(self.uz)
        dwh = np.zeros_like(self.wh)
        dwr = np.zeros_like(self.wr)
        dwz = np.zeros_like(self.wz)
        dbh = np.zeros_like(self.bh)
        dbr = np.zeros_like(self.br)
        dbz = np.zeros_like(self.bz)
        dbo = np.zeros_like(self.bo)
        dv = np.zeros_like(self.v)
        for i in np.arange(1, length, 1):
            lt_p_ot = gru_cell[i].p_mse()
            lt_p_st = lt_p_ot @ self.v
            st_p_st_1 = np.diag(np.reshape(gru_cell[i].zt, self.n))
            st_p_ht = np.diag(np.reshape(1-gru_cell[i].zt, self.n))
            st_p_zt = np.diag(np.reshape(-gru_cell[i].ht + gru_cell[i-1].st, self.n))
            ht_p_st_1 = np.kron(np.ones((1, self.n)), Gru.GruCell.p_tanh(gru_cell[i].ht)) * \
                        np.kron(np.ones((1, self.n)), gru_cell[i].rt) * self.wh
            ht_p_rt = np.kron(np.ones((1, self.n)), Gru.GruCell.p_tanh(gru_cell[i].ht)) * \
                      np.kron(np.ones((1, self.n)), gru_cell[i-1].st) * self.wh
            rt_p_st_1 = np.kron(np.ones((1, self.n)), Gru.GruCell.p_sigmoid(gru_cell[i].rt)) * self.wr
            zt_p_st_1 = np.kron(np.ones((1, self.n)), Gru.GruCell.p_sigmoid(gru_cell[i].zt)) * self.wz
            ht_p_uh = np.kron(np.ones((1, self.n*self.p)), Gru.GruCell.p_tanh(gru_cell[i].ht)) * \
                      np.kron(np.eye(self.n), gru_cell[i].xt.transpose())
            rt_p_ur = np.kron(np.ones((1, self.n*self.p)), Gru.GruCell.p_sigmoid(gru_cell[i].rt)) * \
                      np.kron(np.eye(self.n), gru_cell[i].xt.transpose())
            zt_p_uz = np.kron(np.ones((1, self.n*self.p)), Gru.GruCell.p_sigmoid(gru_cell[i].zt)) * \
                      np.kron(np.eye(self.n), gru_cell[i].xt.transpose())
            ht_p_wh = np.kron(np.ones((1, self.n*self.n)), Gru.GruCell.p_tanh(gru_cell[i].ht)) * \
                      np.kron(np.eye(self.n), (gru_cell[i].rt * gru_cell[i-1].st).transpose())
            rt_p_wr = np.kron(np.ones((1, self.n*self.n)), Gru.GruCell.p_sigmoid(gru_cell[i].rt)) * \
                      np.kron(np.eye(self.n), gru_cell[i-1].st.transpose())
            zt_p_wz = np.kron(np.ones((1, self.n*self.n)), Gru.GruCell.p_sigmoid(gru_cell[i].zt)) * \
                      np.kron(np.eye(self.n), gru_cell[i-1].st.transpose())
            ht_p_bh = np.diag(np.reshape(Gru.GruCell.p_tanh(gru_cell[i].ht), self.n))
            rt_p_br = np.diag(np.reshape(Gru.GruCell.p_sigmoid(gru_cell[i].rt), self.n))
            zt_p_bz = np.diag(np.reshape(Gru.GruCell.p_sigmoid(gru_cell[i].zt), self.n))

            st_p_uh[i] = st_p_st_1 @ st_p_uh[i-1] + \
                         st_p_ht @ (ht_p_uh + ht_p_st_1 @ st_p_uh[i-1] + ht_p_rt @ rt_p_st_1 @ st_p_uh[i-1]) + \
                         st_p_zt @ zt_p_st_1 @ st_p_uh[i-1]
            st_p_ur[i] = st_p_st_1 @ st_p_ur[i-1] + \
                         st_p_ht @ (ht_p_st_1 @ st_p_ur[i-1] + ht_p_rt @ (rt_p_st_1 @ st_p_ur[i-1] + rt_p_ur)) + \
                         st_p_zt @ zt_p_st_1 @ st_p_ur[i-1]
            st_p_uz[i] = st_p_st_1 @ st_p_uz[i-1] + \
                         st_p_ht @ (ht_p_st_1 @ st_p_uz[i-1] + ht_p_rt @ rt_p_st_1 @ st_p_uz[i-1]) + \
                         st_p_zt @ (zt_p_st_1 @ st_p_uz[i-1] + zt_p_uz)
            st_p_wh[i] = st_p_st_1 @ st_p_wh[i-1] + \
                         st_p_ht @ (ht_p_wh + ht_p_st_1 @ st_p_wh[i-1] + ht_p_rt @ rt_p_st_1 @ st_p_wh[i-1]) + \
                         st_p_zt @ zt_p_st_1 @ st_p_wh[i-1]
            st_p_wr[i] = st_p_st_1 @ st_p_wr[i-1] + \
                         st_p_ht @ (ht_p_st_1 @ st_p_wr[i-1] + ht_p_rt @ rt_p_wr + ht_p_rt @ rt_p_st_1 @ st_p_wr[i-1]) + \
                         st_p_zt @ zt_p_st_1 @ st_p_wr[i-1]
            st_p_wz[i] = st_p_st_1 @ st_p_wz[i-1] + \
                         st_p_ht @ (ht_p_st_1 @ st_p_wz[i-1] + ht_p_rt @ rt_p_st_1 @ st_p_wz[i-1]) + \
                         st_p_zt @ (zt_p_wz + zt_p_st_1 @ st_p_wz[i-1])
            st_p_bh[i] = st_p_st_1 @ st_p_bh[i-1] + \
                         st_p_ht @ (ht_p_st_1 @ st_p_bh[i-1] + ht_p_bh + ht_p_rt @ rt_p_st_1 @ st_p_bh[i-1]) + \
                         st_p_zt @ zt_p_st_1 @ st_p_bh[i-1]
            st_p_br[i] = st_p_st_1 @ st_p_br[i-1] + \
                         st_p_ht @ (ht_p_st_1 @ st_p_br[i-1] + ht_p_rt @ rt_p_br + ht_p_rt @ rt_p_st_1 @ st_p_br[i-1]) + \
                         st_p_zt @ zt_p_st_1 @ st_p_br[i-1]
            st_p_bz[i] = st_p_st_1 @ st_p_bz[i-1] + \
                         st_p_ht @ (ht_p_st_1 @ st_p_bz[i-1] + ht_p_rt @ rt_p_st_1 @ st_p_bz[i-1]) + \
                         st_p_zt @ (zt_p_bz + zt_p_st_1 @ st_p_bz[i-1])

            if i >= time_step:
                dv = dv + lt_p_ot.transpose() @ gru_cell[i].st.transpose()
                dbo = dbo + lt_p_ot.transpose()
                duh = duh + np.reshape(lt_p_st @ st_p_uh[i], (self.n, self.p))
                dur = dur + np.reshape(lt_p_st @ st_p_ur[i], (self.n, self.p))
                duz = duz + np.reshape(lt_p_st @ st_p_uz[i], (self.n, self.p))
                dwh = dwh + np.reshape(lt_p_st @ st_p_wh[i], (self.n, self.n))
                dwr = dwr + np.reshape(lt_p_st @ st_p_wr[i], (self.n, self.n))
                dwz = dwz + np.reshape(lt_p_st @ st_p_wz[i], (self.n, self.n))
                dbh = dbh + np.reshape(lt_p_st @ st_p_bh[i], (self.n, 1))
                dbr = dbr + np.reshape(lt_p_st @ st_p_br[i], (self.n, 1))
                dbz = dbz + np.reshape(lt_p_st @ st_p_bz[i], (self.n, 1))

        print(max(np.max(duh), np.max(dur), np.max(duz), np.max(dwh), np.max(dwr), np.max(dwz), np.max(dbh), np.max(dbr), np.max(dbz), np.max(dv), np.max(dbo)))

        # if np.all(np.abs(duh) < precision) and \
        #    np.all(np.abs(dur) < precision) and \
        #    np.all(np.abs(duz) < precision) and \
        #    np.all(np.abs(dwh) < precision / self.n) and \
        #    np.all(np.abs(dwr) < precision / self.n) and \
        #    np.all(np.abs(dwz) < precision / self.n) and \
        #    np.all(np.abs(dbh) < precision) and \
        #    np.all(np.abs(dbr) < precision) and \
        #    np.all(np.abs(dbz) < precision) and \
        #    np.all(np.abs(dv) < precision) and \
        #    np.all(np.abs(dbo) < precision):
        #     return 0
        self.uh = self.uh - learning_rate * duh
        self.ur = self.ur - learning_rate * dur
        self.uz = self.uz - learning_rate * duz
        self.wh = self.wh - learning_rate * dwh
        self.wr = self.wr - learning_rate * dwr
        self.wz = self.wz - learning_rate * dwz
        self.v = self.v - learning_rate * dv
        self.bh = self.bh - learning_rate * dbh
        self.br = self.br - learning_rate * dbr
        self.bz = self.bz - learning_rate * dbz
        self.bo = self.bo - learning_rate * dbo
        return 1

    # @jit
    def train(self, data, learning_rate, precision, train_time, time_step, pre_step):
        self.input_mat = self.get_data(data)
        input_length = np.size(data)
        losses = np.empty(train_time)

        # in each time of training, randomly choose a batch of self.step samples
        # create an array of self.step Rnn cells
        # forward propagate and back_propagate respectively
        for i in np.arange(0, train_time, 1):
            begin = np.random.randint(1, input_length-time_step-pre_step+1)  # start of training step

            print(i, ": begin: ", begin)

            [losses[i], gru_cell] = self.forward_propagate(time_step, begin, pre_step, "train")
            if self.back_propagate(gru_cell, pre_step, learning_rate, precision) == 0:
                break
        return losses

    # @jit
    def predict(self, data, output_start, output_length):
        self.input_mat = self.get_data(data)
        input_length = np.size(data)
        if output_start >= input_length:
            Exception("Start of prediction >= input_length!")
            return
        return self.forward_propagate(output_start, 1, output_length, "predict")


if __name__ == '__main__':
    precision = 0.2
    output_start = 300
    sin_data = np.sin(np.arange(precision, 100+precision, precision))
    cos_data = np.cos(3*np.arange(precision, 100+precision, precision))
    s_c_data = sin_data * np.abs(cos_data)
    pow_data = np.arange(0, 101, 1) ** 2
    line_data = np.arange(precision, 100+precision, precision)
    pow_3_4_data = np.arange(0, 101, 1) ** 0.75
    sqrt_data = np.sqrt(np.arange(0, 101, 1))
    log_data = np.log(np.arange(1, 102, 1))
    con_data = np.ones(101)
    # dec_data = np.exp(-np.ones_like(np.arange(0, 101, 1))) * sin_data
    gru = Gru()
    losses = gru.train(sin_data, learning_rate=0.001, precision=0.0001, train_time=2000, time_step=32, pre_step=10)
    # plt.plot(np.arange(0, np.size(sin_data), 1), sin_data)
    # fig1 = plt.plot(np.arange(0, np.size(losses), 1), losses)

    print(sin_data[output_start])

    output = np.reshape(gru.predict(sin_data, output_start=output_start, output_length=100), 100)
    fig2 = plt.plot(np.arange(output_start*precision, output_start*precision+np.size(output)*precision, precision), output)
    fig3 = plt.plot(np.arange(output_start*precision, output_start*precision+np.size(output)*precision, precision),
                    np.sin(np.arange((output_start+1)*precision, (output_start+1+np.size(output))*precision, precision)))# * \
                #    np.abs(np.cos(3*np.arange((output_start+1)*precision, (output_start+1+np.size(output))*precision, precision))))
    plt.show()

    pass
