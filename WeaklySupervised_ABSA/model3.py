# The code is completely from the weakly supervised code https://github.com/teapot123/JASen but modified based on vdcnn and doconv (https://github.com/yangyanli/DO-Conv)

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import functional as F
from do_conv_pytorch import DOConv2d

class CNN(nn.Module):
    def __init__(self, batch_size, output_size, in_channels, out_channels, kernel_heights, stride, padding, keep_probab,
                 vocab_size, embedding_length, weights):
        super(CNN, self).__init__()

        """
        Arguments
        ---------
        batch_size : Size of each batch which is same as the batch_size of the data returned by the TorchText BucketIterator
        output_size : 2 = (pos, neg)
        in_channels : Number of input channels. Here it is 1 as the input data has dimension = (batch_size, num_seq, embedding_length)
        out_channels : Number of output channels after convolution operation performed on the input matrix
        kernel_heights : A list consisting of 3 different kernel_heights. Convolution will be performed 3 times and finally results from each kernel_height will be concatenated.
        keep_probab : Probability of retaining an activation node during dropout operation
        vocab_size : Size of the vocabulary containing unique words
        embedding_length : Embedding dimension of GloVe word embeddings
        weights : Pre-trained GloVe word_embeddings which we will use to create our word_embedding look-up table
        --------

        """
        self.batch_size = batch_size
        self.output_size = output_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_heights = kernel_heights
        self.stride = stride
        self.padding = padding
        self.vocab_size = vocab_size
        self.embedding_length = embedding_length

        self.word_embeddings = nn.Embedding(vocab_size, embedding_length, padding_idx=0)
        self.word_embeddings.weight = nn.Parameter(weights, requires_grad=True)
        # self.conv1 = nn.Conv2d(in_channels, out_channels, (kernel_heights[0], embedding_length), stride, padding)
        # self.conv2 = nn.Conv2d(in_channels, out_channels, (kernel_heights[1], embedding_length), stride, padding)
        # self.conv3 = nn.Conv2d(in_channels, out_channels, (kernel_heights[2], embedding_length), stride, padding)
        # DO-CONV based VDCNN with 9 layers
        self.conv1 = DOConv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(kernel_heights[0], embedding_length), stride=stride, padding=padding)
        self.conv2 = DOConv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(kernel_heights[1], embedding_length), stride=stride, padding=padding)
        self.conv3 = DOConv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(kernel_heights[2], embedding_length), stride=stride, padding=padding)
        self.conv4 = DOConv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(kernel_heights[3], embedding_length), stride=stride, padding=padding)
        self.conv5 = DOConv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(kernel_heights[4], embedding_length), stride=stride, padding=padding)
        self.conv6 = DOConv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(kernel_heights[5], embedding_length), stride=stride, padding=padding)
        self.conv7 = DOConv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(kernel_heights[6], embedding_length), stride=stride, padding=padding)
        self.conv8 = DOConv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(kernel_heights[7], embedding_length), stride=stride, padding=padding)
        self.conv9 = DOConv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(kernel_heights[8], embedding_length), stride=stride, padding=padding)

        # DO-CONV based VDCNN with 17 layers
        # self.conv1 = DOConv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(kernel_heights[0], embedding_length), stride=stride, padding=padding)
        # self.conv2 = DOConv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(kernel_heights[1], embedding_length), stride=stride, padding=padding)
        # self.conv3 = DOConv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(kernel_heights[2], embedding_length), stride=stride, padding=padding)
        # self.conv4 = DOConv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(kernel_heights[3], embedding_length), stride=stride, padding=padding)
        # self.conv5 = DOConv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(kernel_heights[4], embedding_length), stride=stride, padding=padding)
        # self.conv6 = DOConv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(kernel_heights[5], embedding_length), stride=stride, padding=padding)
        # self.conv7 = DOConv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(kernel_heights[6], embedding_length), stride=stride, padding=padding)
        # self.conv8 = DOConv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(kernel_heights[7], embedding_length), stride=stride, padding=padding)
        # self.conv9 = DOConv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(kernel_heights[8], embedding_length), stride=stride, padding=padding)
        # self.conv10 = DOConv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(kernel_heights[9], embedding_length), stride=stride, padding=padding)
        # self.conv11 = DOConv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(kernel_heights[10], embedding_length), stride=stride, padding=padding)
        # self.conv12 = DOConv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(kernel_heights[11], embedding_length), stride=stride, padding=padding)
        # self.conv13 = DOConv2d(in_channels=in_channels, out_channels=out_channels,
        #                       kernel_size=(kernel_heights[12], embedding_length), stride=stride, padding=padding)
        # self.conv14 = DOConv2d(in_channels=in_channels, out_channels=out_channels,
        #                       kernel_size=(kernel_heights[13], embedding_length), stride=stride, padding=padding)
        # self.conv15 = DOConv2d(in_channels=in_channels, out_channels=out_channels,
        #                       kernel_size=(kernel_heights[14], embedding_length), stride=stride, padding=padding)
        # self.conv16 = DOConv2d(in_channels=in_channels, out_channels=out_channels,
        #                       kernel_size=(kernel_heights[15], embedding_length), stride=stride, padding=padding)
        # self.conv17 = DOConv2d(in_channels=in_channels, out_channels=out_channels,
        #                       kernel_size=(kernel_heights[16], embedding_length), stride=stride, padding=padding)
        # DO-CONV based VDCNN with 29 layers
        # self.conv1 = DOConv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(kernel_heights[0], embedding_length), stride=stride, padding=padding)
        # self.conv2 = DOConv2d(in_channels=in_channels, out_channels=out_channels,
        #                       kernel_size=(kernel_heights[1], embedding_length), stride=stride, padding=padding)
        # self.conv3 = DOConv2d(in_channels=in_channels, out_channels=out_channels,
        #                       kernel_size=(kernel_heights[2], embedding_length), stride=stride, padding=padding)
        # self.conv4 = DOConv2d(in_channels=in_channels, out_channels=out_channels,
        #                       kernel_size=(kernel_heights[3], embedding_length), stride=stride, padding=padding)
        # self.conv5 = DOConv2d(in_channels=in_channels, out_channels=out_channels,
        #                       kernel_size=(kernel_heights[4], embedding_length), stride=stride, padding=padding)
        # self.conv6 = DOConv2d(in_channels=in_channels, out_channels=out_channels,
        #                       kernel_size=(kernel_heights[5], embedding_length), stride=stride, padding=padding)
        # self.conv7 = DOConv2d(in_channels=in_channels, out_channels=out_channels,
        #                       kernel_size=(kernel_heights[6], embedding_length), stride=stride, padding=padding)
        # self.conv8 = DOConv2d(in_channels=in_channels, out_channels=out_channels,
        #                       kernel_size=(kernel_heights[7], embedding_length), stride=stride, padding=padding)
        # self.conv9 = DOConv2d(in_channels=in_channels, out_channels=out_channels,
        #                       kernel_size=(kernel_heights[8], embedding_length), stride=stride, padding=padding)
        # self.conv10 = DOConv2d(in_channels=in_channels, out_channels=out_channels,
        #                        kernel_size=(kernel_heights[9], embedding_length), stride=stride, padding=padding)
        # self.conv11 = DOConv2d(in_channels=in_channels, out_channels=out_channels,
        #                        kernel_size=(kernel_heights[10], embedding_length), stride=stride, padding=padding)
        # self.conv12 = DOConv2d(in_channels=in_channels, out_channels=out_channels,
        #                        kernel_size=(kernel_heights[11], embedding_length), stride=stride, padding=padding)
        # self.conv13 = DOConv2d(in_channels=in_channels, out_channels=out_channels,
        #                        kernel_size=(kernel_heights[12], embedding_length), stride=stride, padding=padding)
        # self.conv14 = DOConv2d(in_channels=in_channels, out_channels=out_channels,
        #                        kernel_size=(kernel_heights[13], embedding_length), stride=stride, padding=padding)
        # self.conv15 = DOConv2d(in_channels=in_channels, out_channels=out_channels,
        #                        kernel_size=(kernel_heights[14], embedding_length), stride=stride, padding=padding)
        # self.conv16 = DOConv2d(in_channels=in_channels, out_channels=out_channels,
        #                        kernel_size=(kernel_heights[15], embedding_length), stride=stride, padding=padding)
        # self.conv17 = DOConv2d(in_channels=in_channels, out_channels=out_channels,
        #                        kernel_size=(kernel_heights[16], embedding_length), stride=stride, padding=padding)
        # self.conv18 = DOConv2d(in_channels=in_channels, out_channels=out_channels,
        #                       kernel_size=(kernel_heights[17], embedding_length), stride=stride, padding=padding)
        # self.conv19 = DOConv2d(in_channels=in_channels, out_channels=out_channels,
        #                       kernel_size=(kernel_heights[18], embedding_length), stride=stride, padding=padding)
        # self.conv20 = DOConv2d(in_channels=in_channels, out_channels=out_channels,
        #                        kernel_size=(kernel_heights[19], embedding_length), stride=stride, padding=padding)
        # self.conv21 = DOConv2d(in_channels=in_channels, out_channels=out_channels,
        #                        kernel_size=(kernel_heights[20], embedding_length), stride=stride, padding=padding)
        # self.conv22 = DOConv2d(in_channels=in_channels, out_channels=out_channels,
        #                        kernel_size=(kernel_heights[21], embedding_length), stride=stride, padding=padding)
        # self.conv23 = DOConv2d(in_channels=in_channels, out_channels=out_channels,
        #                        kernel_size=(kernel_heights[22], embedding_length), stride=stride, padding=padding)
        # self.conv24 = DOConv2d(in_channels=in_channels, out_channels=out_channels,
        #                        kernel_size=(kernel_heights[23], embedding_length), stride=stride, padding=padding)
        # self.conv25 = DOConv2d(in_channels=in_channels, out_channels=out_channels,
        #                        kernel_size=(kernel_heights[24], embedding_length), stride=stride, padding=padding)
        # self.conv26 = DOConv2d(in_channels=in_channels, out_channels=out_channels,
        #                        kernel_size=(kernel_heights[25], embedding_length), stride=stride, padding=padding)
        # self.conv27 = DOConv2d(in_channels=in_channels, out_channels=out_channels,
        #                        kernel_size=(kernel_heights[26], embedding_length), stride=stride, padding=padding)
        # self.conv28 = DOConv2d(in_channels=in_channels, out_channels=out_channels,
        #                       kernel_size=(kernel_heights[27], embedding_length), stride=stride, padding=padding)
        # self.conv29 = DOConv2d(in_channels=in_channels, out_channels=out_channels,
        #                       kernel_size=(kernel_heights[28], embedding_length), stride=stride, padding=padding)
        self.dropout = nn.Dropout(keep_probab)
        self.label = nn.Linear(len(kernel_heights) * out_channels, output_size, bias=True)
        # self.label.weight = nn.Parameter(topic_repeat, requires_grad=True)
        torch.nn.init.xavier_uniform(self.label.weight, gain=10)

    # print(self.label.weight)

    def conv_block(self, input, conv_layer):
        conv_out = conv_layer(input)  # conv_out.size() = (batch_size, out_channels, dim, 1)
        activation = F.relu(conv_out.squeeze(3))  # activation.size() = (batch_size, out_channels, dim1)
        return F.max_pool1d(activation, activation.size()[2]).squeeze(2)

    def forward(self, input_sentences, batch_size=None):
        """
        The idea of the Convolutional Neural Netwok for Text Classification is very simple. We perform convolution operation on the embedding matrix
        whose shape for each batch is (num_seq, embedding_length) with kernel of varying height but constant width which is same as the embedding_length.
        We will be using ReLU activation after the convolution operation and then for each kernel height, we will use max_pool operation on each tensor
        and will filter all the maximum activation for every channel and then we will concatenate the resulting tensors. This output is then fully connected
        to the output layers consisting two units which basically gives us the logits for both positive and negative classes.

        Parameters
        ----------
        input_sentences: input_sentences of shape = (batch_size, num_sequences)
        batch_size : default = None. Used only for prediction on a single sentence after training (batch_size = 1)

        Returns
        -------
        Output of the linear layer containing logits for pos & neg class.
        logits.size() = (batch_size, output_size)

        """

        input = self.word_embeddings(input_sentences)

        # input.size() = (batch_size, num_seq, embedding_length)
        input = input.unsqueeze(1)
        # input.size() = (batch_size, 1, num_seq, embedding_length)
        max_out1 = self.conv_block(input, self.conv1)
        max_out2 = self.conv_block(input, self.conv2)
        max_out3 = self.conv_block(input, self.conv3)
        max_out4 = self.conv_block(input, self.conv4)
        max_out5 = self.conv_block(input, self.conv5)
        max_out6 = self.conv_block(input, self.conv6)
        max_out7 = self.conv_block(input, self.conv7)
        max_out8 = self.conv_block(input, self.conv8)
        max_out9 = self.conv_block(input, self.conv9)

        #vdcnn17
        # max_out1 = self.conv_block(input, self.conv1)
        # max_out2 = self.conv_block(input, self.conv2)
        # max_out3 = self.conv_block(input, self.conv3)
        # max_out4 = self.conv_block(input, self.conv4)
        # max_out5 = self.conv_block(input, self.conv5)
        # max_out6 = self.conv_block(input, self.conv6)
        # max_out7 = self.conv_block(input, self.conv7)
        # max_out8 = self.conv_block(input, self.conv8)
        # max_out9 = self.conv_block(input, self.conv9)
        # max_out10 = self.conv_block(input, self.conv10)
        # max_out11 = self.conv_block(input, self.conv11)
        # max_out12 = self.conv_block(input, self.conv12)
        # max_out13 = self.conv_block(input, self.conv13)
        # max_out14 = self.conv_block(input, self.conv14)
        # max_out15 = self.conv_block(input, self.conv15)
        # max_out16 = self.conv_block(input, self.conv16)
        # max_out17 = self.conv_block(input, self.conv17)

        #vdcnn29
        # max_out1 = self.conv_block(input, self.conv1)
        # max_out2 = self.conv_block(input, self.conv2)
        # max_out3 = self.conv_block(input, self.conv3)
        # max_out4 = self.conv_block(input, self.conv4)
        # max_out5 = self.conv_block(input, self.conv5)
        # max_out6 = self.conv_block(input, self.conv6)
        # max_out7 = self.conv_block(input, self.conv7)
        # max_out8 = self.conv_block(input, self.conv8)
        # max_out9 = self.conv_block(input, self.conv9)
        # max_out10 = self.conv_block(input, self.conv10)
        # max_out11 = self.conv_block(input, self.conv11)
        # max_out12 = self.conv_block(input, self.conv12)
        # max_out13 = self.conv_block(input, self.conv13)
        # max_out14 = self.conv_block(input, self.conv14)
        # max_out15 = self.conv_block(input, self.conv15)
        # max_out16 = self.conv_block(input, self.conv16)
        # max_out17 = self.conv_block(input, self.conv17)
        # max_out18 = self.conv_block(input, self.conv18)
        # max_out19 = self.conv_block(input, self.conv19)
        # max_out20 = self.conv_block(input, self.conv20)
        # max_out21 = self.conv_block(input, self.conv21)
        # max_out22 = self.conv_block(input, self.conv22)
        # max_out23 = self.conv_block(input, self.conv23)
        # max_out24 = self.conv_block(input, self.conv24)
        # max_out25 = self.conv_block(input, self.conv25)
        # max_out26 = self.conv_block(input, self.conv26)
        # max_out27 = self.conv_block(input, self.conv27)
        # max_out28 = self.conv_block(input, self.conv28)
        # max_out29 = self.conv_block(input, self.conv29)

        all_out = torch.cat((max_out1, max_out2, max_out3, max_out4, max_out5, max_out6, max_out7, max_out8, max_out9), 1)
        #all_out = torch.cat((max_out1, max_out2, max_out3, max_out4, max_out5, max_out6, max_out7, max_out8, max_out9, max_out10, max_out11, max_out12, max_out13, max_out14, max_out15, max_out16, max_out17), 1)
        #all_out = torch.cat((max_out1, max_out2, max_out3, max_out4, max_out5, max_out6, max_out7, max_out8, max_out9, max_out10, max_out11, max_out12, max_out13, max_out14, max_out15, max_out16, max_out17, max_out18, max_out19, max_out20, max_out21, max_out22, max_out23, max_out24, max_out25, max_out26, max_out27, max_out28, max_out29), 1)

        # all_out.size() = (batch_size, num_kernels*out_channels)
        fc_in = self.dropout(all_out)
        return self.label(fc_in)