# -*- coding: utf-8 -*-

import paddle.v2 as paddle
import sys
import gzip
import data_provider

# initialize paddle and its trainer
paddle.init(use_gpu=False, trainer_count=1)


def event_handler(event):
    if isinstance(event, paddle.event.EndIteration):
        if event.batch_id % 100 == 0:
            print "\n pass %d, Batch: %d cost: %f, %s" % (event.pass_id, event.batch_id, event.cost, event.metrics)
        else:
            sys.stdout.write('.')
            sys.stdout.flush()
    if isinstance(event, paddle.event.EndPass):
        # save parameters
        feeding = {'image': 0,
                   'label': 1}
        with gzip.open('output/params_pass_%d.tar.gz' % event.pass_id, 'w') as f:
            parameters.to_tar(f)
        filepath = '/Users/vic/Dev/DeepLearning/Paddle/VGG-CIFAR/Images/cifar-10-batches-py/test_batch'
        result = trainer.test(
            reader=paddle.batch(data_provider.data_reader(filepath, 0), batch_size=128),
            feeding=feeding)
        print "\nTest with Pass %d, %s" % (event.pass_id, result.metrics)

image_size = 3 * 32 * 32

image = paddle.layer.data(name='image', type=paddle.data_type.dense_vector(image_size))


num_filters_1 = 64
filter_size = 3
num_channels = 3

conv_part_1 = paddle.networks.img_conv_group(input=image,
                                             num_channels=3,
                                             pool_size=2,
                                             pool_stride=2,
                                             conv_num_filter=[64, 64],
                                             conv_filter_size=3,
                                             conv_with_batchnorm=False,
                                             conv_batchnorm_drop_rate=[0, 0],
                                             conv_act=paddle.activation.Relu(),
                                             pool_type=paddle.pooling.Max()
                                             )

# conv_1 = paddle.layer.img_conv(input=image,
#                                filter_size=filter_size,
#                                num_channels=num_channels,
#                                num_filters=num_filters_1,
#                                stride=1,
#                                padding=1
#                                )
#
# conv_2 = paddle.layer.img_conv(input=conv_1,
#                                filter_size=filter_size,
#                                num_channels=num_channels,
#                                num_filters=num_filters_1,
#                                stride=1,
#                                padding=1
#                                )
#
# pool_1 = paddle.layer.img_pool(input=conv_2,
#                                pool_type=paddle.pooling.Max(),
#                                pool_size=2,
#                                stride=2
#                                )

num_filters_2 = 128
conv_part_2 = paddle.networks.img_conv_group(input=conv_part_1,
                                             num_channels=None,
                                             pool_size=2,
                                             pool_stride=2,
                                             conv_num_filter=[128, 128],
                                             conv_filter_size=3,
                                             conv_with_batchnorm=False,
                                             conv_batchnorm_drop_rate=[0.1, 0.1],
                                             conv_act=paddle.activation.Relu(),
                                             pool_type=paddle.pooling.Max()
                                          )

# conv_3 = paddle.layer.img_conv(input=pool_1,
#                                filter_size=filter_size,
#                                num_channels=num_channels,
#                                num_filters=num_filters_1,
#                                stride=1
#                                )
#
# conv_4 = paddle.layer.img_conv(input=conv_3,
#                                filter_size=filter_size,
#                                num_channels=num_channels,
#                                num_filters=num_filters_1,
#                                stride=1
#                                )
#
# pool_2 = paddle.layer.img_pool(input=conv_4,
#                                pool_type=paddle.pooling.Max(),
#                                pool_size=2,
#                                stride=2
#                                )


num_filters_3 = 256
conv_part_3 = paddle.networks.img_conv_group(input=conv_part_2,
                                             num_channels=None,
                                             pool_size=2,
                                             pool_stride=2,
                                             conv_num_filter=[256, 256],
                                             conv_filter_size=3,
                                             conv_with_batchnorm=False,
                                             conv_batchnorm_drop_rate=[0.1, 0.1],
                                             conv_act=paddle.activation.Relu(),
                                             pool_type=paddle.pooling.Max())
# conv_5 = paddle.layer.img_conv(input=pool_2,
#                                filter_size=filter_size,
#                                num_channels=num_channels,
#                                num_filters=num_filters_1,
#                                stride=1
#                                )
#
# conv_6 = paddle.layer.img_conv(input=conv_5,
#                                filter_size=filter_size,
#                                num_channels=num_channels,
#                                num_filters=num_filters_1,
#                                stride=1,
#                                )
#
# pool_3 = paddle.layer.img_pool(input=conv_6,
#                                pool_type=paddle.pooling.Max(),
#                                pool_size=2,
#                                stride=2
#                                )

#
num_filters_4 = 512
conv_part_4 = paddle.networks.img_conv_group(input=conv_part_3,
                                             num_channels=None,
                                             pool_size=2,
                                             pool_stride=2,
                                             conv_num_filter=[num_filters_4, num_filters_4, num_filters_4],
                                             conv_filter_size=3,
                                             conv_with_batchnorm=True,
                                             conv_batchnorm_drop_rate=[0.1, 0.1, 0.1],
                                             conv_act=paddle.activation.Relu(),
                                             pool_type=paddle.pooling.Max())
# conv_7 = paddle.layer.img_conv(input=pool_3,
#                                filter_size=filter_size,
#                                num_channels=num_channels,
#                                num_filters=num_filters_1,
#                                stride=1
#                                )
#
# conv_8 = paddle.layer.img_conv(input=conv_7,
#                                filter_size=filter_size,
#                                num_channels=num_channels,
#                                num_filters=num_filters_1,
#                                stride=1
#                                )
#
# conv_9 = paddle.layer.img_conv(input=conv_8,
#                                filter_size=filter_size,
#                                num_channels=num_channels,
#                                num_filters=num_filters_1,
#                                stride=1
#                                )
#
# pool_4 = paddle.layer.img_pool(input=conv_9,
#                                pool_type=paddle.pooling.Max(),
#                                pool_size=2,
#                                stride=2
#                                )
#
conv_part_5 = paddle.networks.img_conv_group(input=conv_part_4,
                                             num_channels=None,
                                             pool_size=2,
                                             pool_stride=2,
                                             conv_num_filter=[num_filters_4, num_filters_4, num_filters_4],
                                             conv_filter_size=3,
                                             conv_with_batchnorm=False,
                                             conv_batchnorm_drop_rate=[0.1, 0.1, 0.1],
                                             conv_act=paddle.activation.Relu(),
                                             pool_type=paddle.pooling.Max())

# conv_10 = paddle.layer.img_conv(input=pool_4,
#                                 filter_size=filter_size,
#                                 num_channels=num_channels,
#                                 num_filters=num_filters_1,
#                                 stride=1
#                                 )
#
# conv_11 = paddle.layer.img_conv(input=conv_10,
#                                 filter_size=filter_size,
#                                 num_channels=num_channels,
#                                 num_filters=num_filters_1,
#                                 stride=1
#                                 )
#
# conv_12 = paddle.layer.img_conv(input=conv_11,
#                                 filter_size=filter_size,
#                                 num_channels=num_channels,
#                                 num_filters=num_filters_1,
#                                 stride=1,
#                                 )
#
# pool_5 = paddle.layer.img_pool(input=conv_12,
#                                pool_type=paddle.pooling.Max(),
#                                pool_size=2,
#                                stride=2
#                                )

# def conv_block(ipt, num_filter, groups, dropouts, num_channels=None):
#     return paddle.networks.img_conv_group(
#         input=ipt,
#         num_channels=num_channels,
#         pool_size=2,
#         pool_stride=2,
#         conv_num_filter=[num_filter] * groups,
#         conv_filter_size=3,
#         conv_act=paddle.activation.Relu(),
#         conv_with_batchnorm=True,
#         conv_batchnorm_drop_rate=dropouts,
#         pool_type=paddle.pooling.Max())
#
#
# conv1 = conv_block(image, 64, 2, [0.3, 0], 3)
# conv2 = conv_block(conv1, 128, 2, [0.4, 0])
# conv3 = conv_block(conv2, 256, 3, [0.4, 0.4, 0])
# conv4 = conv_block(conv3, 512, 3, [0.4, 0.4, 0])
# conv5 = conv_block(conv4, 512, 3, [0.4, 0.4, 0])

fc_1 = paddle.layer.fc(input=conv_part_5, size=4096, act=paddle.activation.Sigmoid())

fc_2 = paddle.layer.fc(input=fc_1, size=4096, act=paddle.activation.Sigmoid())

fc_3 = paddle.layer.fc(input=fc_2, size=1000, act=paddle.activation.Sigmoid())

output_layer = paddle.layer.fc(input=fc_3, size=10, act=paddle.activation.Softmax())

label = paddle.layer.data(name='label', type=paddle.data_type.integer_value(10))

cost = paddle.layer.classification_cost(input=output_layer, label=label)

parameters = paddle.parameters.create(cost)

print parameters.keys()

optimizer = paddle.optimizer.Momentum(
    momentum=0.9,
    regularization=paddle.optimizer.L2Regularization(rate=0.0002 * 128),
    learning_rate=0.1 / 128.0,
    learning_rate_decay_a=0.1,
    learning_rate_decay_b=50000 * 100,
    learning_rate_schedule='discexp')

trainer = paddle.trainer.SGD(cost=cost,
                             parameters=parameters,
                             update_equation=optimizer)


feeding = {'image': 0,
           'label': 1}
file_path = '/Users/vic/Dev/DeepLearning/Paddle/VGG-CIFAR/Images/cifar-10-batches-py/data_batch_1'
reader = data_provider.data_reader(file_path, 0)
trainer.train(num_passes=100, reader=paddle.batch(reader, batch_size=16), event_handler=event_handler, feeding=feeding)
