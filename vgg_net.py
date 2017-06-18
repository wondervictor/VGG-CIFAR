# -*- coding: utf-8 -*-

import paddle.v2 as paddle
import sys
import gzip

# initialize paddle and its trainer
paddle.init(use_gpu=False, trainer_count=2)


def event_handler(event):
    if isinstance(event, paddle.event.EndIteration):
        if event.batch_id % 100 == 0:
            print "\n pass %d, Batch: %d cost: %f, %s" % (event.pass_id, event.batch_id, event.cost, event.metrics)
        else:
            sys.stdout.write('.')
            sys.stdout.flush()
    if isinstance(event, paddle.event.EndPass):
        # save parameters
        with gzip.open('output/params_pass_%d.tar.gz' % event.pass_id, 'w') as f:
            parameters.to_tar(f)

        # result = trainer.test(
        #     reader=paddle.batch(
        #         paddle.dataset.cifar.test10(), batch_size=128),
        #     feeding=feeding)
        # print "\nTest with Pass %d, %s" % (event.pass_id, result.metrics)

image_size = 3 * 32 * 32

image = paddle.layer.data(name='image', type=paddle.data_type.dense_vector(image_size))


num_filters_1 = 64
filter_size = 3
num_channels = 3


conv_1 = paddle.layer.img_conv(input=image,
                               filter_size=filter_size,
                               num_channels=num_channels,
                               num_filters=num_filters_1,
                               stride=1,
                               )

conv_2 = paddle.layer.img_conv(conv_1,
                               filter_size=filter_size,
                               num_channels=num_channels,
                               num_filters=num_filters_1,
                               stride=1,
                               )

pool_1 = paddle.layer.img_pool(input=conv_2,
                               pool_type=paddle.pooling.Max(),
                               pool_size=2,
                               stride=2
                               )

num_filters_2 = 128
conv_3 = paddle.layer.img_conv(input=pool_1,
                               filter_size=filter_size,
                               num_channels=num_channels,
                               num_filters=num_filters_1,
                               stride=1,
                               )

conv_4 = paddle.layer.img_conv(input=conv_3,
                               filter_size=filter_size,
                               num_channels=num_channels,
                               num_filters=num_filters_1,
                               stride=1,
                               )

pool_2 = paddle.layer.img_pool(input=conv_4,
                               pool_type=paddle.pooling.Max(),
                               pool_size=2,
                               stride=2
                               )


num_filters_3 = 256
conv_5 = paddle.layer.img_conv(input=pool_2,
                               filter_size=filter_size,
                               num_channels=num_channels,
                               num_filters=num_filters_1,
                               stride=1,
                               )

conv_6 = paddle.layer.img_conv(input=conv_5,
                               filter_size=filter_size,
                               num_channels=num_channels,
                               num_filters=num_filters_1,
                               stride=1,
                               )

pool_3 = paddle.layer.img_pool(input=conv_6,
                               pool_type=paddle.pooling.Max(),
                               pool_size=2,
                               stride=2
                               )


num_filters_4 = 512
conv_7 = paddle.layer.img_conv(input=pool_3,
                               filter_size=filter_size,
                               num_channels=num_channels,
                               num_filters=num_filters_1,
                               stride=1,
                               )

conv_8 = paddle.layer.img_conv(input=conv_7,
                               filter_size=filter_size,
                               num_channels=num_channels,
                               num_filters=num_filters_1,
                               stride=1,
                               )

conv_9 = paddle.layer.img_conv(input=conv_8,
                               filter_size=filter_size,
                               num_channels=num_channels,
                               num_filters=num_filters_1,
                               stride=1,
                               )

pool_4 = paddle.layer.img_pool(input=conv_9,
                               pool_type=paddle.pooling.Max(),
                               pool_size=2,
                               stride=2
                               )


conv_10 = paddle.layer.img_conv(input=pool_4,
                                filter_size=filter_size,
                                num_channels=num_channels,
                                num_filters=num_filters_1,
                                stride=1,
                                )

conv_11 = paddle.layer.img_conv(input=conv_10,
                                filter_size=filter_size,
                                num_channels=num_channels,
                                num_filters=num_filters_1,
                                stride=1,
                                )

conv_12 = paddle.layer.img_conv(input=conv_11,
                                filter_size=filter_size,
                                num_channels=num_channels,
                                num_filters=num_filters_1,
                                stride=1,
                                )

pool_5 = paddle.layer.img_pool(input=conv_12,
                               pool_type=paddle.pooling.Max(),
                               pool_size=2,
                               stride=2
                               )


fc_1 = paddle.layer.fc(input=pool_5, size=4096, act=paddle.activation.Sigmoid())
fc_2 = paddle.layer.fc(input=fc_1, size=4096, act=paddle.activation.Sigmoid())
fc_3 = paddle.layer.fc(input=fc_2, size=1000, act=paddle.activation.Sigmoid())
output_layer = paddle.layer.fc(input=fc_3, size=10, act=paddle.activation.Softmax())

label = paddle.layer.data(name='label', type=paddle.data_type.integer_value(10))

cost = paddle.layer.classification_cost(input=output_layer, label=label)

parameters = paddle.parameters.create(cost)

optimizer = paddle.optimizer.AdaDelta()


trainer = paddle.trainer.SGD(cost=cost,
                             parameters=parameters,
                             update_equation=optimizer)


feeding = {'image': 0,
           'label': 1}

trainer.train(num_passes=100, event_handler=event_handler, feeding=feeding)
