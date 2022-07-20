import tensorflow as tf


class PReLU(tf.keras.layers.Layer):
    def __init__(self, shared_axes=(1, 2), trainable=True, name=None, **kwargs):
        super(PReLU, self).__init__(name=name, trainable=trainable, **kwargs)
        self.relu = tf.keras.layers.PReLU(shared_axes=shared_axes)

    def call(self, inputs, **kwargs):
        return self.relu(inputs)


class ChannelShuffle(tf.keras.layers.Layer):
    def __init__(self, groups=2, name: str = None, trainable: bool = True, **kwargs):
        super(ChannelShuffle, self).__init__(name=name, trainable=trainable, **kwargs)
        self.groups = groups

    def build(self, input_shape):
        if isinstance(input_shape[-1], int):
            self.height = input_shape[1]
            self.width = input_shape[2]
            self.channels = input_shape[3]
        else:
            self.height = input_shape[1].value
            self.width = input_shape[2].value
            self.channels = input_shape[3].value
        assert (self.channels % self.groups == 0)
        super().build(input_shape)

    def call(self, inputs, **kwargs):
        channels_per_group = self.channels // self.groups
        out = tf.reshape(inputs, shape=(-1, self.height, self.width, self.groups, channels_per_group))
        out = tf.transpose(out, perm=(0, 1, 2, 4, 3))
        out = tf.reshape(out, shape=(-1, self.height, self.width, self.channels))
        return out


class Normalization(tf.keras.layers.Layer):
    def __init__(self, momentum=0.1, epsilon=1e-3, activation=None, is_sync: bool = True, name: str = None,
                 trainable: bool = True, **kwargs):
        super(Normalization, self).__init__(name=name, trainable=trainable, **kwargs)
        self.norm = tf.keras.layers.experimental.SyncBatchNormalization(momentum=momentum,
                                                                        epsilon=epsilon) if is_sync else tf.keras.layers.BatchNormalization(
            momentum=momentum, epsilon=epsilon)
        self.activation = activation if activation is not None else tf.keras.layers.Lambda(lambda x: x)

    def build(self, input_shape):
        super().build(input_shape)

    def call(self, inputs, **kwargs):
        return self.activation(self.norm(inputs))


class Conv2D(tf.keras.layers.Layer):
    def __init__(self, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups: int = 1,
                 bias: bool = True, padding_mode: str = 'CONSTANT', apply_padding: bool = False,
                 activation=None, name: str = None, trainable: bool = True, **kwargs):
        super(Conv2D, self).__init__(name=name, trainable=trainable, **kwargs)
        self.padding_type = 'valid'
        if not apply_padding and padding != 0:
            self.padding_type = 'same'
        self.grouped_conv = []
        self.padding_mode = padding_mode
        self.apply_padding = apply_padding
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.groups = groups
        self.padding = padding
        self.bias = bias
        self.is_groups = False
        self.pad = tf.keras.layers.Lambda(
            lambda x: tf.pad(x, tf.constant([[0, 0], [padding, padding], [padding, padding], [0, 0]]),
                             padding_mode)) if (padding != 0 and apply_padding is True) else tf.keras.layers.Lambda(
            lambda x: x)
        self.activation = activation if activation is not None else tf.keras.layers.Lambda(lambda x: x)

    def _split_channels(self, total_filters, num_groups):
        split = [total_filters // num_groups for _ in range(num_groups)]
        split[0] += total_filters - sum(split)
        return split

    def build(self, input_shape):
        if isinstance(input_shape[-1], int):
            self.height = input_shape[1]
            self.width = input_shape[2]
            self.channels = input_shape[3]
        else:
            self.height = input_shape[1].value
            self.width = input_shape[2].value
            self.channels = input_shape[3].value
        if (self.groups == self.channels and self.channels == self.out_channels) or self.out_channels is None:
            self.conv = tf.keras.layers.DepthwiseConv2D(kernel_size=self.kernel_size, strides=self.stride,
                                                        dilation_rate=self.dilation, use_bias=self.bias,
                                                        kernel_regularizer=tf.keras.regularizers.l2(
                                                            4e-4),
                                                        padding=self.padding_type)
        elif self.groups == 1:
            self.conv = tf.keras.layers.Conv2D(filters=self.out_channels, kernel_size=self.kernel_size,
                                               kernel_regularizer=tf.keras.regularizers.l2(
                                                   4e-4),
                                               strides=self.stride, dilation_rate=self.dilation, use_bias=self.bias,
                                               padding=self.padding_type)
        else:
            self.is_groups = True
            splits = self._split_channels(self.out_channels, self.groups)
            for i in range(self.groups):
                self.grouped_conv.append(
                    Conv2D(splits[i], kernel_size=self.kernel_size, stride=self.stride, padding=self.padding,
                           dilation=self.dilation, groups=1, bias=self.bias, padding_mode=self.padding_mode,
                           activation=None,
                           apply_padding=self.apply_padding, name="grouped_{}".format(i))
                )
        super().build(input_shape)

    def call(self, inputs, **kwargs):
        if self.is_groups:
            if len(self.grouped_conv) == 1:
                out = self.grouped_conv[0](inputs)
            else:
                splits = self._split_channels(self.channels, len(self.grouped_conv))
                out = tf.concat([c(x) for x, c in zip(tf.split(inputs, splits, -1), self.grouped_conv)], -1)
        else:
            out = self.conv(self.pad(inputs))
        return self.activation(out)


class SqueezeBlock(tf.keras.layers.Layer):
    def __init__(self, exp_size, divide=4.0, name: str = None, trainable: bool = True, **kwargs):
        super(SqueezeBlock, self).__init__(name=name, trainable=trainable, **kwargs)
        if divide > 1:
            self.dense = tf.keras.Sequential([
                tf.keras.layers.Dense(int(exp_size / divide)),
                PReLU(shared_axes=(1,)),
                tf.keras.layers.Dense(exp_size),
                PReLU(shared_axes=(1,))]
            )
        else:
            self.dense = tf.keras.Sequential([
                tf.keras.layers.Dense(exp_size),
                PReLU(shared_axes=(1,))]
            )

    def call(self, inputs, **kwargs):
        out = tf.reduce_mean(inputs, axis=(1, 2), keepdims=True)
        out = self.dense(out)
        return out * inputs


class SqueezeSeparableConv2D(tf.keras.layers.Layer):
    def __init__(self, out_channels, kernel_size, stride=1, divide=2.0, name: str = None, trainable: bool = True,
                 **kwargs):
        super(SqueezeSeparableConv2D, self).__init__(name=name, trainable=trainable, **kwargs)
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.strides = stride
        self.divide = divide

    def build(self, input_shape):
        if isinstance(input_shape[-1], int):
            self.height = input_shape[1]
            self.width = input_shape[2]
            self.channels = input_shape[3]
        else:
            self.height = input_shape[1].value
            self.width = input_shape[2].value
            self.channels = input_shape[3].value
        padding = int((self.kernel_size - 1) / 2)
        self.conv = tf.keras.Sequential([
            Conv2D(self.channels, self.kernel_size, stride=self.strides,
                   padding=padding,
                   groups=self.channels, bias=False),
            SqueezeBlock(self.channels, divide=self.divide),
            Conv2D(self.out_channels, kernel_size=1, stride=1, bias=False,
                   activation=Normalization(activation=PReLU()))]
        )
        super().build(input_shape)

    def call(self, inputs, **kwargs):
        output = self.conv(inputs)
        return output


class UpsamplingBilinear2D(tf.keras.layers.Layer):
    def __init__(self, scale_factor=2, activation=None, trainable=True, name=None, **kwargs):
        super(UpsamplingBilinear2D, self).__init__(name=name, trainable=trainable, **kwargs)
        self.scale_factor = scale_factor
        self.activation = activation if activation is not None else tf.keras.layers.Lambda(lambda x: x)

    def build(self, input_shape):
        if isinstance(input_shape[-1], int):
            self.height = input_shape[1]
            self.width = input_shape[2]
        else:
            self.height = input_shape[1].value
            self.width = input_shape[2].value
        super().build(input_shape)

    def call(self, inputs, **kwargs):
        return self.activation(tf.image.resize(inputs,
                                               (int(self.height * self.scale_factor),
                                                int(self.width * self.scale_factor))))


class S2Block(tf.keras.layers.Layer):
    def __init__(self, out_channels, config, name: str = None, trainable: bool = True, **kwargs):
        super(S2Block, self).__init__(name=name, trainable=trainable, **kwargs)
        kernel_size = config[0]
        pool_size = config[1]
        self.kernel_size = kernel_size
        self.resolution_down = False
        if pool_size > 1:
            self.resolution_down = True
            self.down_res = tf.keras.layers.AveragePooling2D(pool_size, pool_size)
            self.up_res = UpsamplingBilinear2D(scale_factor=pool_size)
            self.pool_size = pool_size
        self.conv1x1 = Conv2D(out_channels, kernel_size=1, stride=1, bias=False)
        self.norm = Normalization()

    def build(self, input_shape):
        if isinstance(input_shape[-1], int):
            self.height = input_shape[1]
            self.width = input_shape[2]
            self.channels = input_shape[3]
        else:
            self.height = input_shape[1].value
            self.width = input_shape[2].value
            self.channels = input_shape[3].value
        padding = int((self.kernel_size - 1) / 2)
        self.conv = Conv2D(self.channels, kernel_size=self.kernel_size, stride=1,
                           padding=padding, groups=self.channels, bias=False,
                           activation=Normalization(activation=PReLU()))
        super().build(input_shape)

    def call(self, inputs, **kwargs):
        if self.resolution_down:
            inputs = self.down_res(inputs)
        output = self.conv(inputs)
        output = self.conv1x1(output)
        if self.resolution_down:
            output = self.up_res(output)
        return self.norm(output)


class S2Module(tf.keras.layers.Layer):
    def __init__(self, out_channels, add=True, config=None, name: str = None, trainable: bool = True, **kwargs):
        super(S2Module, self).__init__(name=name, trainable=trainable, **kwargs)
        if config is None:
            config = [[3, 1], [5, 1]]
        group_n = len(config)
        split_n = int(out_channels / group_n)
        split_patch = out_channels - group_n * split_n
        self.conv_split = Conv2D(split_n, kernel_size=1, stride=1, padding=0, bias=False, groups=group_n)
        self.s2_d1 = S2Block(split_n + split_patch, config[0])
        self.s2_d2 = S2Block(split_n, config[1])
        self.norm = Normalization(activation=PReLU())
        self.add = add
        self.group_n = group_n
        self.channel_shuffle = ChannelShuffle(groups=group_n)

    def call(self, inputs, **kwargs):
        output = self.channel_shuffle(self.conv_split(inputs))
        combine = tf.concat([self.s2_d2(output), self.s2_d1(output)], -1)
        if self.add:
            combine = inputs + combine
        output = self.norm(combine)
        return output


class SINetEncoder(tf.keras.layers.Layer):
    def __init__(self, classes=20, p=5, q=3, chnn=1.0, name: str = None, trainable: bool = True, **kwargs):
        super(SINetEncoder, self).__init__(name=name, trainable=trainable, **kwargs)
        """
        :param classes: number of classes in the dataset. Default is 20 for the cityscapes
        :param p: depth multiplier
        :param q: depth multiplier
        """
        config = [[[3, 1], [5, 1]], [[3, 1], [3, 1]],
                  [[3, 1], [5, 1]], [[3, 1], [3, 1]], [[5, 1], [3, 2]], [[5, 2], [3, 4]],
                  [[3, 1], [3, 1]], [[5, 1], [5, 1]], [[3, 2], [3, 4]], [[3, 1], [5, 2]]]

        print("SINet Enc bracnch num :  " + str(len(config[0])))
        print("SINet Enc chnn num:  " + str(chnn))
        out_channels = [16, 48 + 4 * (chnn - 1), 96 + 4 * (chnn - 1)]
        self.level1 = Conv2D(12, kernel_size=3, stride=2,
                             padding=1, bias=False,
                             activation=Normalization(activation=PReLU()))
        self.level2_0 = SqueezeSeparableConv2D(out_channels[0], 3, 2, divide=1)
        self.level2 = []
        for i in range(0, p):
            if i == 0:
                self.level2.append(S2Module(out_channels[1], config=config[i], add=False))
            else:
                self.level2.append(S2Module(out_channels[1], config=config[i]))
        self.BR2 = Normalization(activation=PReLU())
        self.level3_0 = SqueezeSeparableConv2D(out_channels[1], 3, 2, divide=2)
        self.level3 = []
        for i in range(0, q):
            if i == 0:
                self.level3.append(S2Module(out_channels[2], config=config[2 + i], add=False))
            else:
                self.level3.append(S2Module(out_channels[2], config=config[2 + i]))
        self.BR3 = Normalization(activation=PReLU())
        self.classifier = Conv2D(classes, kernel_size=1, stride=1, padding=0, bias=False)

    def call(self, inputs, **kwargs):
        output1 = self.level1(inputs)  # 8h 8w
        output2_0 = self.level2_0(output1)  # 4h 4w
        output2 = None
        for i, layer in enumerate(self.level2):
            if i == 0:
                output2 = layer(output2_0)
            else:
                output2 = layer(output2)  # 2h 2w
        output3_0 = self.level3_0(self.BR2(tf.concat([output2_0, output2], - 1)))  # h w
        output3 = None
        for i, layer in enumerate(self.level3):
            if i == 0:
                output3 = layer(output3_0)
            else:
                output3 = layer(output3)
        output3_cat = self.BR3(tf.concat([output3_0, output3], -1))
        classifier = self.classifier(output3_cat)
        return classifier


class SINet(tf.keras.layers.Layer):
    def __init__(self, num_classes=20, p=2, q=8, chnn=1.0, name: str = None, trainable: bool = True, **kwargs):
        super(SINet, self).__init__(name=name, trainable=trainable, **kwargs)
        """
        :param classes: number of classes in the dataset. Default is 20 for the cityscapes
        :param p: depth multiplier
        :param q: depth multiplier 
        """
        self.num_classes = num_classes
        self.p = p
        self.q = q
        self.chnn = chnn
        config = [[[3, 1], [5, 1]], [[3, 1], [3, 1]],
                  [[3, 1], [5, 1]], [[3, 1], [3, 1]], [[5, 1], [3, 2]], [[5, 2], [3, 4]],
                  [[3, 1], [3, 1]], [[5, 1], [5, 1]], [[3, 2], [3, 4]], [[3, 1], [5, 2]]]
        out_channels = [16, 48 + 4 * (chnn - 1), 96 + 4 * (chnn - 1)]
        self.conv_down1 = Conv2D(12, kernel_size=3, stride=2, padding=1, bias=False,
                                 activation=Normalization(activation=PReLU()))
        self.conv_down2 = SqueezeSeparableConv2D(out_channels[0], kernel_size=3, stride=2, divide=1)
        self.encoder_level2 = []
        for i in range(0, p):
            if i == 0:
                self.encoder_level2.append(
                    S2Module(out_channels[1], config=config[i], add=False))
            else:
                self.encoder_level2.append(S2Module(out_channels[1], config=config[i]))
        self.encoder_norm2 = Normalization(activation=PReLU())
        self.conv_down3 = SqueezeSeparableConv2D(out_channels[1], kernel_size=3, stride=2, divide=2)
        self.encoder_level3 = []
        for i in range(0, q):
            if i == 0:
                self.encoder_level3.append(
                    S2Module(out_channels[2], config=config[2 + i], add=False))
            else:
                self.encoder_level3.append(S2Module(out_channels[2], config=config[2 + i]))
        self.encoder_norm3 = Normalization(activation=PReLU())
        self.level3_classifier = Conv2D(num_classes, kernel_size=1, stride=1, padding=0, bias=False)
        self.up1 = UpsamplingBilinear2D(scale_factor=2, activation=Normalization())
        self.up2 = UpsamplingBilinear2D(scale_factor=2, activation=Normalization())
        self.level2_classifier = Conv2D(num_classes, kernel_size=1, stride=1,
                                        padding=0, bias=False,
                                        activation=Normalization(
                                            activation=PReLU()))
        self.up3 = UpsamplingBilinear2D(scale_factor=2)
        self.classifier = Conv2D(num_classes, 3, 1, 1, bias=False,
                                 activation=tf.nn.sigmoid if num_classes == 1 else tf.nn.softmax)

    def get_config(self):
        config = {
            "num_classes": self.num_classes,
            "p": self.p,
            "q": self.q,
            "chnn": self.chnn,
        }
        base_config = super().get_config()
        return {**base_config, **config}

    def call(self, inputs, **kwargs):
        out_down1 = self.conv_down1(inputs)  # 8h 8w
        out_down2 = self.conv_down2(out_down1)  # 4h 4w
        out_level2 = None
        for i, layer in enumerate(self.encoder_level2):
            if i == 0:
                out_level2 = layer(out_down2)
            else:
                out_level2 = layer(out_level2)  # 2h 2w
        out_down3 = self.conv_down3(self.encoder_norm2(tf.concat([out_down2, out_level2], -1)))  # h w
        out_level3 = None
        for i, layer in enumerate(self.encoder_level3):
            if i == 0:
                out_level3 = layer(out_down3)
            else:
                out_level3 = layer(out_level3)
        output3_cat = self.encoder_norm3(tf.concat([out_down3, out_level3], -1))
        enc_final = self.level3_classifier(output3_cat)  # 1/8
        dnc_stage1 = self.up1(enc_final)  # 1/4
        stage1_confidence = tf.reduce_max(
            tf.nn.softmax(dnc_stage1) if self.num_classes != 1 else tf.nn.sigmoid(dnc_stage1), axis=-1, keepdims=True)
        dnc_stage2_0 = self.level2_classifier(out_level2)  # 2h 2w
        dnc_stage2 = self.up2(dnc_stage2_0 * (1. - stage1_confidence) + dnc_stage1)  # 4h 4w
        dnc_stage2 = self.up3(dnc_stage2)
        classifier = self.classifier(dnc_stage2)
        return classifier


def Model(output_resolution=512, num_classes=2):
    print('*** Building SINet Network ***')
    inputs = tf.keras.layers.Input(shape=(output_resolution, output_resolution, 3), name='inputs')
    outputs = SINet(num_classes=num_classes, p=2, q=8)(inputs)
    model = tf.keras.Model(inputs=inputs, outputs=outputs, name='SINet')
    print(f'*** Output_Shape => {model.output_shape} ***')
    model.summary()
    return model


def main():
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    output_resolution = 512
    classes = 1
    r = tf.ones(shape=(1, output_resolution, output_resolution, 1))
    g = tf.ones(shape=(1, output_resolution, output_resolution, 1)) * 2.0
    b = tf.ones(shape=(1, output_resolution, output_resolution, 1)) * 3.0
    inputs = tf.concat([r, g, b], -1)
    output = Model(output_resolution=output_resolution, num_classes=classes)(inputs)
    print(output)

    model = Model(output_resolution=output_resolution, num_classes=classes)
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    open("model.tflite", "wb").write(tflite_model)


if __name__ == "__main__":
    main()
