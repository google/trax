from trax import layers as tl


def EncoderDecoder(kernels, filters, strides=None,
                   output_channels=3,
                   mode='train',
                   norm=tl.BatchNorm,
                   non_linearity=tl.Relu):
    n_layers = len(filters)

    if isinstance(kernels, int):
        kernels = [kernels for _ in range(n_layers)]

    if strides is None:
        strides = [None for _ in range(n_layers)]

    def Encoder():
        return [tl.Conv(_filters, (ks, ks), _strides) for
                _filters, ks, _strides in 
                zip(kernels, filters, strides)]

    def Decoder():
        return [tl.ConvTranspose(_filters, (ks, ks), _strides) for
                _filters, ks, _strides in
                zip(kernels, filters, strides)]

    return tl.Serial(Encoder(), Decoder(),
                     tl.Conv(output_channels, (1, 1)))