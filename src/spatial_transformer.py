from keras.layers.core import Layer
import keras.backend as K

if K.backend() == 'tensorflow':
    import tensorflow as tf
    def K_arange(start, stop=None, step=1, dtype='int32'):
        result = tf.range(start, limit=stop, delta=step, name='arange')
        if dtype != 'int32':
            result = K.cast(result, dtype)
        return result

    def K_meshgrid(x, y):
        return tf.meshgrid(x, y)

    def K_matmul(x, y):
        return tf.matmul(x, y)

    def K_linspace(start, stop, num):
        return tf.linspace(start, stop, num)

elif K.backend() == 'theano':
    from theano import tensor as T
    def K_arange(start, stop=None, step=1, dtype='int32'):
        return T.arange(start, stop=stop, step=step, dtype=dtype)

    def K_meshgrid(x, y):
        return T.mgrid(x, y)

    def K_matmul(x, y):
        return T.dot(x, y)

    def K_linspace(start, stop, num):
        step = ((stop - start) / num)
        return T.arange(start, stop, step)

class SpatialTransformer(Layer):
    """Spatial Transformer Layer
    Implements a spatial transformer layer as described in [1]_.
    Borrowed from [2]_:
    downsample_fator : float
        A value of 1 will keep the orignal size of the image.
        Values larger than 1 will down sample the image. Values below 1 will
        upsample the image.
        example image: height= 100, width = 200
        downsample_factor = 2
        output image will then be 50, 100
    References
    ----------
    .. [1]  Spatial Transformer Networks
            Max Jaderberg, Karen Simonyan, Andrew Zisserman, Koray Kavukcuoglu
            Submitted on 5 Jun 2015
    .. [2]  https://github.com/skaae/transformer_network/blob/master/transformerlayer.py

    .. [3]  https://github.com/EderSantana/seya/blob/keras1/seya/layers/attention.py
    """

    def __init__(self,
                 localization_net,
                 output_size,
                 **kwargs):
        self.locnet = localization_net
        self.output_size = output_size
        super(SpatialTransformer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.locnet.build(input_shape)
        self.trainable_weights = self.locnet.trainable_weights
        self.constraints = self.locnet.constraints

    def compute_output_shape(self, input_shape):
        output_size = self.output_size
        return (None,
                int(output_size[0]),
                int(output_size[1]),
                int(input_shape[-1]))

    def call(self, X, mask=None):
        affine_transformation = self.locnet.call(X)
        output = self._transform(affine_transformation, X, self.output_size)
        return output

    def _repeat(self, x, num_repeats):
        x = K.expand_dims(x, axis=-1)
        x = K.repeat_elements(x, num_repeats, axis=1)
        return K.reshape(x, [-1])

    def _interpolate(self, image, x, y, output_size):
        batch_size = K.shape(image)[0]
        height = K.shape(image)[1]
        width = K.shape(image)[2]
        num_channels = K.shape(image)[3]

        x = K.cast(x , dtype='float32')
        y = K.cast(y , dtype='float32')

        height_float = K.cast(height, dtype='float32')
        width_float = K.cast(width, dtype='float32')

        output_height = output_size[0]
        output_width  = output_size[1]

        x = .5 * (x + 1.0) * (width_float)
        y = .5 * (y + 1.0) * (height_float)

        x0 = K.cast(x, 'int32')
        x1 = x0 + 1
        y0 = K.cast(y, 'int32')
        y1 = y0 + 1

        max_x = int(K.int_shape(image)[1] - 1)
        max_y = int(K.int_shape(image)[2] - 1)
        x0 = K.clip(x0, 0, max_x)
        x1 = K.clip(x1, 0, max_x)
        y0 = K.clip(y0, 0, max_y)
        y1 = K.clip(y1, 0, max_y)

        flat_image_dimensions = width*height
        pixels_batch = K_arange(batch_size)*flat_image_dimensions
        flat_output_dimensions = output_height*output_width
        base = self._repeat(pixels_batch, flat_output_dimensions)
        base_y0 = base + y0*width
        base_y1 = base + y1*width
        indices_a = base_y0 + x0
        indices_b = base_y1 + x0
        indices_c = base_y0 + x1
        indices_d = base_y1 + x1

        flat_image = K.reshape(image, shape=(-1, num_channels))
        flat_image = K.cast(flat_image, dtype='float32')
        pixel_values_a = K.gather(flat_image, indices_a)
        pixel_values_b = K.gather(flat_image, indices_b)
        pixel_values_c = K.gather(flat_image, indices_c)
        pixel_values_d = K.gather(flat_image, indices_d)

        x0 = K.cast(x0, 'float32')
        x1 = K.cast(x1, 'float32')
        y0 = K.cast(y0, 'float32')
        y1 = K.cast(y1, 'float32')

        area_a = K.expand_dims(((x1 - x) * (y1 - y)), 1)
        area_b = K.expand_dims(((x1 - x) * (y - y0)), 1)
        area_c = K.expand_dims(((x - x0) * (y1 - y)), 1)
        area_d = K.expand_dims(((x - x0) * (y - y0)), 1)

        values_a = area_a * pixel_values_a
        values_b = area_b * pixel_values_b
        values_c = area_c * pixel_values_c
        values_d = area_d * pixel_values_d
        output = values_a + values_b + values_c + values_d

        return output

    def _meshgrid(self, height, width):
        x_linspace = K_linspace(-1., 1., width)
        y_linspace = K_linspace(-1., 1., height)
        x_coordinates, y_coordinates = K_meshgrid(x_linspace, y_linspace)
        x_coordinates = K.reshape(x_coordinates, [-1])
        y_coordinates = K.reshape(y_coordinates, [-1])
        ones = K.ones_like(x_coordinates)
        indices_grid = K.concatenate([x_coordinates, y_coordinates, ones], 0)
        return indices_grid

    def _transform(self, affine_transformation, input_shape, output_size):
        batch_size = K.shape(input_shape)[0]
        height = K.shape(input_shape)[1]
        width = K.shape(input_shape)[2]
        num_channels = K.shape(input_shape)[3]

        affine_transformation = K.reshape(affine_transformation,
                                        shape=(batch_size, 2, 3))

        affine_transformation = K.cast(affine_transformation, 'float32')

        width = K.cast(width, dtype='float32')
        height = K.cast(height, dtype='float32')
        output_height = output_size[0]
        output_width = output_size[1]
        indices_grid = self._meshgrid(output_height, output_width)
        indices_grid = K.expand_dims(indices_grid, 0)
        indices_grid = K.reshape(indices_grid, [-1])

        indices_grid = K.tile(indices_grid, K.stack([batch_size]))
        indices_grid = K.reshape(indices_grid, (batch_size, 3, -1))

        transformed_grid = K_matmul(affine_transformation, indices_grid)

        x_s = transformed_grid[0:, 0:1, 0:]
        y_s = transformed_grid[0:, 1:, 0:]
        x_s_flatten = K.reshape(x_s, [-1])
        y_s_flatten = K.reshape(y_s, [-1])

        transformed_image = self._interpolate(input_shape,
                                              x_s_flatten,
                                              y_s_flatten,
                                              output_size)

        transformed_image = K.reshape(transformed_image, shape=(batch_size,
                                                                output_height,
                                                                output_width,
                                                                num_channels))
        return transformed_image


