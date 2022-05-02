#-*-coding:utf-8 -*-

from preprocess import cifarnet_preprocessing
from preprocess import inception_preprocessing
from preprocess import lenet_preprocessing
from preprocess import vgg_preprocessing


def get_preprocessing(name):
  preprocessing_fn_map = {
      'cifarnet': cifarnet_preprocessing,
      'inception': inception_preprocessing,
      'inception_v1': inception_preprocessing,
      'inception_v2': inception_preprocessing,
      'inception_v3': inception_preprocessing,
      'inception_v4': inception_preprocessing,
      'inception_resnet_v2': inception_preprocessing,
      'lenet': lenet_preprocessing,
      'mobilenet_v1': inception_preprocessing,
      'mobilenet_v2': inception_preprocessing,
      'mobilenet_v2_035': inception_preprocessing,
      'mobilenet_v3_small': inception_preprocessing,
      'mobilenet_v3_large': inception_preprocessing,
      'mobilenet_v3_small_minimalistic': inception_preprocessing,
      'mobilenet_v3_large_minimalistic': inception_preprocessing,
      'mobilenet_edgetpu': inception_preprocessing,
      'mobilenet_edgetpu_075': inception_preprocessing,
      'mobilenet_v2_140': inception_preprocessing,
      'nasnet_mobile': inception_preprocessing,
      'nasnet_large': inception_preprocessing,
      'pnasnet_mobile': inception_preprocessing,
      'pnasnet_large': inception_preprocessing,
      'resnet_v1_50': vgg_preprocessing,
      'resnet_v1_101': vgg_preprocessing,
      'resnet_v1_152': vgg_preprocessing,
      'resnet_v1_200': vgg_preprocessing,
      'resnet_v2_50': vgg_preprocessing,
      'resnet_v2_101': vgg_preprocessing,
      'resnet_v2_152': vgg_preprocessing,
      'resnet_v2_200': vgg_preprocessing,
      'vgg': vgg_preprocessing,
      'vgg_a': vgg_preprocessing,
      'vgg_16': vgg_preprocessing,
      'vgg_19': vgg_preprocessing,
  }

  if name not in preprocessing_fn_map:
    raise ValueError('Preprocessing name [%s] was not recognized' % name)

  return preprocessing_fn_map[name].preprocess_image
