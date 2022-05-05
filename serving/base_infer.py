# -*-coding:utf-8 -*-
import time
import random
import grpc
from functools import wraps
from grpc import StatusCode, RpcError
from tools.logger import logger
from tensorflow_serving.apis import predict_pb2, prediction_service_pb2_grpc
from tensorflow.python.framework import tensor_util
from dataset.tokenizer import get_tokenizer

GRPC_RETRY_TIEMS = {
    StatusCode.INTERNAL: 1,
    StatusCode.ABORTED: 3,
    StatusCode.UNAVAILABLE: 3,
    StatusCode.DEADLINE_EXCEEDED: 5  # most-likely grpc channel close, need time to reopen
}

MsTime = lambda: int(round(time.time() * 1000))


def timer(func):
    @wraps(func)
    def helper(*args, **kwargs):
        start_ts = MsTime()
        output = func(*args, **kwargs)
        end_ts = MsTime()
        logger.info('{} latency = {}'.format(func.__name__, end_ts - start_ts))
        return output

    return helper


def retry(retry_times=None, sleep=0.01, backoff=False):
    default_max_retry = 3

    def helper(func):
        @wraps(func)
        def handle_args(*args, **kwargs):
            counter = 0
            while True:
                try:
                    return func(*args, **kwargs)
                except RpcError as e:
                    if retry_times:
                        max_retry = retry_times.get(e.code(), default_max_retry)
                    else:
                        max_retry = default_max_retry
                    if counter >= max_retry:
                        raise e
                    counter += 1
                    logger.info('{} retrying for {} times StatusCode={} Msg={}'.format(func.__name__,
                                                                                       counter, e.code(),
                                                                                       e.details()))
                except Exception as e:
                    if counter >= default_max_retry:
                        raise e
                    counter += 1
                    logger.info('{} retrying for {} times MSG={}'.format(func.__name__, counter, e))

                if backoff:
                    ts = min(sleep * 2 ** counter, 1)  # exponential backoff
                else:
                    ts = sleep
                time.sleep(ts)  # wait for grpc to reopen channel

        return handle_args

    return helper


class BaseInfer(object):
    def __init__(self, server_list, max_seq_len, timeout, nlp_pretrain_model, model_name, model_version):
        self.server_list = server_list
        self.timeout = timeout  # unit in second
        self.max_seq_len = max_seq_len
        self.tokenizer = get_tokenizer(nlp_pretrain_model)
        self.model_name = model_name
        self.model_version = model_version
        self.proto = None
        self.channels = {}
        self.init_channel()

    def init_channel(self):
        for i in self.server_list:
            self.channels[i] = grpc.insecure_channel(i)

    def get_channel(self):
        address = random.choice(self.server_list)
        return self.channels[address]

    def make_feature(self, input):
        raise NotImplementedError()

    def make_request(self, feature):
        request = predict_pb2.PredictRequest()
        request.model_spec.name = self.model_name
        request.model_spec.version.value = self.model_version
        for field in self.proto:
            tensor_proto = tensor_util.make_tensor_proto(feature[field], dtype=self.proto[field])
            request.inputs[field].CopyFrom(tensor_proto)
        return request

    def decode_prediction(self, resp):
        raise NotImplementedError

    @retry(retry_times=GRPC_RETRY_TIEMS)
    def _infer(self, req):
        channel = self.get_channel()
        stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)
        resp = stub.Predict.future(req, self.timeout)
        return resp

    @timer
    def infer(self, input):
        try:
            feature = self.make_feature(input)
            req = self.make_request(feature)
            resp = self._infer(req)
        except Exception as e:
            logger.exception(e)
            resp = None # 每个infer class按各自的decode逻辑返回相同格式相同的返回值
        output = self.decode_prediction(resp)
        return output
