import sys

import grpc
import concurrent.futures as futures

import multiprocessing
import logging

from service import common
from service.series_decomposition import DecompositionForecast

# Importing the generated codes from buildproto.sh
from service.service_spec import series_decomposition_pb2_grpc as grpc_bt_grpc
from service.service_spec.series_decomposition_pb2 import Output

logging.basicConfig(level=10, format="%(asctime)s - [%(levelname)8s] - %(name)s - %(message)s")
log = logging.getLogger("series_decomposition")


def mp_forecast(obj, request, return_dict):
    if request.series and request.period:
        return_dict["response"] = obj.run(request.series, request.period)
    return


# Create a class to be added to the gRPC server
# derived from the protobuf codes.
class ForecastServicer(grpc_bt_grpc.ForecastServicer):
    def __init__(self):
        log.info("ForecastServicer created")

    # The method that will be exposed to the snet-cli call command.
    # request: incoming data
    # context: object that provides RPC-specific information (timeout, etc).
    @staticmethod
    def forecast(request, _):
        manager = multiprocessing.Manager()
        return_dict = manager.dict()
    
        p = multiprocessing.Process(target=mp_forecast, args=(DecompositionForecast(Output),
                                                              request,
                                                              return_dict))
        p.start()
        p.join()

        response = return_dict.get("response", None)
        if not response:
            return Output()

        log.info("forecast({},{})={}".format(len(request.series),
                                             len(response.seasonal),
                                             len(response.forecast)))
        return response


# The gRPC serve function.
#
# Params:
# max_workers: pool of threads to execute calls asynchronously
# port: gRPC server port
#
# Add all your classes to the server here.
# (from generated .py files by protobuf compiler)
def serve(max_workers=10, port=7777):
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=max_workers))
    grpc_bt_grpc.add_ForecastServicer_to_server(ForecastServicer(), server)
    server.add_insecure_port("[::]:{}".format(port))
    return server


if __name__ == "__main__":
    """
    Runs the gRPC server to communicate with the Snet Daemon.
    """
    parser = common.common_parser(__file__)
    args = parser.parse_args(sys.argv[1:])
    common.main_loop(serve, args)
