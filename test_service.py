import sys
import grpc

# import the generated classes
import service.service_spec.series_decomposition_pb2_grpc as grpc_bt_grpc
import service.service_spec.series_decomposition_pb2 as grpc_bt_pb2

from service import registry

if __name__ == "__main__":

    try:
        test_flag = False
        if len(sys.argv) == 2:
            if sys.argv[1] == "auto":
                test_flag = True

        # Service ONE - Arithmetic
        endpoint = input("Endpoint (localhost:{}): ".format(
            registry["series_decomposition"]["grpc"])) if not test_flag else ""
        if endpoint == "":
            endpoint = "localhost:{}".format(registry["series_decomposition"]["grpc"])

        # Open a gRPC channel
        channel = grpc.insecure_channel("{}".format(endpoint))

        grpc_method = input("Method (forecast): ") if not test_flag else ""
        if grpc_method == "":
            grpc_method = "forecast"

        series = []
        input_series = input("Series: ") if not test_flag else series
        if input_series == "":
            input_series = series

        input_period = input("Period (10): ") if not test_flag else 10
        if input_period == "":
            input_period = 10

        if grpc_method == "forecast":
            stub = grpc_bt_grpc.ForecastStub(channel)
            response = stub.forecast(grpc_bt_pb2.Input(series=input_series,
                                                       period=int(input_period)))
            print("\nresponse:")
            print("len(response.observed): {}".format(len(response.observed)))
            print("len(response.seasonal): {}".format(len(response.seasonal)))
            print("len(response.forecast): {}".format(len(response.forecast)))
        else:
            print("Invalid method!")
            exit(1)

    except Exception as e:
        print(e)
        exit(1)
