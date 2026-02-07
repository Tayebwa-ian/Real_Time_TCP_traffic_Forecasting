from influxdb_client import InfluxDBClient, Point, WritePrecision
from influxdb_client.client.write_api import SYNCHRONOUS

class DataVault:
    def __init__(self):
        self.token = "my-super-secret-token"
        self.org = "myorg"
        self.bucket = "network_stats"
        self.client = InfluxDBClient(url="http://localhost:8086", token=self.token, org=self.org)
        self.write_api = self.client.write_api(write_options=SYNCHRONOUS)

    def push_metrics(self, features, prediction):
        """
        features: list containing [size, delay, bif, iat_mean, iat_std, 
                                    throughput_mbps, delay_grad, bif_grad, is_silent]
        prediction: the float output from model.predict()
        """
        point = Point("traffic_prediction") \
            .field("actual_throughput", float(features[5])) \
            .field("predicted_throughput", float(prediction)) \
            .field("rtt_delay", float(features[1])) \
            .field("bytes_in_flight", float(features[2])) \
            .field("delay_gradient", float(features[6])) \
            .field("bif_gradient", float(features[7])) \
            .field("iat_mean", float(features[3])) \
            .field("is_silent", int(features[8]))
        
        # We use synchronous write for real-time monitoring stability
        self.write_api.write(bucket=self.bucket, org=self.org, record=point)
