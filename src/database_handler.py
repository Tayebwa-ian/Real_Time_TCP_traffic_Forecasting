from influxdb_client import InfluxDBClient, Point, WritePrecision
from influxdb_client.client.write_api import SYNCHRONOUS

class DataVault:
    def __init__(self):
        self.token = "my-super-secret-token"
        self.org = "myorg"
        self.bucket = "network_stats"
        self.client = InfluxDBClient(url="http://localhost:8086", token=self.token, org=self.org)
        self.write_api = self.client.write_api(write_options=SYNCHRONOUS)

    def push_metrics(self, actual_thru, pred_thru, delay, bif):
        point = Point("traffic_prediction") \
            .field("actual_throughput", float(actual_thru)) \
            .field("predicted_throughput", float(pred_thru)) \
            .field("rtt_delay", float(delay)) \
            .field("bytes_in_flight", float(bif))
        
        self.write_api.write(bucket=self.bucket, org=self.org, record=point)
