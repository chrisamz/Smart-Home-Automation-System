# iot_integration.py

"""
IoT Integration Module for Smart Home Automation System

This module contains functions for integrating various IoT devices to enable seamless
communication and control within the smart home environment.

Technologies Used:
- MQTT
- REST APIs

Libraries/Tools:
- paho-mqtt
- requests

"""

import paho.mqtt.client as mqtt
import requests
import json

class IoTIntegration:
    def __init__(self, mqtt_broker, mqtt_port, mqtt_topic, rest_api_base_url):
        """
        Initialize the IoTIntegration class.
        
        :param mqtt_broker: str, MQTT broker address
        :param mqtt_port: int, MQTT broker port
        :param mqtt_topic: str, MQTT topic to subscribe to
        :param rest_api_base_url: str, base URL for the REST API
        """
        self.mqtt_broker = mqtt_broker
        self.mqtt_port = mqtt_port
        self.mqtt_topic = mqtt_topic
        self.rest_api_base_url = rest_api_base_url
        self.client = mqtt.Client()

    def on_connect(self, client, userdata, flags, rc):
        """
        Callback function for MQTT connection.
        
        :param client: MQTT client
        :param userdata: user data
        :param flags: connection flags
        :param rc: connection result code
        """
        print(f"Connected with result code {rc}")
        client.subscribe(self.mqtt_topic)

    def on_message(self, client, userdata, msg):
        """
        Callback function for MQTT messages.
        
        :param client: MQTT client
        :param userdata: user data
        :param msg: MQTT message
        """
        print(f"Message received on topic {msg.topic}: {msg.payload.decode()}")
        self.handle_message(msg.payload.decode())

    def handle_message(self, message):
        """
        Handle incoming MQTT messages and send appropriate commands via REST API.
        
        :param message: str, incoming MQTT message
        """
        data = json.loads(message)
        if 'device' in data and 'action' in data:
            device = data['device']
            action = data['action']
            response = requests.post(f"{self.rest_api_base_url}/{device}", json={"action": action})
            print(f"Sent {action} to {device}, response: {response.status_code}")

    def start(self):
        """
        Start the MQTT client and connect to the broker.
        """
        self.client.on_connect = self.on_connect
        self.client.on_message = self.on_message
        self.client.connect(self.mqtt_broker, self.mqtt_port, 60)
        self.client.loop_start()

if __name__ == "__main__":
    mqtt_broker = "broker.hivemq.com"
    mqtt_port = 1883
    mqtt_topic = "smarthome/commands"
    rest_api_base_url = "http://localhost:5000/api"

    iot_integration = IoTIntegration(mqtt_broker, mqtt_port, mqtt_topic, rest_api_base_url)
    iot_integration.start()

    # Keep the script running
    try:
        while True:
            pass
    except KeyboardInterrupt:
        print("Exiting...")
        iot_integration.client.loop_stop()
        iot_integration.client.disconnect()
