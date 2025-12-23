/*
 * ChargeUp EV Car Prototype - ESP32 Firmware
 * Controls: WiFi, MQTT, LED indicators for battery/charging status
 */

#include <WiFi.h>
#include <PubSubClient.h>

// ============ CONFIGURATION ============
const char* WIFI_SSID = "YOUR_WIFI_SSID";
const char* WIFI_PASSWORD = "YOUR_WIFI_PASSWORD";
const char* MQTT_SERVER = "192.168.1.100";  // Your laptop IP
const int MQTT_PORT = 1883;
const char* CAR_ID = "CAR01";

// LED Pin Definitions
#define LED_RED     25    // Low battery / Error
#define LED_YELLOW  26    // Charging in progress
#define LED_GREEN   27    // Ready / Connected
#define LED_BLUE    14    // MQTT connected

// Simulated battery
int batteryLevel = 75;
bool isCharging = false;
String carStatus = "IDLE";

WiFiClient espClient;
PubSubClient mqtt(espClient);

unsigned long lastTelemetry = 0;
const long telemetryInterval = 5000;

void setup() {
    Serial.begin(115200);
    
    pinMode(LED_RED, OUTPUT);
    pinMode(LED_YELLOW, OUTPUT);
    pinMode(LED_GREEN, OUTPUT);
    pinMode(LED_BLUE, OUTPUT);
    
    allLedsOff();
    connectWiFi();
    
    mqtt.setServer(MQTT_SERVER, MQTT_PORT);
    mqtt.setCallback(mqttCallback);
}

void loop() {
    if (!mqtt.connected()) {
        reconnectMQTT();
    }
    mqtt.loop();
    
    // Send telemetry periodically
    if (millis() - lastTelemetry > telemetryInterval) {
        sendTelemetry();
        lastTelemetry = millis();
    }
    
    // Simulate charging
    if (isCharging && batteryLevel < 100) {
        batteryLevel += 1;
        if (batteryLevel >= 100) {
            batteryLevel = 100;
            isCharging = false;
            carStatus = "FULL";
            updateLeds();
        }
        delay(500);
    }
    
    updateLeds();
}

void connectWiFi() {
    Serial.print("Connecting to WiFi");
    WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
    
    while (WiFi.status() != WL_CONNECTED) {
        delay(500);
        Serial.print(".");
        digitalWrite(LED_RED, !digitalRead(LED_RED));
    }
    
    Serial.println("\nWiFi Connected!");
    Serial.println(WiFi.localIP());
    digitalWrite(LED_GREEN, HIGH);
    digitalWrite(LED_RED, LOW);
}

void reconnectMQTT() {
    while (!mqtt.connected()) {
        String clientId = "ChargeUp_Car_" + String(CAR_ID);
        
        if (mqtt.connect(clientId.c_str())) {
            Serial.println("MQTT Connected!");
            digitalWrite(LED_BLUE, HIGH);
            
            // Subscribe to commands
            String cmdTopic = "chargeup/commands/" + String(CAR_ID);
            mqtt.subscribe(cmdTopic.c_str());
            mqtt.subscribe("chargeup/broadcast");
            
            Serial.println("Subscribed to: " + cmdTopic);
        } else {
            Serial.print("MQTT failed, rc=");
            Serial.println(mqtt.state());
            digitalWrite(LED_BLUE, LOW);
            delay(2000);
        }
    }
}

void mqttCallback(char* topic, byte* payload, unsigned int length) {
    String message = "";
    for (int i = 0; i < length; i++) {
        message += (char)payload[i];
    }
    
    Serial.println("Received: " + message);
    
    // Parse commands
    if (message.startsWith("CHARGE,START")) {
        isCharging = true;
        carStatus = "CHARGING";
        Serial.println(">> CHARGING STARTED");
    }
    else if (message.startsWith("CHARGE,STOP")) {
        isCharging = false;
        carStatus = "IDLE";
        Serial.println(">> CHARGING STOPPED");
    }
    else if (message.startsWith("BATTERY,")) {
        batteryLevel = message.substring(8).toInt();
        Serial.println(">> Battery set to: " + String(batteryLevel));
    }
    else if (message.startsWith("STATUS,")) {
        carStatus = message.substring(7);
        Serial.println(">> Status: " + carStatus);
    }
    
    updateLeds();
}

void sendTelemetry() {
    String topic = "chargeup/telemetry/" + String(CAR_ID);
    
    String payload = "{";
    payload += "\"device\":\"" + String(CAR_ID) + "\",";
    payload += "\"battery_level\":" + String(batteryLevel) + ",";
    payload += "\"status\":\"" + carStatus + "\",";
    payload += "\"charging\":" + String(isCharging ? 1 : 0) + ",";
    payload += "\"lat\":9.9674,\"lon\":76.2807,";
    payload += "\"range_km\":" + String(batteryLevel * 4);
    payload += "}";
    
    mqtt.publish(topic.c_str(), payload.c_str());
    Serial.println("Telemetry sent: " + payload);
}

void updateLeds() {
    // Red: Low battery or error
    digitalWrite(LED_RED, batteryLevel < 20 ? HIGH : LOW);
    
    // Yellow: Charging
    digitalWrite(LED_YELLOW, isCharging ? HIGH : LOW);
    
    // Green: Ready (not charging, battery OK)
    digitalWrite(LED_GREEN, (!isCharging && batteryLevel >= 20) ? HIGH : LOW);
}

void allLedsOff() {
    digitalWrite(LED_RED, LOW);
    digitalWrite(LED_YELLOW, LOW);
    digitalWrite(LED_GREEN, LOW);
    digitalWrite(LED_BLUE, LOW);
}
