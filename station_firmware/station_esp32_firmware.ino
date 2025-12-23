/*
 * ChargeUp Charging Station Prototype - ESP32 Firmware
 * Controls: WiFi, MQTT, LED indicators for station status
 */

#include <WiFi.h>
#include <PubSubClient.h>

// ============ CONFIGURATION ============
const char* WIFI_SSID = "YOUR_WIFI_SSID";
const char* WIFI_PASSWORD = "YOUR_WIFI_PASSWORD";
const char* MQTT_SERVER = "192.168.1.100";
const int MQTT_PORT = 1883;
const char* STATION_ID = "STN01";

// LED Pin Definitions
#define LED_RED     25    // Offline / Emergency
#define LED_YELLOW  26    // Queue active / Full
#define LED_GREEN   27    // Available
#define LED_BLUE    14    // Car charging

// Station state
bool isOperational = true;
int availableSlots = 4;
int totalSlots = 4;
bool isCharging = false;
String currentCarId = "";

WiFiClient espClient;
PubSubClient mqtt(espClient);

unsigned long lastStatus = 0;
const long statusInterval = 3000;

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
    
    if (millis() - lastStatus > statusInterval) {
        sendStatus();
        lastStatus = millis();
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
    allLedsOff();
    digitalWrite(LED_GREEN, HIGH);
}

void reconnectMQTT() {
    while (!mqtt.connected()) {
        String clientId = "ChargeUp_Station_" + String(STATION_ID);
        
        if (mqtt.connect(clientId.c_str())) {
            Serial.println("MQTT Connected!");
            
            String cmdTopic = "chargeup/station/" + String(STATION_ID) + "/command";
            mqtt.subscribe(cmdTopic.c_str());
            mqtt.subscribe("chargeup/broadcast");
            
            Serial.println("Subscribed: " + cmdTopic);
        } else {
            delay(2000);
        }
    }
}

void mqttCallback(char* topic, byte* payload, unsigned int length) {
    String message = "";
    for (int i = 0; i < length; i++) {
        message += (char)payload[i];
    }
    
    Serial.println("CMD: " + message);
    
    if (message.startsWith("START_CHARGE")) {
        isCharging = true;
        availableSlots = max(0, availableSlots - 1);
        Serial.println(">> Charging started");
    }
    else if (message.startsWith("STOP_CHARGE")) {
        isCharging = false;
        availableSlots = min(totalSlots, availableSlots + 1);
        Serial.println(">> Charging stopped");
    }
    else if (message.startsWith("EMERGENCY_STOP")) {
        isOperational = false;
        isCharging = false;
        Serial.println(">> EMERGENCY STOP!");
        flashRed(5);
    }
    else if (message.startsWith("RESET")) {
        isOperational = true;
        isCharging = false;
        availableSlots = totalSlots;
        Serial.println(">> Station reset");
    }
    else if (message.startsWith("SLOT_UPDATE,")) {
        availableSlots = message.substring(12).toInt();
    }
    
    updateLeds();
    sendStatus();
}

void sendStatus() {
    String topic = "chargeup/station/" + String(STATION_ID) + "/status";
    
    String payload = "{";
    payload += "\"station_id\":\"" + String(STATION_ID) + "\",";
    payload += "\"operational\":" + String(isOperational ? "true" : "false") + ",";
    payload += "\"available_slots\":" + String(availableSlots) + ",";
    payload += "\"total_slots\":" + String(totalSlots) + ",";
    payload += "\"charging\":" + String(isCharging ? "true" : "false");
    payload += "}";
    
    mqtt.publish(topic.c_str(), payload.c_str());
}

void updateLeds() {
    if (!isOperational) {
        digitalWrite(LED_RED, HIGH);
        digitalWrite(LED_YELLOW, LOW);
        digitalWrite(LED_GREEN, LOW);
        digitalWrite(LED_BLUE, LOW);
        return;
    }
    
    digitalWrite(LED_RED, LOW);
    digitalWrite(LED_BLUE, isCharging ? HIGH : LOW);
    digitalWrite(LED_GREEN, availableSlots > 0 ? HIGH : LOW);
    digitalWrite(LED_YELLOW, availableSlots == 0 ? HIGH : LOW);
}

void flashRed(int times) {
    for (int i = 0; i < times; i++) {
        digitalWrite(LED_RED, HIGH);
        delay(200);
        digitalWrite(LED_RED, LOW);
        delay(200);
    }
}

void allLedsOff() {
    digitalWrite(LED_RED, LOW);
    digitalWrite(LED_YELLOW, LOW);
    digitalWrite(LED_GREEN, LOW);
    digitalWrite(LED_BLUE, LOW);
}
