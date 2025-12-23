/*
 * ChargeUp EV - Enhanced Car Controller v2.0
 * ESP32 with QR Generation, AI Battery Monitoring, Smart Alerts
 * 
 * CONFIGURATION: Change deviceID for each car (CAR01, CAR02, etc.)
 */

#include <WiFi.h>
#include <PubSubClient.h>
#include <FastLED.h>
#include <ArduinoJson.h>
#include <qrcode.h>

// ========== CONFIGURATION ==========
const char* ssid = "ChargeUp-Demo";
const char* password = "chargeup123";
const char* mqtt_server = "192.168.1.10";
String deviceID = "CAR01";  // CHANGE THIS FOR EACH CAR

// ========== PINS ==========
#define LED_PIN 13
#define NUM_LEDS 15

// ========== STATE VARIABLES ==========
int batteryLevel = 85;
float batteryHealth = 100.0;
int cycleCount = 0;
String currentStatus = "IDLE";
bool isCharging = false;
float currentRange = 340.0;
float lat = 9.9312;
float lon = 76.2673;

// QR Code Management
String currentQRData = "";
unsigned long qrGenerationTime = 0;
const unsigned long QR_TIMEOUT = 300000;  // 5 minutes
bool qrActive = false;

// Charging Session
unsigned long chargingStartTime = 0;
int allocatedChargeMinutes = 0;
bool chargingSessionActive = false;

// Hardware
CRGB leds[NUM_LEDS];
WiFiClient wifiClient;
PubSubClient mqttClient(wifiClient);

unsigned long lastTelemetryTime = 0;
unsigned long lastBatteryUpdate = 0;

void setup() {
  Serial.begin(115200);
  Serial.println("\n========================================");
  Serial.println("ChargeUp EV v2.0 - " + deviceID);
  Serial.println("========================================\n");
  
  FastLED.addLeds<WS2812B, LED_PIN, GRB>(leds, NUM_LEDS);
  FastLED.setBrightness(80);
  bootAnimation();
  
  setupWiFi();
  mqttClient.setServer(mqtt_server, 1883);
  mqttClient.setCallback(onMqttMessage);
  mqttClient.setBufferSize(1024);
  connectMQTT();
  
  updateRange();
  sendTelemetry();
}

void loop() {
  if (!mqttClient.connected()) connectMQTT();
  mqttClient.loop();
  
  if (millis() - lastBatteryUpdate >= 1000) {
    simulateBatteryDrain();
    checkChargeTimeout();
    lastBatteryUpdate = millis();
  }
  
  if (qrActive && (millis() - qrGenerationTime > QR_TIMEOUT)) {
    expireQRCode();
  }
  
  if (batteryLevel == 20 && currentStatus != "CHARGING") {
    requestRerouting();
  }
  
  if (millis() - lastTelemetryTime >= 2000) {
    sendTelemetry();
    lastTelemetryTime = millis();
  }
  
  updateStatusLEDs();
  delay(50);
}

// ========== QR CODE GENERATION ==========
void generateChargingQR(int requestedMinutes, String stationID) {
  unsigned long timestamp = millis();
  currentQRData = deviceID + "|" + stationID + "|" + String(timestamp) + "|" + 
                  String(requestedMinutes) + "|" + String(batteryLevel);
  
  qrGenerationTime = timestamp;
  qrActive = true;
  
  Serial.println("\n🔲 QR CODE GENERATED");
  Serial.println("━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
  Serial.println("Data: " + currentQRData);
  Serial.println("Valid for: 5 minutes");
  Serial.println("Station: " + stationID);
  
  QRCode qrcode;
  uint8_t qrcodeData[qrcode_getBufferSize(3)];
  qrcode_initText(&qrcode, qrcodeData, 3, 0, currentQRData.c_str());
  
  Serial.println("\nSCAN THIS QR CODE:");
  for (uint8_t y = 0; y < qrcode.size; y++) {
    for (uint8_t x = 0; x < qrcode.size; x++) {
      Serial.print(qrcode_getModule(&qrcode, x, y) ? "██" : "  ");
    }
    Serial.println();
  }
  Serial.println("━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");
  
  // Publish to MQTT for app display
  StaticJsonDocument<512> doc;
  doc["car_id"] = deviceID;
  doc["qr_data"] = currentQRData;
  doc["generated_at"] = timestamp;
  doc["expires_at"] = timestamp + QR_TIMEOUT;
  doc["station_id"] = stationID;
  doc["charge_time_min"] = requestedMinutes;
  doc["current_battery"] = batteryLevel;
  
  String jsonString;
  serializeJson(doc, jsonString);
  mqttClient.publish(("chargeup/qr/" + deviceID).c_str(), jsonString.c_str());
  
  // Visual feedback
  for (int i = 0; i < 3; i++) {
    fill_solid(leds, NUM_LEDS, CRGB::Cyan);
    FastLED.show();
    delay(200);
    fill_solid(leds, NUM_LEDS, CRGB::Black);
    FastLED.show();
    delay(200);
  }
}

void expireQRCode() {
  if (qrActive) {
    Serial.println("⏱️  QR CODE EXPIRED");
    qrActive = false;
    currentQRData = "";
    
    StaticJsonDocument<128> doc;
    doc["car_id"] = deviceID;
    doc["status"] = "expired";
    String jsonString;
    serializeJson(doc, jsonString);
    mqttClient.publish(("chargeup/qr/" + deviceID + "/status").c_str(), jsonString.c_str());
  }
}

// ========== CHARGING SESSION MANAGEMENT ==========
void startChargingSession(int allocatedMinutes) {
  chargingStartTime = millis();
  allocatedChargeMinutes = allocatedMinutes;
  chargingSessionActive = true;
  isCharging = true;
  currentStatus = "CHARGING";
  
  Serial.println("🔌 CHARGING SESSION STARTED");
  Serial.println("  Allocated Time: " + String(allocatedMinutes) + " minutes");
  Serial.println("  Current Battery: " + String(batteryLevel) + "%");
}

void checkChargeTimeout() {
  if (!chargingSessionActive || !isCharging) return;
  
  unsigned long elapsedMinutes = (millis() - chargingStartTime) / 60000;
  
  if (elapsedMinutes > allocatedChargeMinutes) {
    int overtimeMinutes = elapsedMinutes - allocatedChargeMinutes;
    
    // Publish overtime billing request
    StaticJsonDocument<256> doc;
    doc["car_id"] = deviceID;
    doc["allocated_minutes"] = allocatedChargeMinutes;
    doc["actual_minutes"] = elapsedMinutes;
    doc["overtime_minutes"] = overtimeMinutes;
    doc["battery_level"] = batteryLevel;
    
    String jsonString;
    serializeJson(doc, jsonString);
    mqttClient.publish("chargeup/billing/overtime", jsonString.c_str());
    
    Serial.println("⏰ OVERTIME DETECTED: +" + String(overtimeMinutes) + " min");
    Serial.println("   Additional parking + charging fees apply");
  }
}

// ========== BATTERY MANAGEMENT ==========
void simulateBatteryDrain() {
  if (isCharging) {
    if (batteryLevel < 100) {
      batteryLevel += 2;  // 2% per second while charging
      if (batteryLevel >= 100) {
        batteryLevel = 100;
        isCharging = false;
        chargingSessionActive = false;
        currentStatus = "IDLE";
        cycleCount++;
        updateBatteryHealth();
        Serial.println("✅ FULLY CHARGED! Cycle count: " + String(cycleCount));
      }
    }
  } else {
    if (currentStatus == "MOVING") {
      batteryLevel = max(0, batteryLevel - 1);  // 1% per second
    } else {
      static unsigned long lastIdleDrain = 0;
      if (millis() - lastIdleDrain >= 5000) {
        batteryLevel = max(0, batteryLevel - 1);  // 1% per 5 seconds idle
        lastIdleDrain = millis();
      }
    }
  }
  
  updateRange();
  
  if (batteryLevel <= 15 && currentStatus != "CHARGING") {
    currentStatus = "LOW_BATTERY";
  }
}

void updateBatteryHealth() {
  float healthDegradation = cycleCount * 0.05;
  batteryHealth = max(50.0, 100.0 - healthDegradation);
  
  if (batteryHealth < 80.0) {
    publishHealthAlert("warning");
  }
}

void publishHealthAlert(String severity) {
  StaticJsonDocument<256> doc;
  doc["car_id"] = deviceID;
  doc["battery_health"] = batteryHealth;
  doc["cycle_count"] = cycleCount;
  doc["severity"] = severity;
  doc["recommendation"] = (severity == "critical") ? 
    "Immediate battery service required" : 
    "Schedule battery inspection";
  
  String jsonString;
  serializeJson(doc, jsonString);
  mqttClient.publish(("chargeup/health/" + deviceID).c_str(), jsonString.c_str());
}

void updateRange() {
  const float MAX_RANGE_KM = 400.0;
  currentRange = (batteryLevel / 100.0) * MAX_RANGE_KM * (batteryHealth / 100.0);
}

// ========== REROUTING ==========
void requestRerouting() {
  Serial.println("\n⚠️  CRITICAL: 20% Battery - Requesting reroute");
  
  StaticJsonDocument<256> doc;
  doc["car_id"] = deviceID;
  doc["battery_level"] = batteryLevel;
  doc["current_lat"] = lat;
  doc["current_lon"] = lon;
  doc["current_range_km"] = currentRange;
  doc["urgency"] = 9;
  
  String jsonString;
  serializeJson(doc, jsonString);
  mqttClient.publish("chargeup/rerouting/request", jsonString.c_str());
  
  playAlertAnimation();
}

// ========== MQTT ==========
void onMqttMessage(char* topic, byte* payload, unsigned int length) {
  String message = "";
  for (int i = 0; i < length; i++) {
    message += (char)payload[i];
  }
  
  Serial.println("📨 MQTT: " + String(topic) + " -> " + message);
  
  if (String(topic).endsWith("/commands")) {
    processCommand(message);
  } else if (String(topic).endsWith("/qr/generate")) {
    StaticJsonDocument<256> doc;
    deserializeJson(doc, message);
    generateChargingQR(doc["charge_minutes"] | 30, doc["station_id"] | "STATION01");
  } else if (String(topic).endsWith("/charging/start")) {
    StaticJsonDocument<256> doc;
    deserializeJson(doc, message);
    startChargingSession(doc["allocated_minutes"] | 30);
  }
}

void processCommand(String cmd) {
  if (cmd.startsWith("MOVE")) {
    currentStatus = "MOVING";
    Serial.println("🚗 Moving...");
    delay(2000);
    currentStatus = "IDLE";
  } else if (cmd == "CHARGE,START") {
    isCharging = true;
    currentStatus = "CHARGING";
    Serial.println("🔌 Charging started");
  } else if (cmd == "CHARGE,STOP") {
    isCharging = false;
    chargingSessionActive = false;
    currentStatus = "IDLE";
    Serial.println("🔋 Charging stopped");
  } else if (cmd.startsWith("QR_GENERATE")) {
    int comma = cmd.indexOf(',');
    String stationID = cmd.substring(comma + 1);
    generateChargingQR(30, stationID);
  }
}

void sendTelemetry() {
  if (!mqttClient.connected()) return;
  
  StaticJsonDocument<512> doc;
  doc["device"] = deviceID;
  doc["timestamp"] = String(millis());
  
  String dataString = "BATTERY:" + String(batteryLevel) + ",";
  dataString += "STATUS:" + currentStatus + ",";
  dataString += "RANGE:" + String(currentRange, 1) + ",";
  dataString += "HEALTH:" + String(batteryHealth, 1) + ",";
  dataString += "LAT:" + String(lat, 6) + ",";
  dataString += "LON:" + String(lon, 6);
  
  doc["data"] = dataString;
  doc["battery_level"] = batteryLevel;
  doc["battery_health"] = batteryHealth;
  doc["range_km"] = currentRange;
  doc["lat"] = lat;
  doc["lon"] = lon;
  doc["charging"] = isCharging;
  doc["qr_active"] = qrActive;
  
  String jsonString;
  serializeJson(doc, jsonString);
  mqttClient.publish(("chargeup/telemetry/" + deviceID).c_str(), jsonString.c_str());
}

// ========== LED ANIMATIONS ==========
void updateStatusLEDs() {
  if (currentStatus == "CHARGING") {
    int brightness = (sin(millis() / 300.0) + 1) * 127;
    fill_solid(leds, NUM_LEDS, CRGB(0, brightness, 0));
  } else if (currentStatus == "MOVING") {
    fadeToBlackBy(leds, NUM_LEDS, 40);
    int pos = (millis() / 80) % NUM_LEDS;
    leds[pos] = CRGB::Blue;
  } else if (currentStatus == "LOW_BATTERY") {
    fill_solid(leds, NUM_LEDS, (millis() % 800 < 400) ? CRGB::Red : CRGB::Black);
  } else {
    fill_solid(leds, NUM_LEDS, CRGB(0, 0, 50));
  }
  FastLED.show();
}

void bootAnimation() {
  for (int i = 0; i < NUM_LEDS; i++) {
    leds[i] = CHSV(i * 255 / NUM_LEDS, 255, 200);
    FastLED.show();
    delay(50);
  }
  delay(500);
  fill_solid(leds, NUM_LEDS, CRGB::Black);
  FastLED.show();
}

void playAlertAnimation() {
  for (int i = 0; i < 5; i++) {
    fill_solid(leds, NUM_LEDS, CRGB::Red);
    FastLED.show();
    delay(150);
    fill_solid(leds, NUM_LEDS, CRGB::Black);
    FastLED.show();
    delay(150);
  }
}

// ========== WIFI ==========
void setupWiFi() {
  Serial.print("Connecting to WiFi...");
  WiFi.begin(ssid, password);
  
  int carNumber = deviceID.substring(3).toInt();
  IPAddress local_IP(192, 168, 1, 20 + carNumber);
  IPAddress gateway(192, 168, 1, 1);
  IPAddress subnet(255, 255, 255, 0);
  WiFi.config(local_IP, gateway, subnet);
  
  while (WiFi.status() != WL_CONNECTED && millis() < 15000) {
    delay(500);
    Serial.print(".");
  }
  
  if (WiFi.status() == WL_CONNECTED) {
    Serial.println("\n✅ WiFi connected: " + WiFi.localIP().toString());
  } else {
    Serial.println("\n❌ WiFi failed");
  }
}

void connectMQTT() {
  int attempts = 0;
  while (!mqttClient.connected() && attempts < 5) {
    Serial.print("Connecting MQTT...");
    if (mqttClient.connect((deviceID + "_" + String(random(0xffff), HEX)).c_str())) {
      Serial.println("✅ Connected");
      mqttClient.subscribe(("chargeup/commands/" + deviceID).c_str());
      mqttClient.subscribe(("chargeup/qr/" + deviceID + "/generate").c_str());
      mqttClient.subscribe(("chargeup/charging/" + deviceID + "/start").c_str());
      mqttClient.publish(("chargeup/status/" + deviceID).c_str(), "ONLINE");
    } else {
      Serial.println("❌ Failed, retry...");
      delay(3000);
      attempts++;
    }
  }
}
