// esp32_station_controller.ino
// Unified Firmware for Charging Station Module (ESP32 only)

#include <WiFi.h>
#include <PubSubClient.h>
#include <LiquidCrystal_I2C.h>
#include <FastLED.h>
#include <ArduinoJson.h>

// --- Pin definitions ---
#define LED_PIN 14        // WS2812B Data pin
#define BUTTON_PIN 32     // GPIO 32 for "Car Connected" simulation
#define NUM_LEDS 30       // Number of LEDs in the strip

// --- Network Configuration ---
const char* ssid = "ChargeUp-Demo";
const char* password = "chargeup123";
const char* mqtt_server = "192.168.1.10"; // <--- CHANGE to your Raspberry Pi's IP

// --- Device Configuration ---
// !!! IMPORTANT: CHANGE FOR EACH STATION (e.g., "STATION01", "STATION02", "STATION03") !!!
String stationID = "STATION01"; 
int maxSlots = 1; 
bool isOperational = true;

// --- Hardware Objects ---
LiquidCrystal_I2C lcd(0x27, 16, 2); // Check I2C address (0x27 or 0x3F)
CRGB leds[NUM_LEDS];
WiFiClient wifiClient;
PubSubClient mqttClient(wifiClient);

// --- Station State and Queue Management (Local Copy) ---
struct QueueEntry {
  String carID;
  int priority;
  unsigned long arrivalTime;
  int urgency;
};

#define MAX_QUEUE_SIZE 10
QueueEntry chargingQueue[MAX_QUEUE_SIZE];
int queueLength = 0;
String currentlyChargingCar = ""; // Only one slot for simplicity (maxSlots=1)
bool carPresent = false; // State of the physical button/sensor

void setup() {
  Serial.begin(115200);
  
  // Initialize LCD
  lcd.init();
  lcd.backlight();
  lcd.setCursor(0, 0);
  lcd.print("ChargeUp Station");
  lcd.setCursor(0, 1);
  lcd.print("Initializing...");
  
  // Initialize Button (using internal PULLUP)
  pinMode(BUTTON_PIN, INPUT_PULLUP);
  
  // Initialize LEDs
  FastLED.addLeds<WS2812B, LED_PIN, GRB>(leds, NUM_LEDS);
  FastLED.setBrightness(80);
  
  // Network setup
  setupWiFi();
  mqttClient.setServer(mqtt_server, 1883);
  mqttClient.setCallback(onMqttMessage);
  connectMQTT();
  
  lcd.clear();
  lcd.setCursor(0, 0);
  lcd.print(stationID);
  lcd.setCursor(0, 1);
  lcd.print("Ready");
  
  sendStationStatus();
}

void loop() {
  if (!mqttClient.connected()) {
    connectMQTT();
  }
  mqttClient.loop();
  
  monitorButtonInput();
  monitorChargingState();
  updateDisplay();
  updateStatusLEDs();
  
  // Send periodic status
  static unsigned long lastStatusUpdate = 0;
  if (millis() - lastStatusUpdate > 5000) { 
    sendStationStatus();
    lastStatusUpdate = millis();
  }
  
  delay(100); 
}

void monitorButtonInput() {
    static bool lastButtonState = HIGH; // HIGH due to PULLUP
    bool currentButtonState = digitalRead(BUTTON_PIN);
    
    if (currentButtonState == LOW && lastButtonState == HIGH) {
        // Button Pressed (Car Arrived)
        carPresent = true;
        Serial.println("Car Arrived (Button Pressed)");
        
        if (currentlyChargingCar == "" && isOperational && queueLength > 0) {
            // If spot is free and we have a queue, automatically start charging the first car
            startChargingNextCar();
        }

    } else if (currentButtonState == HIGH && lastButtonState == LOW) {
        // Button Released (Car Left)
        carPresent = false;
        Serial.println("Car Left (Button Released)");
    }
    lastButtonState = currentButtonState;
}

void monitorChargingState() {
    // If car was charging, but physical presence is gone, stop charging.
    if (currentlyChargingCar != "" && !carPresent) {
        Serial.print(currentlyChargingCar + " left the station.");
        
        // Command car to stop charging
        mqttClient.publish(("chargeup/commands/" + currentlyChargingCar).c_str(), "CHARGE,STOP");
        currentlyChargingCar = "";
        
        // Slot is now free, inform RPi (handled by periodic status)
    }
}

void startChargingNextCar() {
    if (queueLength > 0 && currentlyChargingCar == "") {
        String nextCarID = chargingQueue[0].carID;

        // 1. Move car from queue to charging slot
        currentlyChargingCar = nextCarID;
        removeFromQueue(nextCarID); // Remove from local queue
        
        // 2. Command car to start charging
        mqttClient.publish(("chargeup/commands/" + currentlyChargingCar).c_str(), "CHARGE,START");

        Serial.print("Started charging: ");
        Serial.println(nextCarID);
    }
}


// --- MQTT and Command Processing ---

void onMqttMessage(char* topic, byte* payload, unsigned int length) {
  String message;
  for (int i = 0; i < length; i++) {
    message += (char)payload[i];
  }
  
  String topicStr = String(topic);
  Serial.print("MQTT Rcv: "); Serial.print(topicStr); Serial.print(" -> "); Serial.println(message);
  
  if (topicStr.endsWith("/commands")) {
    // Admin commands (EMERGENCY_STOP, RESET)
    processStationCommand(message);
  } else if (topicStr.endsWith("/queue_command")) {
    // Queue management commands (ADD, REMOVE, SWAP)
    processQueueUpdate(message);
    // After any queue change, check if we should start charging
    if (currentlyChargingCar == "" && carPresent) {
        startChargingNextCar();
    }
  }
}

void processStationCommand(String command) {
  if (command == "EMERGENCY_STOP") {
    isOperational = false;
    currentlyChargingCar = "";
    // Optionally publish STOP to any charging car, but RPi should manage overall state
    Serial.println("EMERGENCY STOP!");
  } else if (command == "RESET") {
    isOperational = true;
    queueLength = 0;
    currentlyChargingCar = "";
    Serial.println("Station RESET.");
  }
}

void processQueueUpdate(String queueData) {
  DynamicJsonDocument doc(256); 
  DeserializationError error = deserializeJson(doc, queueData);

  if (error) {
    Serial.print(F("deserializeJson() failed: "));
    Serial.println(error.f_str());
    return;
  }
  
  String action = doc["action"].as<String>();
  String carID = doc["carID"].as<String>();
  
  if (action == "ADD") {
    int priority = doc["priority"] | 5; 
    int urgency = doc["urgency"] | 5; 
    addToQueue(carID, priority, urgency);
  } else if (action == "REMOVE") {
    removeFromQueue(carID);
  } else if (action == "SWAP") {
    String carID2 = doc["carID2"].as<String>();
    swapQueuePositions(carID, carID2);
  }
  sortQueue();
  sendStationStatus(); 
}

void addToQueue(String carID, int priority, int urgency) {
  for (int i=0; i<queueLength; i++) {
    if (chargingQueue[i].carID == carID) return;
  }

  if (queueLength < MAX_QUEUE_SIZE) {
    chargingQueue[queueLength] = {carID, priority, millis(), urgency};
    queueLength++;
    Serial.print(carID + " added to queue. Length: "); Serial.println(queueLength);
  } 
}

void removeFromQueue(String carID) {
  for (int i = 0; i < queueLength; i++) {
    if (chargingQueue[i].carID == carID) {
      for (int j = i; j < queueLength - 1; j++) {
        chargingQueue[j] = chargingQueue[j + 1];
      }
      queueLength--;
      Serial.print(carID + " removed from queue. Length: "); Serial.println(queueLength);
      return;
    }
  }
}

void swapQueuePositions(String carID1, String carID2) {
  int pos1 = -1, pos2 = -1;
  for (int i = 0; i < queueLength; i++) {
    if (chargingQueue[i].carID == carID1) pos1 = i;
    if (chargingQueue[i].carID == carID2) pos2 = i;
  }

  if (pos1 != -1 && pos2 != -1) {
    QueueEntry temp = chargingQueue[pos1];
    chargingQueue[pos1] = chargingQueue[pos2];
    chargingQueue[pos2] = temp;
    Serial.println("Swapped " + carID1 + " and " + carID2 + " in queue.");
  }
}

void sortQueue() {
  // Simple bubble sort by Priority (higher first), then Urgency (higher first), then Arrival Time (earlier first)
  for (int i = 0; i < queueLength - 1; i++) {
    for (int j = 0; j < queueLength - i - 1; j++) {
      bool shouldSwap = false;
      // Primary sort: Priority (higher is better)
      if (chargingQueue[j].priority < chargingQueue[j + 1].priority) {
        shouldSwap = true;
      }
      // Secondary sort: Urgency
      else if (chargingQueue[j].priority == chargingQueue[j + 1].priority &&
               chargingQueue[j].urgency < chargingQueue[j + 1].urgency) {
        shouldSwap = true;
      }
      // Tertiary sort: Arrival Time
      else if (chargingQueue[j].priority == chargingQueue[j + 1].priority &&
               chargingQueue[j].urgency == chargingQueue[j + 1].urgency &&
               chargingQueue[j].arrivalTime > chargingQueue[j + 1].arrivalTime) {
        shouldSwap = true;
      }
      
      if (shouldSwap) {
          QueueEntry temp = chargingQueue[j];
          chargingQueue[j] = chargingQueue[j + 1];
          chargingQueue[j + 1] = temp;
      }
    }
  }
}

// --- Display and LED Functions (Simplified) ---

void updateDisplay() {
  lcd.clear();
  lcd.setCursor(0, 0);
  lcd.print(stationID.substring(7) + ": "); 
  
  if (!isOperational) {
    lcd.print("OUT OF ORDER");
    lcd.setCursor(0, 1);
    lcd.print("Queue: --");
    return;
  }
  
  lcd.print("Queue: " + String(queueLength));
  lcd.setCursor(0, 1);
  if (currentlyChargingCar != "") {
      lcd.print("CHARGING:" + currentlyChargingCar.substring(3));
  } else if (carPresent) {
      lcd.print("WAITING FOR SLOT");
  } else if (queueLength > 0) {
      lcd.print("NEXT:" + chargingQueue[0].carID.substring(3));
  } else {
      lcd.print("AVAILABLE");
  }
}

void updateStatusLEDs() {
  if (!isOperational) {
    fill_solid(leds, NUM_LEDS, CRGB::Red); 
  } else if (currentlyChargingCar != "") {
    fill_solid(leds, NUM_LEDS, CRGB::Blue); // Blue when charging
  } else if (queueLength == 0) {
    fill_solid(leds, NUM_LEDS, CRGB::Green); 
  } else {
    fill_solid(leds, NUM_LEDS, CRGB::Yellow);
  }
  
  // Indicate queue length
  int queueIndicatorLEDs = map(queueLength, 0, MAX_QUEUE_SIZE, 0, NUM_LEDS / 2); 
  for (int i = 0; i < queueIndicatorLEDs; i++) {
    leds[i] = CRGB::Orange;
  }
  
  FastLED.show();
}

void sendStationStatus() {
  String statusTopic = "chargeup/station/" + stationID + "/status";
  
  DynamicJsonDocument doc(1024);
  doc["stationID"] = stationID;
  doc["operational"] = isOperational;
  doc["queueLength"] = queueLength;
  doc["maxSlots"] = maxSlots;
  doc["timestamp"] = String(millis()); 
  
  JsonArray chargingCars = doc.createNestedArray("chargingCars");
  if (currentlyChargingCar != "") chargingCars.add(currentlyChargingCar);

  JsonArray queue = doc.createNestedArray("queue");
  for (int i = 0; i < queueLength; i++) {
    JsonObject entry = queue.createNestedObject();
    entry["carID"] = chargingQueue[i].carID;
    entry["priority"] = chargingQueue[i].priority;
    entry["urgency"] = chargingQueue[i].urgency;
    entry["arrivalTime"] = chargingQueue[i].arrivalTime; 
    entry["waitTime"] = (millis() - chargingQueue[i].arrivalTime) / 1000;
  }
  
  String jsonString;
  serializeJson(doc, jsonString);
  
  mqttClient.publish(statusTopic.c_str(), jsonString.c_str());
}

// --- WiFi Configuration ---

void setupWiFi() {
  Serial.print("Connecting to WiFi ");
  Serial.println(ssid);
  WiFi.begin(ssid, password);

  // Set static IP
  IPAddress local_IP(192, 168, 1, stationID.substring(7).toInt() + 30); // STATION01 -> .31
  IPAddress gateway(192, 168, 1, 1); 
  IPAddress subnet(255, 255, 255, 0);
  IPAddress primaryDNS(8, 8, 8, 8); 

  if (!WiFi.config(local_IP, gateway, subnet, primaryDNS)) {
    Serial.println("STA Failed to configure");
  }
  
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }
  Serial.println("\nWiFi connected");
  Serial.print("IP Address: ");
  Serial.println(WiFi.localIP());
}

void connectMQTT() {
  while (!mqttClient.connected()) {
    Serial.print("Attempting MQTT connection...");
    if (mqttClient.connect(stationID.c_str())) {
      Serial.println("connected");
      
      String commandTopic = "chargeup/station/" + stationID + "/commands";
      mqttClient.subscribe(commandTopic.c_str());
      
      String queueTopic = "chargeup/station/" + stationID + "/queue_command";
      mqttClient.subscribe(queueTopic.c_str());
      
      mqttClient.publish(("chargeup/status/" + stationID).c_str(), "ONLINE");
      
    } else {
      Serial.print("failed, rc=");
      Serial.print(mqttClient.state());
      Serial.println(" retrying in 5 seconds");
      delay(5000);
    }
  }
}