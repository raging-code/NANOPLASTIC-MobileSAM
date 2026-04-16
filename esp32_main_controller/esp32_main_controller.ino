/*
 * ESP32 MAIN CONTROLLER – Asynchronous capture support
 */

#include <WiFi.h>
#include <WebServer.h>
#include <EEPROM.h>
#include <HTTPClient.h>

// ==================== CONFIGURATION ====================
const char* ssid = "HUAWEI-2.4G-SW5u";
const char* password = "NargatanFMLY";
const char* pcServerIP = "192.168.18.2";
const int pcServerPort = 5000;

// ==================== PIN DEFINITIONS ====================
#define MOTOR_A_IA 27
#define MOTOR_A_IB 26
#define MOTOR_B_IA 25
#define MOTOR_B_IB 33
#define LASER_PIN 13

#define EEPROM_SIZE 64
int pumpARotations = 3;
int pumpBRotations = 3;

WebServer server(80);

void loadSettings() {
  EEPROM.begin(EEPROM_SIZE);
  EEPROM.get(0, pumpARotations);
  EEPROM.get(4, pumpBRotations);
  if (pumpARotations < 1 || pumpARotations > 20) pumpARotations = 3;
  if (pumpBRotations < 1 || pumpBRotations > 20) pumpBRotations = 3;
  Serial.printf("Loaded: A=%d, B=%d\n", pumpARotations, pumpBRotations);
}

void saveSettings() {
  EEPROM.put(0, pumpARotations);
  EEPROM.put(4, pumpBRotations);
  EEPROM.commit();
  Serial.println("Settings saved");
}

void stopPumpA() { digitalWrite(MOTOR_A_IA, LOW); digitalWrite(MOTOR_A_IB, LOW); }
void stopPumpB() { digitalWrite(MOTOR_B_IA, LOW); digitalWrite(MOTOR_B_IB, LOW); }

void runPumpA(int rotations) {
  Serial.printf("Running Pump A: %d rot\n", rotations);
  analogWrite(MOTOR_A_IA, 200);
  digitalWrite(MOTOR_A_IB, LOW);
  delay(rotations * 1000);
  stopPumpA();
}

void runPumpB(int rotations) {
  Serial.printf("Running Pump B: %d rot\n", rotations);
  analogWrite(MOTOR_B_IA, 200);
  digitalWrite(MOTOR_B_IB, LOW);
  delay(rotations * 1000);
  stopPumpB();
}

void runBothPumps() {
  runPumpA(pumpARotations);
  delay(500);
  runPumpB(pumpBRotations);
  Serial.println("Mixing done");
}

void setLaser(bool on) {
  digitalWrite(LASER_PIN, on ? HIGH : LOW);
  Serial.printf("Laser %s\n", on ? "ON" : "OFF");
}

bool testPCConnection() {
  if (WiFi.status() != WL_CONNECTED) return false;
  HTTPClient http;
  String url = "http://" + String(pcServerIP) + ":" + String(pcServerPort) + "/ping";
  http.begin(url);
  http.setTimeout(3000);
  int code = http.GET();
  http.end();
  return code == 200;
}

// Trigger asynchronous capture (returns immediately)
bool triggerCaptureAsync() {
  if (WiFi.status() != WL_CONNECTED) return false;
  HTTPClient http;
  String url = "http://" + String(pcServerIP) + ":" + String(pcServerPort) + "/trigger_capture";
  Serial.print("Async capture URL: ");
  Serial.println(url);
  http.begin(url);
  http.setTimeout(5000); // short timeout because it returns immediately
  int code = http.GET();
  Serial.print("HTTP Response: ");
  Serial.println(code);
  http.end();
  return (code == 202); // 202 Accepted
}

// Poll capture status
String getCaptureStatus() {
  if (WiFi.status() != WL_CONNECTED) return "";
  HTTPClient http;
  String url = "http://" + String(pcServerIP) + ":" + String(pcServerPort) + "/api/capture_status";
  http.begin(url);
  http.setTimeout(3000);
  int code = http.GET();
  if (code == 200) {
    String payload = http.getString();
    http.end();
    return payload;
  }
  http.end();
  return "";
}

unsigned long lastWiFiCheck = 0;

void setup() {
  Serial.begin(115200);
  Serial.println("\nESP32 Controller (Async Capture)");
  loadSettings();
  pinMode(MOTOR_A_IA, OUTPUT); pinMode(MOTOR_A_IB, OUTPUT);
  pinMode(MOTOR_B_IA, OUTPUT); pinMode(MOTOR_B_IB, OUTPUT);
  pinMode(LASER_PIN, OUTPUT);
  stopPumpA(); stopPumpB(); digitalWrite(LASER_PIN, LOW);

  WiFi.begin(ssid, password);
  Serial.print("Connecting WiFi");
  int attempts = 0;
  while (WiFi.status() != WL_CONNECTED && attempts < 30) {
    delay(500); Serial.print("."); attempts++;
  }
  if (WiFi.status() != WL_CONNECTED) { Serial.println("\nWiFi Failed"); ESP.restart(); }
  Serial.println("\nWiFi Connected");
  Serial.print("ESP32 IP: "); Serial.println(WiFi.localIP());
  Serial.print("PC Server: http://"); Serial.print(pcServerIP); Serial.print(":"); Serial.println(pcServerPort);

  // API routes
  server.on("/api/settings", HTTP_GET, []() {
    String json = "{\"pumpA\":" + String(pumpARotations) + ",\"pumpB\":" + String(pumpBRotations) + "}";
    server.send(200, "application/json", json);
  });
  server.on("/api/settings", HTTP_POST, []() {
    if (server.hasArg("pumpA") && server.hasArg("pumpB")) {
      pumpARotations = server.arg("pumpA").toInt();
      pumpBRotations = server.arg("pumpB").toInt();
      saveSettings();
      server.send(200, "text/plain", "OK");
    } else server.send(400, "text/plain", "Missing");
  });
  server.on("/api/pump/a", HTTP_POST, []() { runPumpA(pumpARotations); server.send(200, "OK"); });
  server.on("/api/pump/b", HTTP_POST, []() { runPumpB(pumpBRotations); server.send(200, "OK"); });
  server.on("/api/pump/both", HTTP_POST, []() { runBothPumps(); server.send(200, "OK"); });
  server.on("/api/laser/on", HTTP_POST, []() { setLaser(true); server.send(200, "OK"); });
  server.on("/api/laser/off", HTTP_POST, []() { setLaser(false); server.send(200, "OK"); });
  
  // New async capture endpoint
  server.on("/api/capture", HTTP_POST, []() {
    Serial.println("Async capture requested");
    if (triggerCaptureAsync()) {
      server.send(202, "text/plain", "Capture started");
    } else {
      server.send(500, "text/plain", "Failed to start capture");
    }
  });
  
  server.on("/api/test-pc", HTTP_GET, []() {
    if (testPCConnection()) server.send(200, "text/plain", "PC reachable");
    else server.send(500, "text/plain", "PC not reachable");
  });

  // Main UI – consolidated widget
  server.on("/", []() {
    String html = R"rawliteral(
<!DOCTYPE html>
<html lang='en'>
<head>
    <meta charset='UTF-8'>
    <meta name='viewport' content='width=device-width, initial-scale=1.0'>
    <title>ESP32 · Pump & Laser</title>
    <link href='https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap' rel='stylesheet'>
    <style>
        *{margin:0;padding:0;box-sizing:border-box;}
        body{font-family:'Inter',sans-serif;background:#0b1120;color:#e2e8f0;padding:2rem 1rem;min-height:100vh;display:flex;align-items:center;justify-content:center;}
        .controller-card{max-width:700px;width:100%;background:rgba(15,23,42,0.8);backdrop-filter:blur(12px);border:1px solid #1e293b;border-radius:32px;padding:2rem;box-shadow:0 20px 40px rgba(0,0,0,0.6);}
        h1{font-weight:600;font-size:2rem;color:#67e8f9;margin-bottom:0.5rem;letter-spacing:-0.02em;}
        .server-info{color:#94a3b8;margin-bottom:1.8rem;font-size:0.9rem;}
        .stats-mini{display:flex;gap:1rem;margin-bottom:1.5rem;background:#0f172a;border-radius:20px;padding:1rem;}
        .stat-item{flex:1;text-align:center;}
        .stat-value{font-size:1.8rem;font-weight:600;color:#67e8f9;}
        .stat-label{font-size:0.7rem;text-transform:uppercase;color:#64748b;}
        .pump-section{display:grid;grid-template-columns:1fr 1fr;gap:1.5rem;margin-bottom:1.5rem;}
        .pump-card{background:#0f172a;border-radius:20px;padding:1.2rem;border:1px solid #1e293b;}
        .pump-title{font-weight:500;margin-bottom:0.75rem;color:#cbd5e1;}
        .pump-value{font-size:1.3rem;font-weight:600;color:#e2e8f0;margin-bottom:0.5rem;}
        .slider{width:100%;margin:1rem 0;}
        .btn-group{display:flex;gap:0.5rem;flex-wrap:wrap;margin-top:1rem;}
        .btn{background:transparent;border:1px solid #334155;color:#e2e8f0;padding:0.6rem 1rem;border-radius:40px;font-weight:500;cursor:pointer;transition:all 0.2s;font-size:0.9rem;}
        .btn-primary{background:#0ea5e9;border-color:#0ea5e9;color:#fff;}
        .btn-primary:hover{background:#0284c7;}
        .btn:hover{background:#1e293b;}
        .laser-panel{display:flex;gap:1rem;align-items:center;margin-top:1.5rem;padding-top:1.5rem;border-top:1px solid #1e293b;}
        .status-badge{background:#0f172a;border-radius:40px;padding:0.4rem 1rem;font-size:0.8rem;color:#94a3b8;}
        .alert{margin-top:1rem;padding:0.8rem;border-radius:20px;background:#1e293b;}
    </style>
</head>
<body>
<div class='controller-card'>
    <h1>⚙️ ESP32 Controller</h1>
    <div class='server-info'>PC Server: )rawliteral" + String(pcServerIP) + ":" + String(pcServerPort) + R"rawliteral(</div>
    
    <!-- Mini Stats (fetched from PC) -->
    <div class='stats-mini'>
        <div class='stat-item'><div class='stat-value' id='totalSamples'>-</div><div class='stat-label'>Samples</div></div>
        <div class='stat-item'><div class='stat-value' id='highRiskCount'>-</div><div class='stat-label'>High Risk</div></div>
        <div class='stat-item'><div class='stat-value' id='avgParticles'>-</div><div class='stat-label'>Avg Parts</div></div>
    </div>
    
    <!-- Pumps in two columns -->
    <div class='pump-section'>
        <div class='pump-card'>
            <div class='pump-title'>💧 Pump A (Water)</div>
            <div class='pump-value'><span id='pumpAVal'>)rawliteral" + String(pumpARotations) + R"rawliteral(</span> rotations</div>
            <div>Volume: <span id='pumpAVol'>)rawliteral" + String(pumpARotations) + R"rawliteral(.0</span> ml</div>
            <input type='range' id='pumpASlider' class='slider' min='1' max='20' value=')rawliteral" + String(pumpARotations) + R"rawliteral('>
            <div class='btn-group'>
                <button class='btn' onclick='testPump("a")'>Test</button>
                <button class='btn btn-primary' onclick='saveDefaults()'>Save Default</button>
            </div>
        </div>
        <div class='pump-card'>
            <div class='pump-title'>🧪 Pump B (L‑MPN)</div>
            <div class='pump-value'><span id='pumpBVal'>)rawliteral" + String(pumpBRotations) + R"rawliteral(</span> rotations</div>
            <div>Volume: <span id='pumpBVol'>)rawliteral" + String(pumpBRotations) + R"rawliteral(.0</span> ml</div>
            <input type='range' id='pumpBSlider' class='slider' min='1' max='20' value=')rawliteral" + String(pumpBRotations) + R"rawliteral('>
            <div class='btn-group'>
                <button class='btn' onclick='testPump("b")'>Test</button>
                <button class='btn btn-primary' onclick='mixBoth()'>Mix Both</button>
            </div>
        </div>
    </div>
    
    <!-- Laser & Capture -->
    <div class='laser-panel'>
        <button class='btn' onclick='laserOn()'>🔴 Laser ON</button>
        <button class='btn' onclick='laserOff()'>⚫ Laser OFF</button>
        <button class='btn btn-primary' onclick='triggerCapture()'>📸 Capture & Analyze</button>
        <button class='btn' onclick='testPC()'>🔌 Test PC</button>
    </div>
    
    <div id='statusMessage' class='alert' style='display:none;'></div>
</div>

<script>
    const pcServer = 'http://)rawliteral" + String(pcServerIP) + ":" + String(pcServerPort) + R"rawliteral(';
    
    // Slider sync
    document.getElementById('pumpASlider').oninput = function(){
        let v = this.value;
        document.getElementById('pumpAVal').innerText = v;
        document.getElementById('pumpAVol').innerText = v+'.0';
    };
    document.getElementById('pumpBSlider').oninput = function(){
        let v = this.value;
        document.getElementById('pumpBVal').innerText = v;
        document.getElementById('pumpBVol').innerText = v+'.0';
    };
    
    function showStatus(msg, isGood=true){
        let el = document.getElementById('statusMessage');
        el.style.display = 'block';
        el.innerHTML = msg;
        el.style.background = isGood ? '#14532d' : '#7f1d1d';
        setTimeout(()=> el.style.display='none', 5000);
    }
    
    async function testPump(pump){
        showStatus('Running pump ' + pump.toUpperCase() + '...');
        await fetch('/api/pump/'+pump, {method:'POST'});
        showStatus('Pump ' + pump.toUpperCase() + ' done', true);
    }
    async function mixBoth(){
        showStatus('Mixing both pumps...');
        await fetch('/api/pump/both', {method:'POST'});
        showStatus('Mixing completed', true);
    }
    async function laserOn(){
        await fetch('/api/laser/on', {method:'POST'});
        showStatus('Laser ON');
    }
    async function laserOff(){
        await fetch('/api/laser/off', {method:'POST'});
        showStatus('Laser OFF');
    }
    async function testPC(){
        showStatus('Testing PC connection...');
        try{
            let r = await fetch('/api/test-pc');
            if(r.ok) showStatus('✅ PC reachable', true);
            else showStatus('❌ PC unreachable', false);
        }catch(e){ showStatus('❌ Connection error', false); }
    }
    
    async function triggerCapture(){
        showStatus('Starting capture...');
        let r = await fetch('/api/capture', {method:'POST'});
        if(r.status === 202){
            showStatus('✅ Capture started! Processing...');
            pollCaptureStatus();
        } else {
            showStatus('❌ Failed to start capture', false);
        }
    }
    
    async function pollCaptureStatus(){
        let attempts = 0;
        let interval = setInterval(async () => {
            try {
                let res = await fetch(pcServer + '/api/capture_status');
                let data = await res.json();
                if(!data.is_processing){
                    clearInterval(interval);
                    if(data.last_result){
                        showStatus(`✅ Capture complete! Particles: ${data.last_result.particle_count}`, true);
                        loadStats();
                    } else if(data.error){
                        showStatus(`❌ Error: ${data.error}`, false);
                    }
                } else {
                    attempts++;
                    if(attempts > 60){ // 5 minutes max
                        clearInterval(interval);
                        showStatus('⚠️ Capture taking too long, check PC dashboard', false);
                    }
                }
            } catch(e){
                clearInterval(interval);
                showStatus('❌ Lost connection to PC', false);
            }
        }, 5000);
    }
    
    async function saveDefaults(){
        let a = document.getElementById('pumpASlider').value;
        let b = document.getElementById('pumpBSlider').value;
        await fetch('/api/settings', {
            method:'POST',
            headers:{'Content-Type':'application/x-www-form-urlencoded'},
            body:'pumpA='+a+'&pumpB='+b
        });
        showStatus('Defaults saved', true);
    }
    async function loadStats(){
        try{
            let r = await fetch(pcServer+'/api/stats');
            let d = await r.json();
            document.getElementById('totalSamples').innerText = d.total_samples||0;
            document.getElementById('highRiskCount').innerText = d.high_risk_count||0;
            document.getElementById('avgParticles').innerText = d.avg_particles||0;
        }catch(e){ console.log('Stats fetch error'); }
    }
    loadStats();
    setInterval(loadStats, 30000);
</script>
</body>
</html>
)rawliteral";
    server.send(200, "text/html", html);
  });

  server.begin();
  Serial.println("HTTP server started");
}

void loop() {
  server.handleClient();
  if (millis() - lastWiFiCheck > 30000) {
    lastWiFiCheck = millis();
    if (WiFi.status() != WL_CONNECTED) WiFi.reconnect();
  }
  delay(10);
}