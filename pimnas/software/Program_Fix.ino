#define BUTTON_PIN 5 // GPIO21 pin connected to button
#define OUTPUT_PIN 12 // D2 pin

// Variables will change:
int lastState = HIGH; // the previous state from the input pin
int currentState;     // the current reading from the input pin
unsigned long pressedTime = 0; // the time when button is pressed
unsigned long releasedTime = 0; // the time when button is released
unsigned long lastDebounceTime = 0; // the last time the output pin was toggled
unsigned long debounceDelay = 50; // the debounce time; increase if the output flickers

void setup() {
  Serial.begin(115200);
  // initialize the pushbutton pin as a pull-up input
  pinMode(BUTTON_PIN, INPUT_PULLUP);
  
  // initialize D2 pin as an output and set it to HIGH
  pinMode(OUTPUT_PIN, OUTPUT);
  digitalWrite(OUTPUT_PIN, HIGH);
}

void loop() {
  // read the state of the switch/button:
  int reading = digitalRead(BUTTON_PIN);

  // check if the button state has changed
  if (reading != lastState) {
    // reset the debouncing timer
    lastDebounceTime = millis();
  }

  if ((millis() - lastDebounceTime) > debounceDelay) {
    // if the button state has changed:
    if (reading != currentState) {
      currentState = reading;

      // only toggle the LED if the new button state is LOW
      if (currentState == LOW) {
        pressedTime = millis();
      } else if (currentState == HIGH) {
        releasedTime = millis();
        unsigned long pressDuration = releasedTime - pressedTime;

        if (pressDuration < 1000) {
          Serial.println("SHORT_PRESS");
        } else if (pressDuration > 2000) {
          Serial.println("LONG_PRESS");
        }
      }
    }
  }

  // save the reading. Next time through the loop, it'll be the lastState:
  lastState = reading;
}
