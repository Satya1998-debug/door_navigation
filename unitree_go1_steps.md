### ----------------------------------------------------------------------------------------
# Unitree Go1 Robot - Basic info and Steps to start
### Controllers
- Remote Control (RC):
    - power button: ON: one short press + one long press(3 seconds) >>> drip sound
    - power button: OFF: one short press + one long press(3 seconds) >>> 3 drip sounds
    - it has inbuilt bluetooth module for data transmission
    - joysticks
- Label Controller (LC):

### RC Callibration (if needed)
- turn ON the RC
- F1 + F3 (press and release together) >>> enter callibration mode (2 drips, 1 drip per second)
- move joysticks to full rudder positions (up, down, left, right) several times (till the drip sound stops)
- callibration complete
NOTE: Donot touch the joysticks before callibration

### Battery check
- charge the battery fully (using the provided adapter) for 1.5 hours
- charge the Remote Control for 1.5 hours (C-type cable to the battery adapter) (if needed)
- charge the Label Controller via USB-C cable to a power source for 1.5 hours (if needed)
- 4 LEDs indicate battery full
- turn ON/OFF: one short press + one long press(3 seconds) on the batery power button

### Robot start/power ON
- insert battery (check a locking sound mechanism)
- keep the robot flat on the ground (all legs folded)
- Power ON the robot:
    - search power button
    - 1 short + 1 long press (Awhirring sound)
    - robot will stand up (automatically)
- Now in this case, the robot can be remote controlled via the RC or the LC
- after 1 minute of powering on
    - white light will flash near its head (indicates robot is ready to be controlled by label controller)

### Remote Control (RC) operation
- L2+B >>> robot sits down (low power mode, damping state)
- L2+A >>> robot locks (the robot can switch between standing and siting states via multiple operations)
- Start >>> unlocked and unattitude mode
- Start again >>> walk mode (controlled by pushing rockers/joysticks, if handheld is not pressed, the robot will stand still)
- Start twice >>> fast run mode (it will step all the time, without pushing handheld sticks)

### States of the robot
- Static Standing State
- Proning State (Damping/Undamping State)
- Sports Mode (Walking/Running State)

### Robot shut down/power OFF
- make sure the robot is in a static standing state
- Press and Hold L2 + A(3 times) >>> robot will complete squat, stand up & lie down
- Press and Hold L2 + B(2 times) >>> robot will complete prone (damping), prone (undamping)
- Press Power button (once) then long press (3 seconds) >>> Battery is off


### ----------------------------------------------------------------------------------------
### Modes of operation
- High-level mode:
    - mainly for walking & running
    - 2 modes: nomral mode, motion mode (they have separate IPs to connect)
- Low-level mode
NOTE: Only one mode can be active at a time, and cannot be switched during operation.

### High-level mode operation
- run after connecting LAN cable to the robot
```bash
sudo su
source 
```
