# Time Sync Setup


Keep the Go1 NX clock in sync with the external Jetson (time master) over the local robot network.

- **Time Master (Jetson):** `192.168.123.147`
- **Client (Go1 NX):** `192.168.123.15`

---

## 1. One-Time Setup on the Jetson (Time Master)

Install Chrony and configure it to serve time to the robot network.

```bash
sudo apt install chrony -y
```

Append to `/etc/chrony/chrony.conf`:

```
allow 192.168.123.0/24
local stratum 10
```

Restart the service (need to be done once after every reboot):

```bash
sudo systemctl restart chrony # this will sync the time in jetson
```
- verify:
```bash
chronyc tracking
chronyc sources -v
```
- the leap status need to be normal


- also do this if jetson itself shared internet from some local PC:
```bash
sudo ntpdate 192.168.123.148  # use local IP pf thta PC to sync time
```
---

## 2. Per-Boot Manual Sync (Quick Fix)

The Go1 NX loses its clock on power-off. Force a sync at the start of each session.

```bash
ssh unitree@192.168.123.15
sudo ntpdate 192.168.123.147
```

Expect output like: `stepped time server… offset 1.7x sec`.

Then launch ROS nodes as usual.

---

### Enable CLOCK synchronization (in Jetson ORIN)

- install the service: (if not installed before)
```bash
sudo apt-get update
sudo apt-get install -y systemd-timesyncd
```

- then enable the service:
```bash
sudo systemctl enable --now systemd-timesyncd
timedatectl set-ntp true
timedatectl status
```

## 3. Permanent Sync (Recommended)

Configure the Go1 NX to continuously micro-adjust its clock using `systemd-timesyncd` (already installed on Ubuntu).

**On the Go1 NX**, edit `/etc/systemd/timesyncd.conf`:

```ini
[Time]
NTP=192.168.123.147
```

Enable and start the service:

```bash
sudo systemctl unmask systemd-timesyncd
sudo systemctl enable systemd-timesyncd
sudo systemctl restart systemd-timesyncd
```

---

## 4. Verify

**On the Go1 NX:**

```bash
timedatectl status
```

Look for:
- `System clock synchronized: yes`
- `NTP service: active`

**On the Jetson**, check TF latency:

```bash
rosrun tf tf_monitor
```

`Net delay avg` should be under **0.01 s (10 ms)**.

---

## Notes

- After step 3, the Go1 auto-syncs to the Jetson within seconds of boot via Ethernet.
- Adjustments are continuous and sub-millisecond — no manual `ntpdate` needed.
