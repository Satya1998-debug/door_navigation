# Time Sync Setup (PC ↔ Jetson ↔ Go1)

Keep the whole robot stack on the same clock: the Jetson (time master for the
robot LAN) syncs from a local PC on boot, and the Go1 boards then sync from the
Jetson.

## Topology and roles

| Device | IP on robot LAN | Role | Has RTC? |
|---|---|---|---|
| Local PC | `192.168.123.148` | Bootstrap time source (has internet + real time) | yes |
| Jetson (Orin) | `192.168.123.147` | **Chrony client** to PC 148 **and chrony server** for the Go1 | **no** — boots at 1970 |
| Go1 head Nano | `192.168.123.15` | Chrony/ntpdate client of the Jetson | no |
| Go1 other boards | `192.168.123.13`, `.14`, `.161` | Same | no |

The Jetson is the pivot: it's both a client (of PC 148) and a server (to the Go1).
Because it has no RTC, its clock is wrong on every fresh power-on until PC 148
seeds it.

---

## ⚠️ Critical rule for the Jetson

**Never install `systemd-timesyncd` on the Jetson.** The `chrony` and
`systemd-timesyncd` packages are declared `Conflicts:` in Debian/Ubuntu, so
`apt install systemd-timesyncd` will *silently* run `apt remove chrony` in the
same transaction. The Jetson then loses its ability to serve time to the Go1
even though `systemctl status chrony` may still look OK (a stale LSB stub can
report "active (exited)").

The correct Jetson setup is `chrony` doing **both** client and server duties.
`systemd-timesyncd` is only used on client-only machines like the Go1 boards.

---

## 1. One-time Jetson setup (or recovery if chrony was removed)

Run this whole block on the Jetson. It disables timesyncd, installs chrony
cleanly, holds the package so nothing can accidentally remove it again, and
verifies the result.

```bash
set -e

# 1) Kill systemd-timesyncd so it never fights chrony
sudo systemctl disable --now systemd-timesyncd || true
sudo systemctl mask systemd-timesyncd

# 2) Remove any stale LSB/service leftovers and install a clean chrony
sudo rm -f /etc/systemd/system/chronyd.service   # symlink cleanup only, safe
sudo systemctl daemon-reload
sudo apt update
sudo apt install -y chrony

# 3) Lock chrony against accidental removal (this is what saves you next time
#    a script or a colleague runs `apt install systemd-timesyncd`)
sudo apt-mark hold chrony
sudo apt-mark manual chrony

# 4) Start + enable
sudo systemctl enable --now chrony
sudo systemctl restart chrony

# 5) Verify
dpkg -l chrony | tail -n1                  # must start with 'ii'
systemctl is-enabled chrony                # enabled
systemctl is-active  chrony                # active
systemctl is-active  systemd-timesyncd     # inactive
sudo ss -ulnp | grep ':123'                # chronyd bound to 0.0.0.0:123
chronyc tracking | grep -E 'Reference ID|Stratum|Leap status'
apt-mark showhold | grep chrony            # confirms hold
```

### Jetson `/etc/chrony/chrony.conf` (client + server)

Edit `/etc/chrony/chrony.conf` and make sure it contains the block below.
The important lines are the `server 192.168.123.148` (so the Jetson pulls
from the PC over the robot LAN), `allow` (so the Go1 boards can pull from
the Jetson), `local stratum 10` (so the Jetson keeps serving even after
the PC is unplugged), and `makestep 1.0 -1` (so chrony can step the clock
from 1970 to now on any boot, not just the first few measurements).

```conf
# Preferred local source: the PC on the robot LAN (used to seed time on boot)
server 192.168.123.148 iburst prefer minpoll 4 maxpoll 6

# Public fallbacks (only reachable when the PC is sharing internet)
pool ntp.ubuntu.com        iburst maxsources 4
pool 0.ubuntu.pool.ntp.org iburst maxsources 1
pool 1.ubuntu.pool.ntp.org iburst maxsources 1
pool 2.ubuntu.pool.ntp.org iburst maxsources 2

keyfile   /etc/chrony/chrony.keys
driftfile /var/lib/chrony/chrony.drift
logdir    /var/log/chrony
maxupdateskew 100.0
rtcsync

# Allow chrony to STEP (not just slew) even huge offsets like 1970 → now.
# The default `makestep 1 3` only allows stepping in the first 3 measurements,
# which is fragile when the network comes up slowly.
makestep 1.0 -1

# Serve time to the Go1 subnet
allow 192.168.123.0/24

# Keep serving even when the Jetson has no upstream source
local stratum 10
```

Apply the config:

```bash
sudo systemctl restart chrony
chronyc sources -v
chronyc tracking
```

---

## 2. One-time PC (`192.168.123.148`) setup

The PC needs to be an NTP server on the robot LAN. Same package, simpler config.

```bash
sudo apt install -y chrony
sudo apt-mark hold chrony
```

Ensure `/etc/chrony/chrony.conf` on the PC contains:

```conf
# Whatever upstream pool the PC normally uses (keep the distro defaults)
pool pool.ntp.org iburst

# Serve time to the robot LAN (Jetson + Go1)
allow 192.168.123.0/24
local stratum 8
```

```bash
sudo systemctl enable --now chrony
sudo systemctl restart chrony
sudo ss -ulnp | grep ':123'      # confirm chronyd is listening
chronyc tracking                 # PC must itself be synchronised
```

If the PC is also sharing internet to the Jetson (USB/Ethernet tether), no
extra work is needed for NTP itself — the `allow` line is enough.

---

## 3. One-time Go1 setup (per onboard board)

Each Go1 board can either do a one-shot `ntpdate` at session start (Section 4)
or continuously auto-sync via `systemd-timesyncd`. On the Go1 boards
`systemd-timesyncd` is fine (no chrony conflict there, and they are client-only).

On each Go1 board (`ssh unitree@192.168.123.15`, `.13`, `.14`, `.161`):

```bash
sudo mkdir -p /etc/systemd/timesyncd.conf.d
sudo tee /etc/systemd/timesyncd.conf.d/jetson.conf >/dev/null <<'EOF'
[Time]
NTP=192.168.123.147
FallbackNTP=
EOF

sudo systemctl unmask systemd-timesyncd
sudo systemctl enable --now systemd-timesyncd
sudo systemctl restart systemd-timesyncd
timedatectl set-ntp true
timedatectl status
```

Look for `NTP service: active` and `System clock synchronized: yes`.

---

## 4. Per-boot ritual (what to do every session)

This is the actual daily flow, in order:

1. **Power on the Jetson.** It boots at ~`1970-01-01`.
2. **Plug PC 148 into the Jetson (Ethernet) and share internet.**
   The Jetson's `eth0` should get `192.168.123.147`, PC is `192.168.123.148`.
3. **On the Jetson, kick chrony so it re-resolves and re-syncs:**

   ```bash
   sudo systemctl restart chrony
   chronyc sources -v          # 192.168.123.148 should be reachable (^*)
   chronyc tracking            # Leap status must become 'Normal'
   date                        # sanity check
   ```

   Because `makestep 1.0 -1` is in the config, chrony will step the clock
   from 1970 to now automatically once it has a source.

   *If you're impatient*, force a step:

   ```bash
   sudo chronyc -a makestep
   ```

5. **Unplug PC 148, plug the Go1 in.**
   The Jetson keeps serving time thanks to `local stratum 10`.

6. **On each Go1 board you care about:**

   ```bash
   ssh unitree@192.168.123.15
   sudo ntpdate 192.168.123.147          # one-shot
   # or, if Section 3 was done, timesyncd will handle it automatically
   timedatectl status
   ```

7. **On the Jetson, confirm the Go1 is talking to it:**

   ```bash
   sudo chronyc clients        # Go1 IPs should appear here
   ```

---

## 5. Verification checklist

Everything is healthy when **all** of the following are true.

On the Jetson:

```bash
dpkg -l chrony | tail -n1              # starts with 'ii'
systemctl is-active chrony             # active
systemctl is-active systemd-timesyncd  # inactive
sudo ss -ulnp | grep ':123'            # chronyd on 0.0.0.0:123
chronyc tracking | grep 'Leap status'  # Normal
```

On a Go1 board:

```bash
sudo ntpdate -q 192.168.123.147        # returns an offset (not "no server suitable")
timedatectl status                     # NTP service: active, System clock synchronized: yes
```

Back on the Jetson:

```bash
sudo chronyc clients                   # Go1 IPs listed with non-zero NTP count
```

Optional end-to-end check (ROS TF latency should be small):

```bash
rosrun tf tf_monitor
# Net delay avg < 10 ms across nodes on Jetson and Go1
```

---

## 6. Troubleshooting

### "chronyd: command not found" or `dpkg -l chrony` starts with `rc`

The package was removed. This almost always means someone (or a script, or an
agent following outdated notes) ran `apt install systemd-timesyncd` — which
implicitly removes chrony because of the package conflict. Confirm:

```bash
grep -E 'chrony|timesyncd' /var/log/apt/history.log | tail -n 20
```

You'll see paired `Install: systemd-timesyncd` / `Remove: chrony` lines. Fix
by re-running the block in Section 1 (it also `apt-mark hold`s chrony to stop
this recurring).

### `systemctl status chrony` says "active (exited)" but `chronyc` is missing

The package is gone but a leftover `/etc/init.d/chrony` LSB script is still
being invoked. `Loaded: loaded (/etc/init.d/chrony; generated)` in the status
output is the tell. Re-run Section 1.

### `chrony.service is masked`

Something explicitly masked the unit. Undo and re-enable:

```bash
sudo systemctl unmask chrony
sudo systemctl daemon-reload
sudo systemctl enable --now chrony
```

### Jetson clock is right but Go1 gets "no server suitable for synchronization"

Chrony is running but not reachable from the Go1 subnet. Check:

```bash
sudo ss -ulnp | grep ':123'                    # must show 0.0.0.0:123, not just 127.0.0.1
grep -E '^(allow|bindaddress|local)' /etc/chrony/chrony.conf
```

Must contain `allow 192.168.123.0/24` and no `bindaddress` line (or a
`bindaddress` that covers `eth0`).

Also from the Go1:

```bash
nc -u -z -v 192.168.123.147 123
```

If this fails, it's an IP/route/firewall issue — not chrony.

### Nothing at all in `chronyc clients`

Either no Go1 board has queried yet, or your Jetson chrony is bound only to
`127.0.0.1`. Run `sudo chronyc clients` (needs sudo). If the Go1 boards have
made at least one query, they'll appear here.

---

## Notes / gotchas

- **RTC.** The Jetson Orin dev carrier ships without a coin-cell backup for the
  RTC. Adding one would eliminate the "boots at 1970" problem and make this
  whole ritual unnecessary. Until then, PC 148 is the seed source.
- **Automatic Go1 sync.** After Section 3, `ntpdate` in Section 4 is only a
  belt-and-suspenders one-shot; `systemd-timesyncd` on the Go1 boards keeps them
  in sub-millisecond agreement with the Jetson continuously.
- **Never** repeat: **do not install `systemd-timesyncd` on the Jetson.** If a
  future note, script, or AI suggests it, ignore it.
