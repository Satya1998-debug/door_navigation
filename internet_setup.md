# Internet Setup

Use the local IAS Linux PC to temporarily provide internet to the external Jetson.

- **Local PC:** `192.168.123.148`
- **External Jetson:** uses the shared connection first, then connects back to the Go1 network

---

## 1. Share Internet from the local PC

Connect the external Jetson to the local PC with Ethernet.

Internet sharing on the PC is already configured, so wait a short time for:

- internet access to become available on the Jetson
- the Jetson clock to sync

or else

- run the gateway service:

```bash
sudo systemctl restart set-gateway.service # change the IP of local PC as per the local IP (for ias local PC is 148)
```

- this will set the gateway available to the Jetson (either the local PC or subnet of Go1)
- the service runs this script: /usr/local/bin/set-gateway
- The script should contain the local PC's IP (in the gateway candidates)

---



## 2. Connect to Uni Stuttgart Wi-Fi

After the Jetson clock is synced, connect the Jetson to:

```text
uni-stuttgart open
```

---



## 3. Reconnect to the Go1 Network

Disconnect the Jetson from the IAS PC.

Connect the Jetson normally to the Go1 Ethernet port. The Jetson should now be:

- on the local Go1 network over Ethernet
- still connected to Uni Stuttgart internet over Wi-Fi

---
