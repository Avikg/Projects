# Network Programming Assignment (CS69011)

## Overview

This repository contains the solution to **Assignment: Network Programming** as part of the **CS69011 Computing Lab** course. The assignment involves implementing:
- **Section-A**: A simplified version of the `ping` command using TCP.
- **Section-B**: A basic File Transfer Protocol (FTP) system with client-server architecture.

Both sections include server and client implementations using **C programming language**.

---

## Repository Structure

```
.
├── Section-A/
│   ├── ping_server.c            # Server implementation for Ping command
│   ├── ping_client.c            # Client implementation for Ping command
│   ├── ping.log                 # Log file for ping requests and RTTs
│   ├── README.md                # Documentation for Section-A
├── Section-B/
│   ├── ftp_server.c             # Server implementation for FTP
│   ├── ftp_client.c             # Client implementation for FTP
│   ├── README.md                # Documentation for Section-B
├── Assignment9.zip              # Submission package
└── README.md                    # This file
```

---

## Requirements

### Software
- **Linux OS**: Recommended for development and testing.
- **GCC Compiler**: To compile the C programs.

---

## Compilation and Execution

### Section-A: Ping Command

#### Compilation
```bash
gcc ping_server.c -o ping_server
gcc ping_client.c -o ping_client
```

#### Execution
1. Start the server:
   ```bash
   ./ping_server <port_number>
   ```
2. Start the client:
   ```bash
   ./ping_client <server_ip> <port_number> <number_of_pings> <interval_in_seconds>
   ```

#### Features
- Measures the **Round Trip Time (RTT)** for each ping request.
- Supports handling **concurrent clients**.
- Logs requests in `ping.log` with source IP, port, and RTT.

---

### Section-B: FTP System

#### Compilation
```bash
gcc ftp_server.c -o ftp_server
gcc ftp_client.c -o ftp_client
```

#### Execution
1. Start the FTP server:
   ```bash
   ./ftp_server <port_number>
   ```
2. Start the FTP client:
   ```bash
   ./ftp_client <server_ip> <port_number>
   ```

#### Features
- Commands supported:
  1. `put <file_name>`: Upload a file to the server.
  2. `get <file_name>`: Download a file from the server.
  3. `close`: Close the connection.
  4. `cd <directory_name>`: Change directory on the server.
  5. `ls`: List contents of the current directory.
- **Concurrent connections**: Supports multiple clients interacting with the server simultaneously.
- User-friendly command-line prompt: `ftp_client>` for user interaction.

---

## Logging and Error Handling

### Section-A
- Logs each request in `ping.log` with:
  - Source IP.
  - Source port.
  - RTT (in milliseconds).

### Section-B
- Error messages for failed file operations, incorrect commands, or connection issues.
- Appropriate success or failure messages displayed for each FTP operation.

---

## References

- [TCP Client-Server Programming in C](https://nikhilroxtomar.medium.com/tcp-client-server-implementation-in-c-idiot-developer-52509a6c1f59)

---

## Author

**Avik Pramanick**  
**Roll No:** 23CS60R78  
**Course:** CS69011 Computing Lab  
