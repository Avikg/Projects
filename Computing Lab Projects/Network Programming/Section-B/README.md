FTP SERVER AND CLIENT IMPLEMENTATION:
This repository contains the source code for a simple FTP server and client implemented in C.

PREREQUISITES:
1. GCC Compiler
2. Linux/Unix environment (for the pthread library)

COMPILATION:
FTP SERVER:
To compile the FTP server, navigate to the directory containing the ftp_server.c file and run:
gcc ftp_server.c -o ftp_server -lpthread

This will produce an executable named ftp_server.

FTP CLIENT:
To compile the FTP client, navigate to the directory containing the ftp_client.c file and run:
gcc ftp_client.c -o ftp_client

This will produce an executable named ftp_client.

RUNNING THE PROGRAMS:
FTP SERVER:
To run the FTP server, use:
./ftp_server 

Replace <PORT> with the desired port number (e.g., 3456). Default PORT in the Code is 8080.

EXAMPLE:
./ftp_server 

FTP CLIENT:
To run the FTP client, use:
./ftp_client 

Replace <SERVER_IP> with the IP address of the server and <PORT> with the port number the server is listening on. Default SERVER_IP 127.0.0.1 and Default PORT 8080.

EXAMPLE:
./ftp_client

SUPPORTED FTP COMMANDS:
The client and server implementation supports the following commands:
1. put <filename>: Upload a file to the server.
2. get <filename>: Download a file from the server.
3. cd <directory>: Change the current directory on the server.
4. ls: List the contents of the current directory on the server.
5. close: Disconnect from the server.