#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <unistd.h>
#include <arpa/inet.h> 

#define SERVER_IP "127.0.0.1"
#define PORT 8080
#define BUFFER_SIZE 1024

void handle_put(int client_socket, char *filename) {
    char buffer[BUFFER_SIZE];
    FILE *fp = fopen(filename, "rb");
    if (fp == NULL) {
        perror("Failed to open file");
        return;
    }
    
    // Get the file size
    fseek(fp, 0, SEEK_END);
    long filesize = ftell(fp);
    fseek(fp, 0, SEEK_SET);
    
    // Send the file size to the server
    sprintf(buffer, "%ld", filesize);
    write(client_socket, buffer, strlen(buffer));
    
    // Wait for an acknowledgment from the server
    read(client_socket, buffer, BUFFER_SIZE);
    
    int bytesRead;
    while ((bytesRead = fread(buffer, 1, BUFFER_SIZE, fp)) > 0) {
        write(client_socket, buffer, bytesRead);
    }
    fclose(fp);

    // Wait for the server's confirmation message with "EOM"
    while (1) {
        memset(buffer, 0, sizeof(buffer));
        read(client_socket, buffer, sizeof(buffer));
        if (strstr(buffer, "EOM")) {
            break;  // Stop reading when EOM is encountered
        }
    }
    printf("Server: %s\n", buffer);
}


void handle_get(int client_socket, char *filename) {
    char buffer[BUFFER_SIZE];
    FILE *fp = fopen(filename, "wb");
    if (fp == NULL) {
        perror("Failed to open file");
        return;
    }

    // Read the file size from the server
    read(client_socket, buffer, BUFFER_SIZE);
    long filesize = atol(buffer);

    // Send an acknowledgment to the server
    write(client_socket, "ACK", 3);

    int bytesReceived;
    long totalBytesReceived = 0;
    while (totalBytesReceived < filesize) {
        bytesReceived = read(client_socket, buffer, BUFFER_SIZE);
        fwrite(buffer, 1, bytesReceived, fp);
        totalBytesReceived += bytesReceived;
    }
    fclose(fp);

    // Send a confirmation to the server
    write(client_socket, "Received file", 13);

    // Read the server's confirmation message
    read(client_socket, buffer, BUFFER_SIZE);
    printf("Server: %s\n", buffer);
}

void handle_cd(int client_socket, char *path) {
    char buffer[BUFFER_SIZE];

    // Send the cd command with the path to the server
    sprintf(buffer, "cd %s", path);
    write(client_socket, buffer, strlen(buffer));

    // Wait for the server's confirmation message with "EOM"
    while (1) {
        memset(buffer, 0, sizeof(buffer));
        read(client_socket, buffer, sizeof(buffer));
        if (strstr(buffer, "EOM")) {
            break;  // Stop reading when EOM is encountered
        }
    }
    printf("Server: %s\n", buffer);
}


void handle_ls(int client_socket) {
    char buffer[BUFFER_SIZE];
    char full_response[BUFFER_SIZE * 10] = {0};  // Large buffer to accumulate the response

    // Send the ls command to the server
    write(client_socket, "ls", 2);

    // Wait for the server's directory listing with "EOM"
    while (1) {
        memset(buffer, 0, sizeof(buffer));
        read(client_socket, buffer, sizeof(buffer));
        strcat(full_response, buffer);
        if (strstr(buffer, "EOM")) {
            break;  // Stop reading when EOM is encountered
        }
    }

    // Remove the "EOM" from the response and print the directory listing
    char *eom_pos = strstr(full_response, "EOM");
    if (eom_pos) {
        *eom_pos = '\0';  // Terminate the string at the EOM signal
    }
    printf("%s\n", full_response);
}



int main() {
    int client_socket;
    struct sockaddr_in server_addr;
    char buffer[BUFFER_SIZE];
    char command[5], argument[BUFFER_SIZE];

    client_socket = socket(AF_INET, SOCK_STREAM, 0);
    server_addr.sin_family = AF_INET;
    server_addr.sin_port = htons(PORT);
    server_addr.sin_addr.s_addr = inet_addr(SERVER_IP);

    connect(client_socket, (struct sockaddr *)&server_addr, sizeof(server_addr));
    printf("Connected to FTP server.\n");

    while (1) {
        printf("ftp_client> ");
        fgets(buffer, BUFFER_SIZE, stdin);
        buffer[strcspn(buffer, "\n")] = 0;  // Remove newline character

        sscanf(buffer, "%s %s", command, argument);
    
        if (strcmp(command, "put") == 0) {
            write(client_socket, buffer, strlen(buffer));
            handle_put(client_socket, argument);
            // Read the server's confirmation message
            read(client_socket, buffer, BUFFER_SIZE);
            printf("Server: %s\n", buffer);
        } else if (strcmp(command, "get") == 0) {
            write(client_socket, buffer, strlen(buffer));
            handle_get(client_socket, argument);
        } else if(strcmp(command, "close") == 0){
            write(client_socket, "close", 5);
            read(client_socket, buffer, BUFFER_SIZE);
            printf("Server: %s\n", buffer);
            break;
        } else if (strcmp(command, "cd") == 0) {
            handle_cd(client_socket, argument);  // Call the handle_cd function
        } 
        
        else if (strcmp(command, "ls") == 0) {
            handle_ls(client_socket);  // Call the handle_ls function
        }
        else {
            printf("Unknown command.\n");
        }
    }
    close(client_socket);
    return 0;
}
