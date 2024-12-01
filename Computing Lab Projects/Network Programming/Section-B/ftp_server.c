#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <unistd.h>
#include <dirent.h>
#include <pthread.h>

#define PORT 8080
#define BUFFER_SIZE 1024

// Add a function to send a confirmation message to the client
void send_confirmation(int client_socket, const char *message) {
    write(client_socket, message, strlen(message));
}

void handle_put(int client_socket, char *filename) {
    printf("Handling put command for file: %s\n", filename);
    char buffer[BUFFER_SIZE];
    
    // Read the file size from the client
    read(client_socket, buffer, BUFFER_SIZE);
    long filesize = atol(buffer);
    
    // Send an acknowledgment to the client
    write(client_socket, "ACK", 3);
    
    FILE *fp = fopen(filename, "wb");
    if (fp == NULL) {
        perror("Failed to open file");
        write(client_socket, "Error: Failed to open file for writing.", 40);
        return;
    }
    
    int bytesReceived;
    long totalBytesReceived = 0;
    while (totalBytesReceived < filesize) {
        bytesReceived = read(client_socket, buffer, BUFFER_SIZE);
        if (bytesReceived <= 0) {
            break;  // End of transmission or error
        }
        fwrite(buffer, 1, bytesReceived, fp);
        totalBytesReceived += bytesReceived;
    }
    fclose(fp);

    // Send the "EOM" confirmation to the client
    send_confirmation(client_socket, "EOM");
}

void handle_close(int client_socket) {
    send_confirmation(client_socket, "Closing connection.");
    close(client_socket);
    printf("Connection closed.\n");
}

void handle_get(int client_socket, char *filename) {
    printf("Handling get command for file: %s\n", filename);
    char buffer[BUFFER_SIZE];
    FILE *fp = fopen(filename, "rb");
    if (fp == NULL) {
        perror("Failed to open file");
        write(client_socket, "Error: Failed to open file for reading.", 40);
        return;
    }

    // Send the file size to the client
    fseek(fp, 0, SEEK_END);
    long filesize = ftell(fp);
    fseek(fp, 0, SEEK_SET);
    sprintf(buffer, "%ld", filesize);
    write(client_socket, buffer, strlen(buffer));

    // Wait for an acknowledgment from the client
    read(client_socket, buffer, BUFFER_SIZE);

    int bytesRead;
    while ((bytesRead = fread(buffer, 1, BUFFER_SIZE, fp)) > 0) {
        write(client_socket, buffer, bytesRead);
    }
    fclose(fp);

    // Wait for a confirmation from the client
    read(client_socket, buffer, BUFFER_SIZE);

    // Send a confirmation message to the client
    send_confirmation(client_socket, "File transfer complete.");
}


void handle_ls(int client_socket, char *args) {
    printf("Handling ls command with args: %s\n", args ? args : "None"); 
    DIR *d;
    struct dirent *dir;
    d = opendir(".");
    char response[BUFFER_SIZE * 10] = {0};  // Large buffer to accumulate the response
    if (d) {
        while ((dir = readdir(d)) != NULL) {
            if (strcmp(dir->d_name, ".") != 0 && strcmp(dir->d_name, "..") != 0) {  // Exclude . and ..
                strcat(response, dir->d_name);
                strcat(response, "|");  // Delimiter
            }
        }
        closedir(d);
    } else {
        strcat(response, "Error reading directory.");
    }
    strcat(response, "EOM\n");  // End of message
    write(client_socket, response, strlen(response));
}

void handle_cd(int client_socket, char *path) {
    if (chdir(path) == 0) {
        send_confirmation(client_socket, "Changed directory successfully. EOM");
    } else {
        perror("Failed to change directory");
        send_confirmation(client_socket, "Failed to change directory. EOM");
    }
}

void *handle_client(void *client_sock) {
    int client_socket = *((int *)client_sock);
    char buffer[BUFFER_SIZE];
    while (1) {
        memset(buffer, 0, BUFFER_SIZE);
        int bytesRead = read(client_socket, buffer, BUFFER_SIZE);
        if (bytesRead <= 0) {
            printf("Client disconnected or error occurred.\n");
            break;
        }
        buffer[bytesRead] = '\0';  // Properly null-terminate the buffer

        if (strncmp(buffer, "put ", 4) == 0) {
            handle_put(client_socket, buffer + 4);
            send_confirmation(client_socket, "Operation completed.");
        } else if (strncmp(buffer, "get ", 4) == 0) {
            handle_get(client_socket, buffer + 4);
            send_confirmation(client_socket, "Operation completed.");
        } else if (strncmp(buffer, "ls", 2) == 0) {
            char *args = NULL;
            if (buffer[2] == ' ') {
                args = buffer + 3;
            }
            handle_ls(client_socket, args);
            send_confirmation(client_socket, "Operation completed.");
        } else if (strncmp(buffer, "cd ", 3) == 0) {  // Corrected from "cd " to 3
            if (strlen(buffer) > 3) {  // Check if there's an argument after "cd "
                handle_cd(client_socket, buffer + 3);
            } else {
                write(client_socket, "Directory name required.", 24);
            }
        } 
        else if(strncmp(buffer, "close", 5) == 0){
            handle_close(client_socket);
            break;
        } else {
            // For any unrecognized command
            send_confirmation(client_socket, "Unrecognized command.");
        }
    }
    close(client_socket);
    free(client_sock);
    return NULL;
}

int main() {
    int server_socket, *new_sock;
    struct sockaddr_in server_addr, client_addr;
    socklen_t addr_len;
    pthread_t client_thread;

    //server_socket = socket(AF_INET, SOCK_STREAM, 0);
    if ((server_socket = socket(AF_INET, SOCK_STREAM, 0)) == -1) {
        perror("Could not create socket");
        exit(EXIT_FAILURE);
    }

    memset(&server_addr, 0, sizeof(server_addr));

    server_addr.sin_family = AF_INET;
    server_addr.sin_port = htons(PORT);
    server_addr.sin_addr.s_addr = INADDR_ANY;

    int opt = 1;
    if (setsockopt(server_socket, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt)) == -1) {
        perror("setsockopt");
        exit(EXIT_FAILURE);
    }

    if (bind(server_socket, (struct sockaddr *)&server_addr, sizeof(server_addr)) == -1) {
        perror("Bind failed");
        exit(EXIT_FAILURE);
    }

    if (listen(server_socket, 5) == -1) {
        perror("Listen failed");
        exit(EXIT_FAILURE);
    }

    printf("FTP Server listening on port %d...\n", PORT);
    while (1) {
        new_sock = malloc(sizeof(int));
        *new_sock = accept(server_socket, (struct sockaddr *)&client_addr, &addr_len);
        printf("Client connected.\n");

        if (pthread_create(&client_thread, NULL, handle_client, (void *)new_sock) < 0) {
            perror("Could not create thread");
            return 1;
        }
        pthread_detach(client_thread);
    }
    close(server_socket);
    return 0;
}
