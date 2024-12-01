#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <arpa/inet.h>
#include <sys/socket.h>
#include <pthread.h>

#define BUFFER_SIZE 2048

// Function to handle incoming messages from the server
void *receive_message(void *socket) {
    int sockfd = *((int *)socket);
    char message[BUFFER_SIZE];
    
    while (1) {
        int length = recv(sockfd, message, BUFFER_SIZE - 1, 0);
        if (length > 0) {
            message[length] = '\0';
            // Print server's message directly without the "user>" prefix
            printf("%s\n", message);
        } else {
            perror("Disconnected from server.");
            exit(EXIT_FAILURE);
        }
    }
    return NULL;
}

int main() {
    struct sockaddr_in server_addr;
    int sockfd;
    char buffer[BUFFER_SIZE];
    char server_ip[20];
    int server_port;

    // User inputs server IP and port
    printf("Enter SERVER IP: ");
    fgets(server_ip, 20, stdin);
    server_ip[strcspn(server_ip, "\n")] = 0;

    printf("Enter SERVER PORT: ");
    scanf("%d", &server_port);
    getchar();

    // Socket creation and connection
    sockfd = socket(AF_INET, SOCK_STREAM, 0);
    if (sockfd < 0) {
        perror("Could not create socket");
        return 1;
    }
    server_addr.sin_family = AF_INET;
    server_addr.sin_port = htons(server_port);
    server_addr.sin_addr.s_addr = inet_addr(server_ip);
    if (connect(sockfd, (struct sockaddr *)&server_addr, sizeof(server_addr)) < 0) {
        perror("Connect failed");
        return 1;
    }
    printf("Connected to the server.\n");

    // Create a thread to receive messages from the server
    pthread_t recv_thread;
    if (pthread_create(&recv_thread, NULL, receive_message, (void*)&sockfd) != 0) {
        perror("Could not create thread for receiving messages.");
        return 1;
    }

    // Initial user name prompt without "user>" prefix
    printf("Enter your name: ");
    fgets(buffer, BUFFER_SIZE, stdin);
    buffer[strcspn(buffer, "\n")] = 0;
    send(sockfd, buffer, strlen(buffer), 0);

    // Loop for sending messages with "user>" prompt
    while (1) {
        printf("user> ");
        fgets(buffer, BUFFER_SIZE, stdin);
        buffer[strcspn(buffer, "\n")] = 0;
        
        send(sockfd, buffer, strlen(buffer), 0);
        
        if (strcmp(buffer, "/logout") == 0) {
            break; // Exit loop on logout command
        }
    }

    // Cleanup
    pthread_join(recv_thread, NULL); // Wait for receive thread to finish
    close(sockfd); // Close socket
    printf("Disconnected from server.\n");
    return 0;
}
