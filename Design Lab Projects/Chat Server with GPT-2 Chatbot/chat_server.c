#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <arpa/inet.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <netinet/in.h>
#include <signal.h>
#include <pthread.h>
#include <uuid/uuid.h>
#include <ctype.h>
#include <dirent.h>

#define MAX_CLIENTS 10
#define BUFFER_SIZE 1024
#define PORT 8080
#define MAX_FAQS 1000

typedef struct {
    struct sockaddr_in address;
    int sockfd;
    char uid[37];
    char name[32];
    int chatbot_enabled;  // Chatbot feature status
    int chatbot_v2_enabled;  // Track if the GPT-2 chatbot is enabled for the client
} client_t;

typedef struct {
    char question[BUFFER_SIZE];
    char answer[BUFFER_SIZE * 2];
} FAQ;

typedef struct chat_message {
    char sender_uid[37];
    char recipient_uid[37];
    char message[BUFFER_SIZE];
    struct chat_message* next; // For a linked list
} chat_message_t;

typedef struct chat_history {
    chat_message_t* head; // Head of the linked list of messages
} chat_history_t;


client_t* clients[MAX_CLIENTS];

FAQ faqs[MAX_FAQS];
int faq_count = 0;


void load_faqs() {
    FILE *file = fopen("FAQs.txt", "r");
    if (!file) {
        perror("Failed to open FAQs file");
        exit(1);
    }

    char line[BUFFER_SIZE * 3];
    while (fgets(line, sizeof(line), file) && faq_count < MAX_FAQS) {
        char *question = strtok(line, "|||");
        char *rawAnswer = strtok(NULL, "");
        if (question && rawAnswer && rawAnswer[0] == '|' && rawAnswer[1] == '|') {
            // Skipping past the initial "||"
            char *answer = rawAnswer + 2;
            // Trim leading whitespace after "||"
            while (*answer && isspace((unsigned char)*answer)) {
                answer++;
            }
            // Ensure the answer doesn't exceed the buffer size
            strncpy(faqs[faq_count].question, question, BUFFER_SIZE);
            faqs[faq_count].question[BUFFER_SIZE - 1] = '\0';
            strncpy(faqs[faq_count].answer, answer, BUFFER_SIZE * 2);
            faqs[faq_count].answer[BUFFER_SIZE * 2 - 1] = '\0';
            faq_count++;
        }
    }

    fclose(file);
    printf("%d FAQs loaded.\n", faq_count);
}


const char* get_answer(const char* input_question) {
    // Normalize the input question for comparison
    char normalized_question[BUFFER_SIZE] = {0};
    for (int i = 0; i < strlen(input_question) && i < BUFFER_SIZE - 1; i++) {
        normalized_question[i] = tolower(input_question[i]);
    }

    printf("Normalized Query: %s\n", normalized_question);

    for (int i = 0; i < faq_count; i++) {
        // Normalize FAQ question for comparison
        char normalized_faq_question[BUFFER_SIZE] = {0};
        for (int j = 0; j < strlen(faqs[i].question) && j < BUFFER_SIZE - 1; j++) {
            normalized_faq_question[j] = tolower(faqs[i].question[j]);
        }

        // Log for debugging
        printf("Checking against FAQ: %s\n", normalized_faq_question);

        if (strstr(normalized_faq_question, normalized_question) != NULL) {
            return faqs[i].answer; // Found a matching FAQ
        }
    }

    return "System Malfunction, I couldn't understand your query.";
}


void delete_all_chat_history(const char* client_id) {
    DIR *d;
    struct dirent *dir;
    d = opendir("."); // Assume chat history files are in the current directory
    if (d) {
        while ((dir = readdir(d)) != NULL) {
            // Construct the pattern to check if the file name matches the expected chat history file pattern
            char pattern1[1024], pattern2[1024];
            snprintf(pattern1, sizeof(pattern1), "%s_", client_id); // client_id as sender
            snprintf(pattern2, sizeof(pattern2), "_%s_history.txt", client_id); // client_id as recipient
            
            // Check if filename contains client_id in either position
            if (strstr(dir->d_name, pattern1) != NULL || strstr(dir->d_name, pattern2) != NULL) {
                // File related to chat history of client_id found, delete it
                if (remove(dir->d_name) == 0) {
                    printf("Deleted chat history file: %s\n", dir->d_name);
                } else {
                    perror("Error deleting chat history file");
                }
            }
        }
        closedir(d);
    }
}

void store_message(const char* sender_id, const char* recipient_id, const char* message) {
    char filename[100];
    snprintf(filename, sizeof(filename), "%s_%s_history.txt", sender_id, recipient_id);
    
    FILE* file = fopen(filename, "a");
    if (file != NULL) {
        fprintf(file, "%s: %s\n", sender_id, message);
        fclose(file);
    }
}

void get_chat_history_filename(char* filename, const char* sender_id, const char* recipient_id, size_t filename_size) {
    snprintf(filename, filename_size, "history_%s_%s.txt", sender_id, recipient_id);
}

char* get_chat_history(const char* sender_id, const char* recipient_id) {
    // Example file name generation
    char filename[256];
    snprintf(filename, sizeof(filename), "%s_%s_history.txt", sender_id, recipient_id);

    // Open the file
    FILE* file = fopen(filename, "r");
    if (!file) {
        return NULL;
    }

    // Example of dynamically allocating memory for the history
    // Note: In a real application, you would likely need to read the file contents
    // to determine the appropriate size before allocation.
    char* history = malloc(4096); // Example allocation
    if (!history) {
        fclose(file);
        return NULL;
    }

    // Populate history with the file content
    // For simplicity, assuming the history content fits within 4096 bytes
    fread(history, 4096, 1, file);
    fclose(file);

    return history; // Caller is responsible for freeing this memory
}

void append_to_chat_history(const char* sender_id, const char* recipient_id, const char* message) {
    char filename[100];
    get_chat_history_filename(filename, sender_id, recipient_id, sizeof(filename));

    FILE* file = fopen(filename, "a");
    if (file) {
        fprintf(file, "%s -> %s: %s\n", sender_id, recipient_id, message);
        fclose(file);
    }
}

void send_chat_history(int sockfd, const char* sender_id, const char* recipient_id) {
    char filename[100];
    get_chat_history_filename(filename, sender_id, recipient_id, sizeof(filename));

    FILE* file = fopen(filename, "r");
    if (!file) {
        send(sockfd, "No history found.\n", 18, 0);
        return;
    }

    char line[1024];
    while (fgets(line, sizeof(line), file)) {
        send(sockfd, line, strlen(line), 0);
    }
    fclose(file);
}

void delete_chat_history(const char* sender_id, const char* recipient_id) {
    // Attempt to generate and delete the first filename convention
    char filename1[100];
    snprintf(filename1, sizeof(filename1), "%s_%s_history.txt", sender_id, recipient_id);
    if (remove(filename1) == 0) {
        printf("Chat history between %s and %s deleted.\n", sender_id, recipient_id);
        return; // Success, no need to try the second filename
    } else {
        // Attempt failed, try the second filename convention
        char filename2[100];
        snprintf(filename2, sizeof(filename2), "%s_%s_history.txt", recipient_id, sender_id);
        if (remove(filename2) == 0) {
            printf("Chat history between %s and %s deleted.\n", recipient_id, sender_id);
        } else {
            perror("Failed to delete chat history");
        }
    }
}





// Utility function to print the client's IP address
void print_client_addr(struct sockaddr_in addr) {
    printf("%s", inet_ntoa(addr.sin_addr));
}

// Adds client to the array of clients
void add_client(client_t* cl) {
    for (int i = 0; i < MAX_CLIENTS; ++i) {
        if (!clients[i]) {
            clients[i] = cl;
            break;
        }
    }
}

// Removes client from the array by UUID
void remove_client(char* uid) {
    for (int i = 0; i < MAX_CLIENTS; ++i) {
        if (clients[i]) {
            if (strcmp(clients[i]->uid, uid) == 0) {
                clients[i] = NULL;
                break;
            }
        }
    }
}

// Sends message to all clients except the sender
void send_message(char* s, char* uid) {
    for (int i = 0; i < MAX_CLIENTS; ++i) {
        if (clients[i]) {
            if (strcmp(clients[i]->uid, uid) != 0) {
                if (write(clients[i]->sockfd, s, strlen(s)) < 0) {
                    perror("Failed to send message");
                    continue;
                }
            }
        }
    }
}

// The thread function for handling communication with the client
void* handle_client(void* arg) {
    char buffer[BUFFER_SIZE];
    int leave_flag = 0;
    client_t* cli = (client_t*)arg;
    cli->chatbot_enabled = 0;
    cli->chatbot_v2_enabled = 0;

    // First message from client is the name
    int name_len = recv(cli->sockfd, cli->name, sizeof(cli->name), 0);
    if (name_len <= 0) {
        puts("Failed to receive client's name.");
        return NULL;
    }
    cli->name[name_len] = '\0'; // Null-terminate the name string

    sprintf(buffer, "%s has joined with ID: %s\n", cli->name, cli->uid);
    // Send welcome message or perform other initializations...

    //recv(cli->sockfd, cli->name, sizeof(cli->name), 0); // Receive client's name
    sprintf(buffer, "%s has joined with ID: %s\n", cli->name, cli->uid);
    printf("%s", buffer);
    send_message(buffer, cli->uid); // Announce new client to all

    while (!leave_flag) {
        bzero(buffer, BUFFER_SIZE);
        int receive = recv(cli->sockfd, buffer, BUFFER_SIZE, 0);
        if (receive > 0) {
            buffer[receive] = '\0'; // Null-terminate received message

            printf("Received: %s\n", buffer); // Debugging: print received message

            // Strip newline
            size_t ln = strlen(buffer) - 1;
            if (buffer[ln] == '\n') buffer[ln] = '\0';

            // Chatbot activation command
            if (strcmp(buffer, "/chatbot_v2 login") == 0) {
                cli->chatbot_v2_enabled = 1;
                char login_msg[] = "gpt2bot> Hi, I am updated bot, I am able to answer any question be it correct or incorrect\n";
                send(cli->sockfd, login_msg, strlen(login_msg), 0);
                continue;
            }

            // Chatbot deactivation command
            if (strcmp(buffer, "/chatbot_v2 logout") == 0) {
                cli->chatbot_v2_enabled = 0;
                char logout_msg[] = "gpt2bot> Bye! Have a nice day and hope you do not have any complaints about me\n";
                send(cli->sockfd, logout_msg, strlen(logout_msg), 0);
                continue;
            }

            // Process chatbot interaction
            if (cli->chatbot_v2_enabled) {
                // The command should exclude the chatbot activation/deactivation part
                char *actualCmd = NULL;
                if (strncmp(buffer, "/send ", 6) == 0) {
                    actualCmd = buffer + 6; // Skip "/send " to get the actual message
                } else {
                    actualCmd = buffer; // Use the buffer directly if not a "/send " command
                }

                char cmd[1024];
                snprintf(cmd, sizeof(cmd), "python3 gpt2.py \"%s\"", actualCmd);

                FILE *fp = popen(cmd, "r");
                if (fp == NULL) {
                    printf("Failed to run command\n");
                    continue; // Skip further processing for this message
                }

                char pythonOutput[BUFFER_SIZE];
                while (fgets(pythonOutput, BUFFER_SIZE, fp) != NULL) {
                    // Assuming pythonOutput contains the response prefixed with "gpt2bot>"
                    send(cli->sockfd, pythonOutput, strlen(pythonOutput), 0);
                }
                pclose(fp);
                continue; // Ensure no further processing for this message
            }

            
            // Correcting chatbot login command recognition
            if (strcmp(buffer, "/chatbot_login") == 0) {
                cli->chatbot_enabled = 1;
                printf("Chatbot enabled for: %s\n", cli->name); // Debugging
                char login_msg[] = "stupidbot> Hi, I am stupid bot, I am able to answer a limited set of your questions\n";
                send(cli->sockfd, login_msg, strlen(login_msg), 0);
                continue;
            }

            // Chatbot logout command
            if (strcmp(buffer, "/chatbot_logout") == 0) {
                cli->chatbot_enabled = 0;
                printf("Chatbot disabled for: %s\n", cli->name); // Debugging
                char logout_msg[] = "stupidbot> Bye! Have a nice day and do not complain about me\n";
                send(cli->sockfd, logout_msg, strlen(logout_msg), 0);
                continue;
            }

            // If chatbot is enabled, treat all messages as questions (except logout)
            if (cli->chatbot_enabled) {
                // Skip processing if it's a logout command to allow proper session closure
                if (strcmp(buffer, "/logout") == 0) {
                    leave_flag = 1;
                    char farewell_msg[] = "Bye!! Have a nice day\n";
                    send(cli->sockfd, farewell_msg, strlen(farewell_msg), 0);
                } else {
                    // Treat received message as a question and fetch the answer
                    const char* answer = get_answer(buffer);
                    char response[BUFFER_SIZE * 2];
                    snprintf(response, sizeof(response), "stupidbot> %s\n", answer);
                    send(cli->sockfd, response, strlen(response), 0);
                }
                continue;
            }

            // Example handling of /history command
            if (strncmp(buffer, "/history ", 9) == 0) {
                char *recipient_id = buffer + 9;
                // Retrieve and send chat history with recipient_id
                // This requires implementing get_chat_history()
                char *history = get_chat_history(cli->uid, recipient_id);
                if (history) {
                    send(cli->sockfd, history, strlen(history), 0);
                    free(history); // Assuming dynamically allocated
                } else {
                    send(cli->sockfd, "No history found.\n", 18, 0);
                }
                continue;
            }

            // Handling of /history_delete command
            if (strncmp(buffer, "/history_delete ", 16) == 0) {
                char *recipient_id = buffer + 16;
                // Delete chat history with recipient_id
                // This requires implementing delete_chat_history()
                delete_chat_history(cli->uid, recipient_id);
                send(cli->sockfd, "History deleted.\n", 17, 0);
                continue;
            }

            // Handling of /delete_all command
            if (strcmp(buffer, "/delete_all") == 0) {
                // Delete all chat history for the client
                // This requires implementing delete_all_chat_history()
                delete_all_chat_history(cli->uid);
                send(cli->sockfd, "All history deleted.\n", 21, 0);
                continue;
            }


            // Handle logout
            if (strcmp(buffer, "/logout") == 0) {
                leave_flag = 1;
                // Send farewell message to the client
                char farewell_msg[] = "Bye!! Have a nice day";
                send(cli->sockfd, farewell_msg, strlen(farewell_msg), 0);
            }
            // Handle active clients listing
            else if (strcmp(buffer, "/active") == 0) {
                char list[2048] = "Active clients:\n";
                for (int i = 0; i < MAX_CLIENTS; ++i) {
                    if (clients[i]) {
                        strcat(list, clients[i]->name);
                        strcat(list, " - ");
                        strcat(list, clients[i]->uid);
                        strcat(list, "\n");
                    }
                }
                send(cli->sockfd, list, strlen(list), 0); // Send list back to requester
            } 
            
            // Handling /history command
            if (strncmp(buffer, "/history ", 9) == 0) {
                char *recipient_id = buffer + 9;
                send_chat_history(cli->sockfd, cli->uid, recipient_id);
            }

            // Handling /history_delete command
            else if (strncmp(buffer, "/history_delete ", 16) == 0) {
                char *recipient_id = buffer + 16;
                delete_chat_history(cli->uid, recipient_id);
                send(cli->sockfd, "History deleted.\n", 17, 0);
            }

            // Handling /delete_all command
            else if (strcmp(buffer, "/delete_all") == 0) {
                delete_all_chat_history(cli->uid);
                send(cli->sockfd, "All history deleted.\n", 21, 0);
            }

            // Handle sending messages to specific clients
            else if (strncmp(buffer, "/send ", 6) == 0) {
               // Modified to store chat history
                char *token = strtok(buffer + 6, " "); // Get dest_id
                char *message = strtok(NULL, ""); // Get message

                if (token != NULL && message != NULL) {
                    for (int i = 0; i < MAX_CLIENTS; ++i) {
                        if (clients[i] && strcmp(clients[i]->uid, token) == 0) {
                            // Found the intended recipient, send the message
                            char formatted_message[2048];
                            sprintf(formatted_message, "%s: %s", cli->name, message);
                            send(clients[i]->sockfd, formatted_message, strlen(formatted_message), 0);
                            // Store message in chat history for both clients
                            store_message(cli->uid, clients[i]->uid, message);
                            store_message(clients[i]->uid, cli->uid, message);
                            break;
                        }
                    }
                }
            } else {
                // Generic message handling (broadcast to all)
                send_message(buffer, cli->uid);
                printf("%s\n", buffer);
            }
        } else if (receive == 0 || strcmp(buffer, "/logout") == 0) {
            // Handle client disconnection
            leave_flag = 1;
            // Send farewell message to the client
        } else {
            // Error occurred
            perror("ERROR");
            leave_flag = 1;
        }

        bzero(buffer, BUFFER_SIZE); // Clear the buffer
    }

    // Announce client's departure
    sprintf(buffer, "%s has left\n", cli->name);
    printf("%s", buffer);
    send_message(buffer, cli->uid);

    close(cli->sockfd); // Close the client's socket
    remove_client(cli->uid); // Remove client from the clients list
    free(cli); // Free the client structure
    pthread_detach(pthread_self()); // Detach the thread

    return NULL;
}


int main() {
    int option = 1;
    int listenfd = 0, connfd = 0;
    struct sockaddr_in serv_addr;
    struct sockaddr_in cli_addr;
    pthread_t tid;

    load_faqs();

    listenfd = socket(AF_INET, SOCK_STREAM, 0);
    serv_addr.sin_family = AF_INET;
    serv_addr.sin_addr.s_addr = INADDR_ANY;
    serv_addr.sin_port = htons(PORT);

    signal(SIGPIPE, SIG_IGN);

    setsockopt(listenfd, SOL_SOCKET, SO_REUSEADDR, (char*)&option, sizeof(option));

    if (bind(listenfd, (struct sockaddr*)&serv_addr, sizeof(serv_addr)) < 0) {
        perror("ERROR: Socket binding failed");
        return EXIT_FAILURE;
    }

    if (listen(listenfd, MAX_CLIENTS) < 0) {
        perror("ERROR: Socket listening failed");
        return EXIT_FAILURE;
    }

    printf("Server started on port %d\n", PORT);

    while (1) {
        socklen_t clilen = sizeof(cli_addr);
        connfd = accept(listenfd, (struct sockaddr*)&cli_addr, &clilen);

        // Checking for max clients is now based on the presence of a null in the clients array
        int slots_full = 1;
        for (int i = 0; i < MAX_CLIENTS; i++) {
            if (!clients[i]) {
                slots_full = 0; // Found an available slot
                break;
            }
        }

        if (slots_full) {
            printf("Max clients reached. Rejecting: ");
            print_client_addr(cli_addr);
            printf("\n");
            close(connfd);
            continue;
        }

        // Client settings
        client_t *cli = (client_t *)malloc(sizeof(client_t));
        cli->address = cli_addr;
        cli->sockfd = connfd;

        // Generate a unique ID (UUID) for each client
        uuid_t uuid;
        uuid_generate_random(uuid);
        uuid_unparse_lower(uuid, cli->uid);

        // Add client to the array
        add_client(cli);
        pthread_create(&tid, NULL, &handle_client, (void*)cli);

        // Reduce CPU usage
        sleep(1);
    }

    // Close the listening socket before exiting
    close(listenfd);

    return 0;
}
