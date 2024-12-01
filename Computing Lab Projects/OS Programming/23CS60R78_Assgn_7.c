#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <readline/readline.h>
#include <readline/history.h>
#include <ncurses.h>
#include <pthread.h>

#define MAX_INPUT_SIZE 1024

typedef struct {
    double *vector1;
    double *vector2;
    double *resultVector;
    int startIndex;
    int endIndex;
} ThreadData;

typedef struct {
    int lineCount;
    int wordCount;
    int charCount;
} EditorStatistics;

int is_builtin_command(char* cmd) {
    return (strncmp(cmd, "cd", 2) == 0 || strcmp(cmd, "exit") == 0 || 
            strcmp(cmd, "help") == 0 || strncmp(cmd, "vi ", 3) == 0 || 
            strncmp(cmd, "addvec ", 7) == 0 || strncmp(cmd, "subvec ", 7) == 0 || 
            strncmp(cmd, "dotprod ", 8) == 0);
}

double* read_vector_from_file(const char* filename, int* vecLength) {
    FILE *file = fopen(filename, "r");
    if (!file) {
        perror("Error opening file");
        return NULL;
    }

    double* vec = (double*) malloc(MAX_INPUT_SIZE * sizeof(double));
    int count = 0;
    while (fscanf(file, "%lf", &vec[count]) != EOF) {
        count++;
    }
    fclose(file);

    *vecLength = count;
    return vec;
}

void* compute_addvec(void* data) {
    ThreadData* td = (ThreadData*) data;
    for (int i = td->startIndex; i < td->endIndex; i++) {
        td->resultVector[i] = td->vector1[i] + td->vector2[i];
    }
    return NULL;
}

void perform_vector_addition(const char* file1, const char* file2, int threadCount) {
    int len1, len2;
    double* v1 = read_vector_from_file(file1, &len1);
    double* v2 = read_vector_from_file(file2, &len2);

    if (len1 != len2) {
        printf("Vectors have different dimensions!\n");
        free(v1);
        free(v2);
        return;
    }

    double* resultVector = (double*) malloc(len1 * sizeof(double));
    pthread_t threads[threadCount];
    ThreadData threadData[threadCount];
    int chunkSize = len1 / threadCount;

    for (int i = 0; i < threadCount; i++) {
        threadData[i].vector1 = v1;
        threadData[i].vector2 = v2;
        threadData[i].resultVector = resultVector;
        threadData[i].startIndex = i * chunkSize;
        threadData[i].endIndex = (i == threadCount - 1) ? len1 : (i + 1) * chunkSize;
        pthread_create(&threads[i], NULL, compute_addvec, &threadData[i]);
    }

    for (int i = 0; i < threadCount; i++) {
        pthread_join(threads[i], NULL);
    }

    for (int i = 0; i < len1; i++) {
        printf("%f ", resultVector[i]);
    }
    printf("\n");

    free(v1);
    free(v2);
    free(resultVector);
}

void* compute_subvec(void* data) {
    ThreadData* td = (ThreadData*) data;
    for (int i = td->startIndex; i < td->endIndex; i++) {
        td->resultVector[i] = td->vector1[i] - td->vector2[i];
    }
    return NULL;
}

void perform_vector_subtraction(const char* file1, const char* file2, int threadCount) {
    int len1, len2;
    double* v1 = read_vector_from_file(file1, &len1);
    double* v2 = read_vector_from_file(file2, &len2);

    if (len1 != len2) {
        printf("Vectors have different dimensions!\n");
        free(v1);
        free(v2);
        return;
    }

    double* resultVector = (double*) malloc(len1 * sizeof(double));
    pthread_t threads[threadCount];
    ThreadData threadData[threadCount];
    int chunkSize = len1 / threadCount;

    for (int i = 0; i < threadCount; i++) {
        threadData[i].vector1 = v1;
        threadData[i].vector2 = v2;
        threadData[i].resultVector = resultVector;
        threadData[i].startIndex = i * chunkSize;
        threadData[i].endIndex = (i == threadCount - 1) ? len1 : (i + 1) * chunkSize;
        pthread_create(&threads[i], NULL, compute_subvec, &threadData[i]);
    }

    for (int i = 0; i < threadCount; i++) {
        pthread_join(threads[i], NULL);
    }

    for (int i = 0; i < len1; i++) {
        printf("%f ", resultVector[i]);
    }
    printf("\n");

    free(v1);
    free(v2);
    free(resultVector);
}

void* compute_dotprod(void* data) {
    ThreadData* td = (ThreadData*) data;
    double sum = 0.0;
    for (int i = td->startIndex; i < td->endIndex; i++) {
        sum += td->vector1[i] * td->vector2[i];
    }
    *(td->resultVector) = sum;
    return NULL;
}

void perform_dot_product(const char* file1, const char* file2, int threadCount) {
    int len1, len2;
    double* v1 = read_vector_from_file(file1, &len1);
    double* v2 = read_vector_from_file(file2, &len2);

    if (len1 != len2) {
        printf("Vectors have different dimensions!\n");
        free(v1);
        free(v2);
        return;
    }

    double results[threadCount];
    pthread_t threads[threadCount];
    ThreadData threadData[threadCount];
    int chunkSize = len1 / threadCount;

    for (int i = 0; i < threadCount; i++) {
        threadData[i].vector1 = v1;
        threadData[i].vector2 = v2;
        threadData[i].resultVector = &results[i];
        threadData[i].startIndex = i * chunkSize;
        threadData[i].endIndex = (i == threadCount - 1) ? len1 : (i + 1) * chunkSize;
        pthread_create(&threads[i], NULL, compute_dotprod, &threadData[i]);
    }

    double dotProduct = 0.0;
    for (int i = 0; i < threadCount; i++) {
        pthread_join(threads[i], NULL);
        dotProduct += results[i];
    }

    printf("Dot Product: %f\n", dotProduct);
    free(v1);
    free(v2);
}

int count_words_in_string(const char *str) {
    int wordCount = 0;
    char *tempString = strdup(str);
    char *token = strtok(tempString, " \t\r\n");
    while (token) {
        wordCount++;
        token = strtok(NULL, " \t\r\n");
    }
    free(tempString);
    return wordCount;
}

EditorStatistics launch_vi_editor(char *filename) {
    int y, x;    
    int ch;      
    int max_y, max_x;
    char **editor_content = (char **)malloc(MAX_INPUT_SIZE * sizeof(char *));
    for (int i = 0; i < MAX_INPUT_SIZE; i++) {
        editor_content[i] = (char *)malloc(MAX_INPUT_SIZE * sizeof(char));
        memset(editor_content[i], 0, MAX_INPUT_SIZE);
    }

    FILE *file = fopen(filename, "r+");  
    if (!file) {
        file = fopen(filename, "w");     
        if (!file) {
            perror("Failed to create the file");
            exit(1);
        }
        fclose(file);
        file = fopen(filename, "r+");
    }
    fseek(file, 0, SEEK_SET);

    char line[MAX_INPUT_SIZE];
    int line_num = 0;
    while (fgets(line, sizeof(line), file) && line_num < MAX_INPUT_SIZE) {
        strncpy(editor_content[line_num], line, MAX_INPUT_SIZE - 1);
        editor_content[line_num][MAX_INPUT_SIZE - 1] = '\0';
        int len = strlen(editor_content[line_num]);
        if (len > 0 && editor_content[line_num][len - 1] == '\n') {
            editor_content[line_num][len - 1] = '\0';
        }
        line_num++;
    }
    fclose(file);

    initscr();
    cbreak();
    noecho();
    keypad(stdscr, TRUE);
    getmaxyx(stdscr, max_y, max_x);
    y = 0;
    x = 0;

    while (1) {
        clear();
        for (int i = 0; i < max_y && i < MAX_INPUT_SIZE; i++) {
            mvprintw(i, 0, "%s", editor_content[i]);
        }
        mvchgat(y, x, 1, A_REVERSE, 0, NULL);
        move(y, x);
        ch = getch();

        switch (ch) {
            case KEY_UP:
                if (y > 0) y--;
                break;
            case KEY_DOWN:
                if (y < max_y - 1 && editor_content[y+1][0] != '\0') y++;
                break;
            case KEY_LEFT:
                if (x > 0) x--;
                else if (y > 0) {
                    y--;
                    x = strlen(editor_content[y]);
                }
                break;
            case KEY_RIGHT:
                if (editor_content[y][x] != '\0') x++;
                else if (y < max_y - 1 && editor_content[y+1][0] != '\0') {
                    y++;
                    x = 0;
                }
                break;
            case 10: // Enter key
                if (y < max_y - 1) {
                    for (int i = max_y - 2; i >= y + 1; i--) {
                        strcpy(editor_content[i + 1], editor_content[i]);
                    }
                    editor_content[y+1][0] = '\0';
                    y++;
                    x = 0;
                }
                break;
            case KEY_BACKSPACE:
                if (x > 0) {
                    for (int i = x; i < strlen(editor_content[y]); i++) {
                        editor_content[y][i - 1] = editor_content[y][i];
                    }
                    editor_content[y][strlen(editor_content[y]) - 1] = '\0';
                    x--;
                } else if (y > 0) {
                    x = strlen(editor_content[y - 1]);
                    strcat(editor_content[y-1], editor_content[y]);
                    for (int i = y; i < max_y - 1; i++) {
                        strcpy(editor_content[i], editor_content[i+1]);
                    }
                    editor_content[max_y-1][0] = '\0';
                    y--;
                }
                break;
            case 27:  // Escape key
                endwin();

                file = fopen(filename, "w");
                for (int i = 0; i < MAX_INPUT_SIZE && editor_content[i][0] != '\0'; i++) {
                    fprintf(file, "%s\n", editor_content[i]);
                }
                fclose(file);

                EditorStatistics stats;
                stats.lineCount = 0;
                stats.wordCount = 0;
                stats.charCount = 0;
                for (int i = 0; i < MAX_INPUT_SIZE && editor_content[i][0] != '\0'; i++) {
                    stats.lineCount++;
                    stats.wordCount += count_words_in_string(editor_content[i]);
                    stats.charCount += strlen(editor_content[i]);
                }

                for (int i = 0; i < MAX_INPUT_SIZE; i++) {
                    free(editor_content[i]);
                }
                free(editor_content);

                return stats;
            default:
                if (ch >= 32 && ch <= 126) {
                    for (int i = strlen(editor_content[y]); i >= x; i--) {
                        editor_content[y][i+1] = editor_content[y][i];
                    }
                    editor_content[y][x] = ch;
                    x++;
                }
                break;
        }
    }
}


void handle_builtin_command(char* user_input) {
    if (strcmp(user_input, "cd") == 0) {
        printf("Usage: cd <directory_name>\n");
    } else if (strncmp(user_input, "cd ", 3) == 0) {
        char *directory = user_input + 3;
        if (chdir(directory) == -1) {
            perror("cd failed");
        }
    } else if (strcmp(user_input, "exit") == 0) {
        printf("Exiting shell...\n");
        exit(0);
    } else if (strcmp(user_input, "help") == 0) {
        printf("Available commands:\n");
        printf("1. pwd\n");
        printf("2. cd <directory_name>\n");
        printf("3. mkdir <directory_name>\n");
        printf("4. ls <flag>\n");
        printf("5. exit\n");
        printf("6. help\n");
    } else if (strncmp(user_input, "vi ", 3) == 0) {
        char *filename = user_input + 3;
        EditorStatistics stats = launch_vi_editor(filename);
        printf("Lines: %d, Words: %d, Characters: %d\n", stats.lineCount, stats.wordCount, stats.charCount);
    } else if (strncmp(user_input, "addvec ", 7) == 0) {
        char* file1 = strtok(user_input + 7, " ");
        char* file2 = strtok(NULL, " ");
        char* threadArg = strtok(NULL, " ");
        int threadCount = 3; // default value
        if (threadArg && threadArg[0] == '-') {
            threadCount = atoi(threadArg + 1);
        }
        perform_vector_addition(file1, file2, threadCount);
    } else if (strncmp(user_input, "subvec ", 7) == 0) {
        char* file1 = strtok(user_input + 7, " ");
        char* file2 = strtok(NULL, " ");
        char* threadArg = strtok(NULL, " ");
        int threadCount = 3; // default value
        if (threadArg && threadArg[0] == '-') {
            threadCount = atoi(threadArg + 1);
        }
        perform_vector_subtraction(file1, file2, threadCount);
    } else if (strncmp(user_input, "dotprod ", 8) == 0) {
        char* file1 = strtok(user_input + 8, " ");
        char* file2 = strtok(NULL, " ");
        char* threadArg = strtok(NULL, " ");
        int threadCount = 3; // default value
        if (threadArg && threadArg[0] == '-') {
            threadCount = atoi(threadArg + 1);
        }
        perform_dot_product(file1, file2, threadCount);
    }
}

void execute_single_command(char *cmd, int input_fd, int output_fd) {
    char *args[MAX_INPUT_SIZE];
    int index = 0;
    char *token = strtok(cmd, " ");
    while (token) {
        args[index++] = token;
        token = strtok(NULL, " ");
    }
    args[index] = NULL;

    pid_t pid = fork();
    if (pid == 0) {
        if (input_fd != STDIN_FILENO) { dup2(input_fd, STDIN_FILENO); close(input_fd); }
        if (output_fd != STDOUT_FILENO) { dup2(output_fd, STDOUT_FILENO); close(output_fd); }
        if (execvp(args[0], args) == -1) {
            perror("Command execution failed");
            exit(1);
        }
    } else {
        waitpid(pid, NULL, 0);
    }
}

void handle_external_command(char* user_input) {
    char *pipe_symbol = strchr(user_input, '|');
    if (pipe_symbol) {
        *pipe_symbol = '\0';
        char *cmd1 = user_input;
        char *cmd2 = pipe_symbol + 1;

        int pipe_fd[2];
        pipe(pipe_fd);

        execute_single_command(cmd1, STDIN_FILENO, pipe_fd[1]);
        close(pipe_fd[1]);
        execute_single_command(cmd2, pipe_fd[0], STDOUT_FILENO);
        close(pipe_fd[0]);
    } else {
        execute_single_command(user_input, STDIN_FILENO, STDOUT_FILENO);
    }
}

int main() {
    using_history();

    while (1) {
        char *input = readline("shell> ");
        if (!input) {
            printf("\nExiting shell..\n");
            break;
        }
        while (input[strlen(input) - 1] == '\\') {
            input[strlen(input) - 1] = '\0';
            char *next_line = readline("> ");
            input = realloc(input, strlen(input) + strlen(next_line) + 1);
            strcat(input, next_line);
            free(next_line);
        }

        if (input[0] != '\0') {
            add_history(input);
        }
        if (is_builtin_command(input)) {
            handle_builtin_command(input);
        } else {
            handle_external_command(input);
        }

        free(input);
    }

    return 0;
}
