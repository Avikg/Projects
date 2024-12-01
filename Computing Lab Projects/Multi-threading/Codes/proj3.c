#include <stdio.h>
#include <pthread.h>
#include <stdlib.h>
#include <stdbool.h>
#include <limits.h>
#include <curl/curl.h>
#include <zlib.h>
#include <time.h> // Added for the time function

#define NUM_NODES 58228
#define NUM_GRAPH_UPDATE_THREADS 5
#define NUM_PATH_FINDER_THREADS 20
#define NUM_PATH_STITCHER_THREADS 10
#define NUM_LANDMARKS 100
#define NUM_NODE_PAIRS 10000
#define INF INT_MAX

typedef struct Node {   
    int vertex;
    struct Node* next;
} Node;

Node* graph[NUM_NODES];
pthread_mutex_t graph_mutex;
pthread_mutex_t rand_mutex; // Mutex for rand function
int distances[NUM_NODES];
bool visited[NUM_NODES];
int landmarkNodes[NUM_LANDMARKS];
int nodeDegrees[NUM_NODES];
pthread_mutex_t *edge_mutexes; // Moved declaration to top for clarity

Node* createNode(int v) {
    Node* newNode = malloc(sizeof(Node));
    newNode->vertex = v;
    newNode->next = NULL;
    return newNode;
}

void addEdge(int src, int dest) {
    Node* newNode = createNode(dest);
    newNode->next = graph[src];
    graph[src] = newNode;
    nodeDegrees[src]++;

    newNode = createNode(src);
    newNode->next = graph[dest];
    graph[dest] = newNode;
    nodeDegrees[dest]++;
}

int findMinDistanceNode() {
    int minVal = INF, minIndex = -1;
    for (int v = 0; v < NUM_NODES; v++) {
        if (!visited[v] && distances[v] <= minVal) {
            minVal = distances[v];
            minIndex = v;
        }
    }
    return minIndex;
}

void dijkstra(int src) {
    pthread_mutex_lock(&graph_mutex); // Locking to ensure thread safety
    for (int i = 0; i < NUM_NODES; i++) {
        distances[i] = INF;
        visited[i] = false;
    }

    distances[src] = 0;

    for (int count = 0; count < NUM_NODES - 1; count++) {
        int u = findMinDistanceNode();
        visited[u] = true;

        Node* temp = graph[u];
        while (temp) {
            int v = temp->vertex;
            if (!visited[v] && distances[u] != INF && distances[u] + 1 < distances[v]) {
                distances[v] = distances[u] + 1;
            }
            temp = temp->next;
        }
    }
    pthread_mutex_unlock(&graph_mutex); // Unlocking after updates
}

void removeEdge(int src, int dest) {
    Node* temp = graph[src];
    Node* prev = NULL;
    while (temp) {
        if (temp->vertex == dest) {
            if (prev) {
                prev->next = temp->next;
            } else {
                graph[src] = temp->next;
            }
            free(temp);
            break;
        }
        prev = temp;
        temp = temp->next;
    }
}

bool edgeExists(int src, int dest) {
    Node* temp = graph[src];
    while (temp) {
        if (temp->vertex == dest) {
            return true;
        }
        temp = temp->next;
    }
    return false;
}

void initialize_edge_mutexes() {
    edge_mutexes = malloc(NUM_NODES * sizeof(pthread_mutex_t));
    for (int i = 0; i < NUM_NODES; i++) {
        pthread_mutex_init(&edge_mutexes[i], NULL);
    }
}

int** paths; // 2D array to store paths

void initialize_paths() {
    paths = malloc(NUM_NODES * sizeof(int*));
    for (int i = 0; i < NUM_NODES; i++) {
        paths[i] = malloc(NUM_NODES * sizeof(int));
        for (int j = 0; j < NUM_NODES; j++) {
            paths[i][j] = -1; // Initialize with -1 indicating no path
        }
    }
}

FILE *updateLog;
FILE *pathFoundLog;

void* graph_update_thread(void* arg) {
    pthread_mutex_lock(&rand_mutex); // Locking rand for thread safety
    double prob = (double)rand() / RAND_MAX;
    int src = rand() % NUM_NODES;
    int dest;
    do {
        dest = rand() % NUM_NODES;
    } while (src == dest);
    pthread_mutex_unlock(&rand_mutex); // Unlocking after generating random numbers

    if (prob < 0.8) {
        pthread_mutex_lock(&edge_mutexes[src]);
        if (edgeExists(src, dest)) {
            removeEdge(src, dest);
            removeEdge(dest, src);
            fprintf(updateLog, "<REMOVE> <%d, %d> %ld\n", src, dest, time(NULL));
        }
        pthread_mutex_unlock(&edge_mutexes[src]);
    } else {
        pthread_mutex_lock(&edge_mutexes[src]);
        if (!edgeExists(src, dest)) {
            addEdge(src, dest);
            fprintf(updateLog, "<ADD> <%d, %d> %ld\n", src, dest, time(NULL));
        }
        pthread_mutex_unlock(&edge_mutexes[src]);
    }
    return NULL;
}

void* path_stitcher_thread(void* arg) {
    pthread_mutex_lock(&rand_mutex); // Locking rand for thread safety
    int src, dest;
    do {
        src = rand() % NUM_NODES;
        dest = rand() % NUM_NODES;
    } while (src == dest);
    pthread_mutex_unlock(&rand_mutex); // Unlocking after generating random numbers

    int landmarkSrc = landmarkNodes[src % NUM_LANDMARKS];
    int landmarkDest = landmarkNodes[dest % NUM_LANDMARKS];

    int distanceToLandmarkSrc = paths[src][landmarkSrc];
    int distanceFromLandmarkDest = paths[dest][landmarkDest];
    int distanceBetweenLandmarks = paths[landmarkSrc][landmarkDest];

    int approximateDistance = distanceToLandmarkSrc + distanceFromLandmarkDest + distanceBetweenLandmarks;

    pthread_mutex_lock(&graph_mutex);
    if (approximateDistance < INF) {
        fprintf(pathFoundLog, "PATH_FOUND <%d,%d> <%d -> ... -> %d> %ld\n", src, dest, src, dest, time(NULL));
        removeEdge(src, dest);
        fprintf(pathFoundLog, "PATH_REMOVED <%d,%d> <%d -> ... -> %d> %ld\n", src, dest, src, dest, time(NULL));
    } else {
        fprintf(pathFoundLog, "PATH_NOT_FOUND <%d, %d> %ld\n", src, dest, time(NULL));
    }
    pthread_mutex_unlock(&graph_mutex);

    return NULL;    
}

void downloadAndUnzipDataset() {
    system("curl -O https://snap.stanford.edu/data/loc-brightkite_edges.txt.gz");
    system("gzip -d loc-brightkite_edges.txt.gz");
}

void loadGraph() {
    downloadAndUnzipDataset();

    FILE* file = fopen("loc-brightkite_edges.txt", "r");
    if (!file) {
        perror("Failed to open loc-brightkite_edges.txt");
        exit(EXIT_FAILURE);
    }
    int src, dest;
    while (fscanf(file, "%d %d", &src, &dest) != EOF) {
        addEdge(src, dest);
    }
    fclose(file);
}

void* path_finder_thread(void* arg) {
    pthread_mutex_lock(&rand_mutex); // Locking rand for thread safety
    int landmark = rand() % NUM_LANDMARKS;
    pthread_mutex_unlock(&rand_mutex); // Unlocking after generating random number
    dijkstra(landmark);

    pthread_mutex_lock(&graph_mutex);
    for (int i = 0; i < NUM_NODES; i++) {
        paths[landmark][i] = distances[i];
    }
    pthread_mutex_unlock(&graph_mutex);

    return NULL;
}

void selectLandmarks() {
    int selected[NUM_NODES] = {0};

    for (int i = 0; i < 50; i++) {
        int node;
        do {
            node = rand() % NUM_NODES;
        } while (selected[node]);
        landmarkNodes[i] = node;
        selected[node] = 1;
    }

    // Selecting 50 highest degree nodes
    for (int i = 50; i < 100; i++) {
        int maxDegree = -1;
        int maxNode = -1;
        for (int j = 0; j < NUM_NODES; j++) {
            if (!selected[j] && nodeDegrees[j] > maxDegree) {
                maxDegree = nodeDegrees[j];
                maxNode = j;
            }
        }
        landmarkNodes[i] = maxNode;
        selected[maxNode] = 1;
    }
}

void partitionNodes() {
    for (int i = 0; i < NUM_NODES; i++) {
        int partition = i % NUM_LANDMARKS;
        landmarkNodes[partition] = i;
    }
}

int main() {
    // Initialize the graph
    loadGraph();

    // Initialize the mutexes
    pthread_mutex_init(&graph_mutex, NULL);
    pthread_mutex_init(&rand_mutex, NULL); // Mutex for rand function
    initialize_edge_mutexes();
    initialize_paths();

    // Select landmarks and partition nodes
    selectLandmarks();
    partitionNodes();

    // Open log files for writing
    updateLog = fopen("update.log", "w");
    if (!updateLog) {
        perror("Failed to open update.log");
        exit(EXIT_FAILURE);
    }
    pathFoundLog = fopen("path_found.log", "w");
    if (!pathFoundLog) {
        perror("Failed to open path_found.log");
        exit(EXIT_FAILURE);
    }

    // Create threads
    pthread_t graphUpdateThreads[NUM_GRAPH_UPDATE_THREADS];
    pthread_t pathFinderThreads[NUM_PATH_FINDER_THREADS];
    pthread_t pathStitcherThreads[NUM_PATH_STITCHER_THREADS];

    for (int i = 0; i < NUM_GRAPH_UPDATE_THREADS; i++) {
        pthread_create(&graphUpdateThreads[i], NULL, graph_update_thread, NULL);
    }
    for (int i = 0; i < NUM_PATH_FINDER_THREADS; i++) {
        pthread_create(&pathFinderThreads[i], NULL, path_finder_thread, NULL);
    }
    for (int i = 0; i < NUM_PATH_STITCHER_THREADS; i++) {
        pthread_create(&pathStitcherThreads[i], NULL, path_stitcher_thread, NULL);
    }

    // Join threads to ensure they complete
    for (int i = 0; i < NUM_GRAPH_UPDATE_THREADS; i++) {
        pthread_join(graphUpdateThreads[i], NULL);
    }
    for (int i = 0; i < NUM_PATH_FINDER_THREADS; i++) {
        pthread_join(pathFinderThreads[i], NULL);
    }
    for (int i = 0; i < NUM_PATH_STITCHER_THREADS; i++) {
        pthread_join(pathStitcherThreads[i], NULL);
    }

    // Cleanup: Free graph memory and close log files
    for (int i = 0; i < NUM_NODES; i++) {
        while (graph[i] != NULL) {
            Node* temp = graph[i];
            graph[i] = graph[i]->next;
            free(temp);
        }
        free(paths[i]); // Freeing paths 2D array
        pthread_mutex_destroy(&edge_mutexes[i]); // Destroying edge mutexes
    }
    free(paths);
    free(edge_mutexes);

    // Close log files
    fclose(updateLog);
    fclose(pathFoundLog);

    // Destroy the mutexes
    pthread_mutex_destroy(&graph_mutex);
    pthread_mutex_destroy(&rand_mutex);

    return 0;
}
