#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <time.h>

#define DATASET_PATH "rating/ml-100k.csv"

#define LATENT_DIM 100
#define MAX_ITER 30
#define ALS_TOL_DIFF 1e-3
#define ALS_TOL 0.1
#define CG_MAX_ITER 100
#define CG_TOL 1e-6
#define LAMBDA 10

#define INITIAL_CAPACITY 1000000

#define MIN_USER_RATINGS_FOR_TEST 10
#define TEST_RATIO 0.2

typedef struct {
    int *row_ptr;
    int *col_indices;
    float *values;
    int num_users;
    int num_movies;
    int num_ratings;
} SparseMatrixCSR;

typedef struct {
    int user_id;
    float rating;
} UserRating;

typedef struct {
    UserRating *entries;
    int count;
    int capacity;
} MovieUserList;

typedef struct EntryNode {
    int col;
    float val;
    struct EntryNode *next;
} EntryNode;

typedef struct {
    int u, m;
    float r;
} Record;


Record *test_set = NULL;
int test_count = 0;

SparseMatrixCSR *read_as_sparse_matrix_with_split(const char *path) {
    char line[128];
    int max_user_id = 0, max_movie_id = 0;
    int total_lines = 0;
    int capacity = INITIAL_CAPACITY;

    Record *records = malloc(capacity * sizeof(Record));
    if (!records) {
        fprintf(stderr, "Failed to allocate initial memory.\n");
        exit(1);
    }

    FILE *fp = fopen(path, "r");
    if (!fp) {
        perror("Failed to open file");
        exit(1);
    }

    while (fgets(line, sizeof(line), fp)) {
        int u, m;
        float r;
        if (sscanf(line, "%d,%d,%f", &u, &m, &r) != 3) continue;

        // 动态扩容逻辑
        if (total_lines >= capacity) {
            capacity *= 2;
            records = realloc(records, capacity * sizeof(Record));
            if (!records) {
                fprintf(stderr, "Failed to realloc memory.\n");
                exit(1);
            }
        }

        if (u > max_user_id) max_user_id = u;
        if (m > max_movie_id) max_movie_id = m;

        records[total_lines++] = (Record){u, m, r};
    }

    fclose(fp);

    // count ratings group by user id
    int *user_rating_count = NULL;
    user_rating_count = calloc(max_user_id, sizeof(int));
    for (int i = 0; i < total_lines; i++) {
        user_rating_count[records[i].u - 1]++;
    }

    int *user_map = malloc(max_user_id * sizeof(int));
    int *movie_map = malloc(max_movie_id * sizeof(int));
    memset(user_map, -1, max_user_id * sizeof(int));
    memset(movie_map, -1, max_movie_id * sizeof(int));

    int num_users = 0, num_movies = 0, num_ratings = 0;
    test_set = malloc(total_lines * sizeof(Record));

    EntryNode **user_rows = calloc(max_user_id, sizeof(EntryNode *));

    // srand(time(NULL));

    for (int i = 0; i < total_lines; i++) {
        int u_raw = records[i].u;
        int m_raw = records[i].m;
        float rating = records[i].r;

        if (user_map[u_raw - 1] == -1) user_map[u_raw - 1] = num_users++;
        if (movie_map[m_raw - 1] == -1) movie_map[m_raw - 1] = num_movies++;

        int uid = user_map[u_raw - 1];
        int mid = movie_map[m_raw - 1];

        if (user_rating_count[u_raw - 1] >= MIN_USER_RATINGS_FOR_TEST && ((float) rand() / RAND_MAX < TEST_RATIO)) {
            test_set[test_count++] = (Record){uid, mid, rating};
            continue;
        }

        // insert into linkedlist
        EntryNode *new_node = malloc(sizeof(EntryNode));
        new_node->col = mid;
        new_node->val = rating;
        new_node->next = user_rows[uid];
        user_rows[uid] = new_node;
        num_ratings++;
    }

    // build CSR
    int *row_ptr = malloc((num_users + 1) * sizeof(int));
    int *col_indices = malloc(num_ratings * sizeof(int));
    float *values = malloc(num_ratings * sizeof(float));
    int nnz = 0;
    row_ptr[0] = 0;

    for (int i = 0; i < num_users; i++) {
        EntryNode *node = user_rows[i];
        while (node) {
            col_indices[nnz] = node->col;
            values[nnz++] = node->val;
            EntryNode *tmp = node;
            node = node->next;
            free(tmp);
        }
        row_ptr[i + 1] = nnz;
    }

    SparseMatrixCSR *R = malloc(sizeof(SparseMatrixCSR));
    R->row_ptr = row_ptr;
    R->col_indices = col_indices;
    R->values = values;
    R->num_users = num_users;
    R->num_movies = num_movies;
    R->num_ratings = num_ratings;

    free(user_map);
    free(movie_map);
    free(user_rating_count);
    free(user_rows);
    free(records);

    printf("[Load dataset] total data num: %d\n", total_lines);
    printf("[Load dataset] train data num: %d\n", num_ratings);
    printf("[Load dataset] test data num: %d\n", test_count);

    return R;
}


float compute_test_rmse(float *U, float *V) {
    float sum = 0.0f;
    int count_valid = 0;

    for (int i = 0; i < test_count; i++) {
        int u = test_set[i].u;
        int m = test_set[i].m;
        float r_true = test_set[i].r;

        float r_pred = 0.0f;
        for (int k = 0; k < LATENT_DIM; k++) {
            r_pred += U[u * LATENT_DIM + k] * V[m * LATENT_DIM + k];
        }

        if (isnan(r_pred)) {
            continue;
        }

        float diff = r_true - r_pred;
        float sq = diff * diff;

        sum += sq;
        count_valid++;
    }

    return sqrt(sum / count_valid);
}

MovieUserList *build_movie_to_user_map_c(const SparseMatrixCSR *R) {
    int num_movies = R->num_movies;

    MovieUserList *movie_map = malloc(num_movies * sizeof(MovieUserList));

    for (int i = 0; i < num_movies; i++) {
        movie_map[i].count = 0;
        movie_map[i].capacity = 4;
        movie_map[i].entries = malloc(4 * sizeof(UserRating));
    }

    for (int user = 0; user < R->num_users; user++) {
        for (int idx = R->row_ptr[user]; idx < R->row_ptr[user + 1]; idx++) {
            int movie = R->col_indices[idx];
            float rating = R->values[idx];

            MovieUserList *list = &movie_map[movie];

            if (list->count == list->capacity) {
                list->capacity *= 2;
                list->entries = realloc(list->entries, list->capacity * sizeof(UserRating));
            }

            list->entries[list->count].user_id = user;
            list->entries[list->count].rating = rating;
            list->count++;
        }
    }

    return movie_map;
}

void vector_add(float *x, float *y, float alpha, int n) {
    for (int i = 0; i < n; i++) x[i] += alpha * y[i];
}

void vector_sub(float *x, float *y, float alpha, int n) {
    for (int i = 0; i < n; i++) x[i] -= alpha * y[i];
}

float dot_product(float *a, float *b, int n) {
    float sum = 0.0f;
    for (int i = 0; i < n; i++) sum += a[i] * b[i];
    return sum;
}

// solve Ax = b using CG
void cg_solve(float *A, float *b, float *x, float *r, float *p, float *Ap) {
    float alpha, beta, rsold, rsnew;

    // init r = b - A * x，p = r
    for (int i = 0; i < LATENT_DIM; i++) {
        r[i] = b[i];
        for (int j = 0; j < LATENT_DIM; j++) {
            r[i] -= A[i * LATENT_DIM + j] * x[j];
        }
        p[i] = r[i];
    }

    rsold = dot_product(r, r, LATENT_DIM);

    for (int iter = 0; iter < CG_MAX_ITER; iter++) {
        // Ap = A * p
        for (int i = 0; i < LATENT_DIM; i++) {
            Ap[i] = 0.0f;
            for (int j = 0; j < LATENT_DIM; j++) {
                Ap[i] += A[i * LATENT_DIM + j] * p[j];
            }
        }

        alpha = rsold / dot_product(p, Ap, LATENT_DIM);
        vector_add(x, p, alpha, LATENT_DIM); // x += alpha * p
        vector_sub(r, Ap, alpha, LATENT_DIM); // r -= alpha * Ap

        rsnew = dot_product(r, r, LATENT_DIM);
        if (sqrtf(rsnew) < CG_TOL) {
            // printf("[Debug] CG Converged at iter %d\n", iter);
            break;
        }

        beta = rsnew / rsold;
        for (int i = 0; i < LATENT_DIM; i++) {
            p[i] = r[i] + beta * p[i];
        }

        rsold = rsnew;
    }
}

void als_update_U(SparseMatrixCSR *R, float *U, float *V, int rank, int size) {
    float *A = malloc(LATENT_DIM * LATENT_DIM * sizeof(float));
    float *b = malloc(LATENT_DIM * sizeof(float));
    float *r = malloc(LATENT_DIM * sizeof(float));
    float *p = malloc(LATENT_DIM * sizeof(float));
    float *Ap = malloc(LATENT_DIM * sizeof(float));

    int num_users = R->num_users;

    int users_per_proc = num_users / size;
    int extra_users = num_users % size;
    int local_user_start = rank * users_per_proc + (rank < extra_users ? rank : extra_users);
    int local_user_count = users_per_proc + (rank < extra_users ? 1 : 0);
    int local_user_end = local_user_start + local_user_count;

    for (int i = local_user_start; i < local_user_end; i++) {
        int global_u = i;
        int idx_start = R->row_ptr[i];
        int idx_end = R->row_ptr[i + 1];

        memset(A, 0, LATENT_DIM * LATENT_DIM * sizeof(float));
        memset(b, 0, LATENT_DIM * sizeof(float));

        for (int j = idx_start; j < idx_end; j++) {
            int movie_id = R->col_indices[j];
            float rating = R->values[j];
            for (int k1 = 0; k1 < LATENT_DIM; k1++) {
                for (int k2 = 0; k2 < LATENT_DIM; k2++) {
                    A[k1 * LATENT_DIM + k2] += V[movie_id * LATENT_DIM + k1] * V[movie_id * LATENT_DIM + k2];
                }
                b[k1] += rating * V[movie_id * LATENT_DIM + k1];
            }
        }
        for (int k = 0; k < LATENT_DIM; k++) {
            A[k * LATENT_DIM + k] += LAMBDA;
        }

        cg_solve(A, b, &U[global_u * LATENT_DIM], r, p, Ap);
    }


    // gather U
    int *recvcounts = malloc(size * sizeof(int));
    int *displs = malloc(size * sizeof(int));

    for (int r = 0; r < size; r++) {
        int count = num_users / size + (r < num_users % size ? 1 : 0);
        recvcounts[r] = count * LATENT_DIM;
        displs[r] = (r == 0) ? 0 : displs[r - 1] + recvcounts[r - 1];
    }

    float *sendbuf = &U[local_user_start * LATENT_DIM];
    float *recvbuf = malloc(num_users * LATENT_DIM * sizeof(float));

    MPI_Allgatherv(sendbuf, local_user_count * LATENT_DIM, MPI_FLOAT,
                   recvbuf, recvcounts, displs, MPI_FLOAT,
                   MPI_COMM_WORLD);

    memcpy(U, recvbuf, num_users * LATENT_DIM * sizeof(float));


    free(A);
    free(b);
    free(r);
    free(p);
    free(Ap);

    free(recvcounts);
    free(displs);
    free(recvbuf);

}

void als_update_V(float *U, float *V, MovieUserList *movie_to_users, int num_movies, int rank, int size) {
    float *A = malloc(LATENT_DIM * LATENT_DIM * sizeof(float));
    float *b = malloc(LATENT_DIM * sizeof(float));
    float *r = malloc(LATENT_DIM * sizeof(float));
    float *p = malloc(LATENT_DIM * sizeof(float));
    float *Ap = malloc(LATENT_DIM * sizeof(float));

    int movies_per_proc = num_movies / size;
    int extra_movies = num_movies % size;
    int local_movie_start = rank * movies_per_proc + (rank < extra_movies ? rank : extra_movies);
    int local_movie_count = movies_per_proc + (rank < extra_movies ? 1 : 0);
    int local_movie_end = local_movie_start + local_movie_count;

    for (int j = local_movie_start; j < local_movie_end; j++) {
        memset(A, 0, LATENT_DIM * LATENT_DIM * sizeof(float));
        memset(b, 0, LATENT_DIM * sizeof(float));

        MovieUserList *users = &movie_to_users[j];

        for (int u = 0; u < users->count; u++) {
            int user_id = users->entries[u].user_id;
            float rating = users->entries[u].rating;

            for (int k1 = 0; k1 < LATENT_DIM; k1++) {
                for (int k2 = 0; k2 < LATENT_DIM; k2++) {
                    A[k1 * LATENT_DIM + k2] +=
                            U[user_id * LATENT_DIM + k1] * U[user_id * LATENT_DIM + k2];
                }
                b[k1] += rating * U[user_id * LATENT_DIM + k1];
            }
        }

        for (int k = 0; k < LATENT_DIM; k++) A[k * LATENT_DIM + k] += LAMBDA;

        cg_solve(A, b, &V[j * LATENT_DIM], r, p, Ap);
    }

    // gather V
    int *recvcounts = malloc(size * sizeof(int));
    int *displs = malloc(size * sizeof(int));

    for (int r = 0; r < size; r++) {
        int count = num_movies / size + (r < num_movies % size ? 1 : 0);
        recvcounts[r] = count * LATENT_DIM;
        displs[r] = (r == 0) ? 0 : displs[r - 1] + recvcounts[r - 1];
    }

    float *sendbuf = &V[local_movie_start * LATENT_DIM];
    float *recvbuf = malloc(num_movies * LATENT_DIM * sizeof(float));

    MPI_Allgatherv(sendbuf, local_movie_count * LATENT_DIM, MPI_FLOAT,
                   recvbuf, recvcounts, displs, MPI_FLOAT,
                   MPI_COMM_WORLD);

    memcpy(V, recvbuf, num_movies * LATENT_DIM * sizeof(float));

    free(A);
    free(b);
    free(r);
    free(p);
    free(Ap);

    free(recvcounts);
    free(displs);
    free(recvbuf);
}

float compute_rmse_parallel(SparseMatrixCSR *R, float *U, float *V, int rank, int size,
                            MPI_Comm comm) {
    int num_users = R->num_users;

    int users_per_proc = num_users / size;
    int extra_users = num_users % size;
    int local_user_start = rank * users_per_proc + (rank < extra_users ? rank : extra_users);
    int local_user_count = users_per_proc + (rank < extra_users ? 1 : 0);
    int local_user_end = local_user_start + local_user_count;

    float local_error = 0.0;
    int local_nnz = 0;
    for (int u = local_user_start; u < local_user_end; u++) {
        for (int j = R->row_ptr[u]; j < R->row_ptr[u + 1]; j++) {
            int m = R->col_indices[j];
            float R_true = R->values[j];
            float R_pred = 0.0;
            for (int k = 0; k < LATENT_DIM; k++) {
                R_pred += U[u * LATENT_DIM + k] * V[m * LATENT_DIM + k];
            }
            float error = R_true - R_pred;
            local_error += error * error;
            local_nnz++;
        }
    }

    float global_error;
    int global_nnz;

    MPI_Allreduce(&local_error, &global_error, 1, MPI_FLOAT, MPI_SUM, comm);
    MPI_Allreduce(&local_nnz, &global_nnz, 1, MPI_INT, MPI_SUM, comm);

    return sqrt(global_error / global_nnz);
}


int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);

    double t_all_start = MPI_Wtime();

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    SparseMatrixCSR *R = NULL;
    int nnz;
    int num_users;
    int num_movies;

    // Only rank 0 reads the dataset and builds CSR matrix
    if (rank == 0) {
        double tgen_start = MPI_Wtime();
        R = read_as_sparse_matrix_with_split(DATASET_PATH);
        double tgen_end = MPI_Wtime();
        printf("[Read dataset and build CSR matrix] %.4f seconds \n", tgen_end - tgen_start);

        nnz = R->num_ratings;
        num_users = R->num_users;
        num_movies = R->num_movies;
    }

    // Broadcast CSR matrix metadata to all ranks
    double t_bcast_start = MPI_Wtime();
    MPI_Bcast(&nnz, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&num_users, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&num_movies, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // Allocate and broadcast CSR matrix content to all other ranks
    if (rank != 0) {
        R = malloc(sizeof(SparseMatrixCSR));
        R->num_users = num_users;
        R->num_movies = num_movies;
        R->num_ratings = nnz;

        R->row_ptr = malloc((num_users + 1) * sizeof(int));
        R->col_indices = malloc(nnz * sizeof(int));
        R->values = malloc(nnz * sizeof(float));
    }

    MPI_Bcast(R->row_ptr, num_users + 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(R->col_indices, nnz, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(R->values, nnz, MPI_FLOAT, 0, MPI_COMM_WORLD);


    // Allocate and initialize latent matrices U and V
    float *U = malloc(num_users * LATENT_DIM * sizeof(float));
    float *V = malloc(num_movies * LATENT_DIM * sizeof(float));

    // init U and V with random value to facilitate converge on rank 0
    if (rank == 0) {
        for (int i = 0; i < num_users * LATENT_DIM; i++)
            U[i] = ((float) rand() / RAND_MAX) * 0.1f;
        for (int j = 0; j < num_movies * LATENT_DIM; j++)
            V[j] = ((float) rand() / RAND_MAX) * 0.1f;
    }

    // Broadcast U and V to all ranks
    MPI_Bcast(U, num_users * LATENT_DIM, MPI_FLOAT, 0, MPI_COMM_WORLD);
    MPI_Bcast(V, num_movies * LATENT_DIM, MPI_FLOAT, 0, MPI_COMM_WORLD);

    double t_bcast_end = MPI_Wtime();

    double t_build_map_start = MPI_Wtime();
    // Build inverted index: map from movie to users who rated it
    MovieUserList *movie_to_users = build_movie_to_user_map_c(R);
    double t_build_map_end = MPI_Wtime();

    if (rank == 0) {
        printf("[MPI_Bcast] %.4f seconds \n", t_bcast_end - t_bcast_start);
        printf("[Build movie user map] %.4f seconds \n", t_build_map_end - t_build_map_start);
    }

    // Start ALS training loop
    float prev_rmse = 1e10f;
    for (int iter = 0; iter < MAX_ITER; iter++) {
        double t0 = MPI_Wtime();

        // Update U on this rank's assigned users
        als_update_U(R, U, V, rank, size);

        double t1 = MPI_Wtime();

        // Update V on this rank's assigned movies
        als_update_V(U, V, movie_to_users, num_movies, rank, size);

        double t2 = MPI_Wtime();

        // Compute training RMSE parallely
        float curr_rmse = compute_rmse_parallel(R, U, V, rank, size, MPI_COMM_WORLD);

        double t3 = MPI_Wtime();
        if (rank == 0) {
            printf("[ALS Iter %d]\n", iter);
            printf("Train Data RMSE = %.6f \n", curr_rmse);
            printf("Time (ALS U-update): %.4fs \n", t1 - t0);
            printf("Time (ALS V-update): %.4fs \n", t2 - t1);
            printf("Time (RMSE): %.4fs \n", t3 - t2);
            printf("Total iteration time: %.4fs \n\n", t3 - t0);
        }

        if (fabs(curr_rmse - prev_rmse) < ALS_TOL_DIFF || curr_rmse < ALS_TOL) {
            if (rank == 0) printf("ALS Converged at iter %d\n", iter);
            break;
        }
        prev_rmse = curr_rmse;
    }

    double t7 = MPI_Wtime();

    if (rank == 0) {
        // Output final test RMSE on rank 0
        float test_rmse = compute_test_rmse(U, V);
        double t8 = MPI_Wtime();
        printf("\nTime (TEST_RMSE):%.4fs\n", t8 - t7);
        printf("Test Data RMSE:%.4f \n", test_rmse);

        printf("Total Execution Time %.4f seconds \n", t8 - t_all_start);
    }


    for (int j = 0; j < R->num_movies; j++) {
        free(movie_to_users[j].entries);
    }
    free(movie_to_users);
    free(U);
    free(V);
    free(R->row_ptr);
    free(R->col_indices);
    free(R->values);
    free(R);

    MPI_Finalize();
    return 0;
}
