//KHALID & VIDYA


#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
//----------------------------------DATA GENERATION------------------------------------------------------------
// random data generation
float* random_data(int total_datapoints) {
    float *data = (float *)malloc(sizeof(float) * total_datapoints);
    for (int i = 0; i < total_datapoints; i++) {
        data[i] = (rand()/ (float)(RAND_MAX)) * 100;
    }
    return data;
}
//------------------------------------DISTANCE----------------------------------------------------------------
// calculate distance between two data points according to euclidean formula
float distance(float *pnt1, float *pnt2, int dim) {
    float dist = 0.0;
    for (int i = 0; i<dim; i++) {
        float temp = pnt1[i] - pnt2[i];
        temp = temp * temp;
        dist += temp;
    }
    return dist;
}
//----------------------------------------CENTROIDS-------------------------------------------------------------
//function to print the centroids
void centroids(float * centroids, int cluster_num, int dim) {

    printf("---------CENTROIDS FOUND----------\n");
    printf("\n{");
    for (int i = 0; i < cluster_num * dim; i++) {
        printf("%f,", centroids[i]);
    }
    printf("}\n");
    printf("==================================================================================\n");
}
//-----------------------------------------SEARCH--------------------------------------------------------------
//function for search nearest data point
void search(float *query, float *datapoints, int dim){
    float shortest_path=0.0,dist=0.0;
    int near = 0,count=0;
    for(int i=0; i< 10000; i+=dim)
    {
        dist= distance(query,&datapoints[i], dim);
        if(i==0)
        {
            shortest_path=dist;
            near =i;
        }
        if(shortest_path > dist)
        {
            shortest_path=dist;
            near =i;
        }
        count++;
    }
    printf("\nTotal no of Data Points visited:%d\n",count);
    printf("\nShortest distance is=%f\n",shortest_path);
    printf("\nNearest datapoint to querypoint is { ");
    for(int k=near;k<dim+near;k++ )
    {
        printf(" %f, ",datapoints[k]);
    }
    printf("}\n");
}
//----------------------------------------Main Function---------------------------------------------------------
int main(int argc,char *argv[]) {

    int no_of_datapoints=10000,dim=128;

//-------------MPI Functions--------------------------------------------------------------------
    MPI_Init(&argc, &argv); //initializing the MPI process
    int cluster_rank, no_of_clusters; //no of clusters is no of processor running
    MPI_Comm_rank(MPI_COMM_WORLD, &cluster_rank); //rank of the MPI process
    MPI_Comm_size(MPI_COMM_WORLD, &no_of_clusters); //size of the MPI process
    printf( "Hello world from process %d of %d\n", cluster_rank, no_of_clusters );

    //----------------------------------------------------------------------------------------------
    int data_points_per_cluster = no_of_datapoints / no_of_clusters; //data points per cluster
    int total_of_cluster = no_of_clusters;// total number 0f clusters.

//----------------------Memory allocation for pointers------------------------------------------

    //dataarray per cluster size
    float* datapoints_buffer = malloc(dim * data_points_per_cluster * sizeof(float));
    //cluster size array per cluster
    float* cluster_size = malloc(dim * total_of_cluster * sizeof(float));
    //total cluster size based on no of clusters
    int* cluster = malloc(total_of_cluster * sizeof(int));
    // array for storing cluster centroids
    float* initial_centroids = malloc(dim * total_of_cluster * sizeof(float));
    //  cluster on every process
    int* cluster_process = malloc(data_points_per_cluster * sizeof(int));
//---------------------Initializing the pointer array---------------------------------------
    float* initial_dataset = NULL; //dataset of each cluster
    float* total_data = NULL; //total data points in cluster
    int* total_clusters = NULL; //total no of clusters
    int* new_centroids = NULL;
//--------------------------------------------------------------------------------------------
//-----------------------------Actual Task----------------------------------------------------
    if (cluster_rank == 0)
    {

        // function call for random data generation
        initial_dataset = random_data(dim * data_points_per_cluster * no_of_clusters);

        // initial k centroids taken randomly
        for (int i = 0; i < dim * total_of_cluster; i++) {
            initial_centroids[i] = initial_dataset[i];
        }
        printf("INITIAL CENTROIDS\n");
        centroids(initial_centroids, total_of_cluster, dim);//call for printing the centroids


        //--------------------------memory allocation------------------------------
        total_data = malloc(dim * total_of_cluster * sizeof(float));
        total_clusters = malloc(total_of_cluster * sizeof(int));
        new_centroids = malloc(no_of_clusters * data_points_per_cluster * sizeof(int));
    }


//------------------------ send initial dataset to all process-----------------------------------------
    MPI_Scatter(initial_dataset, dim *data_points_per_cluster, MPI_FLOAT, datapoints_buffer,
                dim*data_points_per_cluster, MPI_FLOAT, 0, MPI_COMM_WORLD);
    float min_dist = 1.0;
    while (min_dist > 0.00001) { // Check for threshhold
//-------------------- Broadcast the current cluster centroids to all processes--------------------------
        MPI_Bcast(initial_centroids, dim * total_of_cluster, MPI_FLOAT, 0, MPI_COMM_WORLD);
        //--------initializing values for array-------------
        for (int i = 0; i < total_of_cluster; i++) {
            cluster[i] = 0;
        }
        for (int i = 0; i < dim * total_of_cluster; i++) {
            cluster_size[i] = 0.0;
        }
        float* current_data_point = datapoints_buffer;
//----------for each datapoint calculate the nearest centroid and update the sum of the centroid-----------------
        for (int i = 0; i < data_points_per_cluster; i++) {
            //recalculate datapoint to new cluster
            int new_cluster_index = 0; //initial cluster index
            //distance from datapoints to initial centroids
            float dist = distance(current_data_point, initial_centroids, dim);
            //new cluster to which datapoint need to be assigned
            float* nextCluster = initial_centroids + dim;

            for (int c = 1; c < total_of_cluster; c++, nextCluster += dim) {
                float new_dist = distance(current_data_point, nextCluster, dim);
                if (dist > new_dist) {
                    new_cluster_index = c;
                    dist = new_dist;
                }
            }
            cluster[new_cluster_index]++;
            //-------- update the sum vector for centroid----------------
            float *sum_of_cluster = &cluster_size[new_cluster_index*dim];
            for (int i = 0; i<dim; i++) {
                sum_of_cluster[i] += current_data_point[i];
            }
            current_data_point += dim;
        }
//----------------------------------------------------------------------------------------------------------
        MPI_Reduce(cluster_size, total_data, dim * total_of_cluster, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);
        MPI_Reduce(cluster, total_clusters, total_of_cluster, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

        if (cluster_rank == 0) {
            //---- calculate the new centroids-------------------------------
            for (int i = 0; i<total_of_cluster; i++) {
                for (int j = 0; j<dim; j++) {
                    int tmp = dim*i + j;
                    total_data[tmp] = total_data[tmp] / total_clusters[i];
                }
            }
            //------------------ check for the minimum centroid--------------------
            min_dist = distance(total_data, initial_centroids, dim * total_of_cluster);
            for (int i = 0; i<dim * total_of_cluster; i++) {
                initial_centroids[i] = total_data[i];
            }
            printf("Next SET OF CENTROIDS");
            centroids(initial_centroids, total_of_cluster, dim);
        }
        //--------------broadcast flag to each process----------------------------------
        MPI_Bcast(&min_dist, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
    }
    //-------------------------- centroid are fixed----------------------------------
    float* data_point = datapoints_buffer;
    for (int i = 0; i < data_points_per_cluster; i++, data_point += dim) {
        int new_cluster_index = 0;
        float new_dist = distance(data_point, initial_centroids, dim);

        float* nextCentroid = initial_centroids + dim;
        for (int c = 1; c < total_of_cluster; c++, nextCentroid += dim) {
            float dist = distance(data_point, nextCentroid, dim);
            if (dist < new_dist) {
                new_cluster_index = c;
                new_dist = dist;
            }
        }
        cluster_process[i] = new_cluster_index;
    }
//--------------------------------------------------------------------------------------------------------
    MPI_Gather(cluster_process, data_points_per_cluster, MPI_INT,
               new_centroids, data_points_per_cluster, MPI_INT, 0, MPI_COMM_WORLD);
//------------------------------------------------------------------------------------------------------------
    if (cluster_rank == 0)
    {
        float* data = initial_dataset;
        for (int i = 0; i < no_of_clusters * data_points_per_cluster; i++) {
            printf("-----Cluster Assign------");
            printf(" DataPoint { ");
            for (int j = 0; j < dim; j++)
            {
                printf("%f,", data[j]);
            }
            printf("} ");
            printf(" assigned to cluster : %d \n", new_centroids[i] + 1);

            data += dim;
        }
    }
    if (cluster_rank == 0) {
        int i, j;
        float querypoint[dim];
        //---------------random query point generator-----------------------------
        for (i = 0; i < dim; i++) {
            querypoint[i] = rand() % 100 + 1;

        }
        //---------------printing query point---------------------------------------
        printf("\n==================================================================================");
        printf("\nHello world from process %d of %d\n", cluster_rank, no_of_clusters);
        printf("\nQuerypoint { ");
        for (i = 0; i < dim; i++) {

            printf("%f, ", querypoint[i]);
        }
        printf("}\n");
        printf("\n==================================================================================");
        //-------------call for search value------------------------------------------------
        search(querypoint, data_point, dim);
        printf("\n==================================================================================\n");
    }
    MPI_Finalize();

}