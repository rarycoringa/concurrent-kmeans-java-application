package br.edu.ufrn.kmeans;

public record KMeansResult(
    double[][] centroids,
    String[] featureNames,
    long[] clusterSizes,
    int iterations,
    double sumSquaredError
) {}
