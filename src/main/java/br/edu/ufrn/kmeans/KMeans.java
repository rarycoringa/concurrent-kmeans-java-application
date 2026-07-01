package br.edu.ufrn.kmeans;

import java.io.IOException;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.StructuredTaskScope;
import java.util.logging.Logger;

public final class KMeans {

    private static final Logger logger = Logger.getLogger(KMeans.class.getName());
    private static final ScopedValue<double[][]> CENTROIDS = ScopedValue.newInstance();

    // Path-based entry point: loads points from disk, then delegates
    public static KMeansResult cluster(Path datasetPath, CsvDataset dataset, int k, int maxIterations, double tolerance) throws IOException {
        logger.info("Loading dataset into memory.");
        double[][] points = dataset.loadAllPoints(datasetPath);
        List<double[]> seeds = dataset.readInitialCentroids(datasetPath, k);
        return cluster(points, seeds, dataset, k, maxIterations, tolerance);
    }

    // Points-based entry point: used by benchmarks (points already in memory)
    public static KMeansResult cluster(double[][] points, CsvDataset dataset, int k, int maxIterations, double tolerance) throws IOException {
        List<double[]> seeds = new ArrayList<>(k);
        for (int i = 0; i < k; i++) seeds.add(points[i].clone());
        return cluster(points, seeds, dataset, k, maxIterations, tolerance);
    }

    private static KMeansResult cluster(double[][] points, List<double[]> seeds, CsvDataset dataset, int k, int maxIterations, double tolerance) throws IOException {
        logger.info("Initialized " + k + " centroids from the first rows of the dataset.");

        double[][] centroids = new double[k][];
        for (int i = 0; i < k; i++) centroids[i] = seeds.get(i).clone();

        int n = Runtime.getRuntime().availableProcessors();
        int completedIterations = 0;

        try {
            int[] boundaries = chunkBoundaries(points.length, n);

            for (int iteration = 1; iteration <= maxIterations; iteration++) {
                int currentIteration = iteration;
                logger.info("Starting iteration " + currentIteration + " of " + maxIterations + ".");

                double[][] snapshot = copy(centroids);
                List<PartialResult> results = ScopedValue.where(CENTROIDS, snapshot).call(() -> {
                    try (var scope = StructuredTaskScope.open()) {
                        List<StructuredTaskScope.Subtask<PartialResult>> subtasks = new ArrayList<>(n);
                        for (int t = 0; t < n; t++) {
                            final int from = boundaries[t];
                            final int to   = boundaries[t + 1];
                            subtasks.add(scope.fork(() -> processChunk(points, from, to)));
                        }
                        scope.join();
                        return subtasks.stream().map(StructuredTaskScope.Subtask::get).toList();
                    }
                });

                double[][] sums = new double[k][dataset.featureCount()];
                long[] counts = new long[k];
                for (PartialResult pr : results) {
                    for (int c = 0; c < k; c++) {
                        counts[c] += pr.counts()[c];
                        for (int d = 0; d < sums[c].length; d++)
                            sums[c][d] += pr.sums()[c][d];
                    }
                }

                double[][] updatedCentroids = recomputeCentroids(centroids, sums, counts);
                completedIterations = iteration;

                double shift = maxShift(centroids, updatedCentroids);
                logger.info("Finished iteration " + currentIteration + " with max centroid shift " + shift + ".");

                centroids = updatedCentroids;

                if (shift <= tolerance) {
                    logger.info("Converged at iteration " + currentIteration + ".");
                    break;
                }
            }

            logger.info("Evaluating final clusters.");
            return evaluate(points, dataset, centroids, completedIterations, n);

        } catch (Exception e) {
            throw new IOException("Clustering failed", e);
        }
    }

    private static PartialResult processChunk(double[][] points, int from, int to) {
        double[][] centroids = CENTROIDS.get();
        int k = centroids.length, d = centroids[0].length;
        double[][] localSums = new double[k][d];
        long[] localCounts = new long[k];

        for (int i = from; i < to; i++) {
            int c = closestCentroid(points[i], centroids);
            localCounts[c]++;
            addPoint(localSums[c], points[i]);
        }

        return new PartialResult(localSums, localCounts);
    }

    private static KMeansResult evaluate(double[][] points, CsvDataset dataset, double[][] centroids, int iterations, int n) throws InterruptedException {
        int k = centroids.length;
        int[] boundaries = chunkBoundaries(points.length, n);

        long[] clusterSizes = new long[k];
        double sse = 0.0;

        List<EvalResult> results = ScopedValue.where(CENTROIDS, centroids).call(() -> {
            try (var scope = StructuredTaskScope.open()) {
                List<StructuredTaskScope.Subtask<EvalResult>> subtasks = new ArrayList<>(n);
                for (int t = 0; t < n; t++) {
                    final int from = boundaries[t];
                    final int to   = boundaries[t + 1];
                    subtasks.add(scope.fork(() -> evaluateChunk(points, from, to, k)));
                }
                scope.join();
                return subtasks.stream().map(StructuredTaskScope.Subtask::get).toList();
            }
        });

        for (EvalResult result : results) {
            sse += result.sse();
            for (int c = 0; c < k; c++)
                clusterSizes[c] += result.sizes()[c];
        }

        return new KMeansResult(copy(centroids), dataset.featureNames(), clusterSizes, iterations, sse);
    }

    private static EvalResult evaluateChunk(double[][] points, int from, int to, int k) {
        double[][] centroids = CENTROIDS.get();
        long[] sizes = new long[k];
        double sse = 0.0;

        for (int i = from; i < to; i++) {
            int c = closestCentroid(points[i], centroids);
            sizes[c]++;
            sse += squaredDistance(points[i], centroids[c]);
        }

        return new EvalResult(sizes, sse);
    }

    private record EvalResult(long[] sizes, double sse) {}

    private static int[] chunkBoundaries(int total, int n) {
        int[] b = new int[n + 1];
        int base = total / n, rem = total % n;
        b[0] = 0;
        for (int i = 0; i < n; i++)
            b[i + 1] = b[i] + base + (i < rem ? 1 : 0);
        return b;
    }

    private record PartialResult(double[][] sums, long[] counts) {}

    private static double[][] recomputeCentroids(double[][] currentCentroids, double[][] sums, long[] counts) {
        double[][] updated = new double[currentCentroids.length][currentCentroids[0].length];

        for (int cluster = 0; cluster < currentCentroids.length; cluster++) {
            if (counts[cluster] == 0) {
                updated[cluster] = currentCentroids[cluster].clone();
                continue;
            }

            updated[cluster] = new double[currentCentroids[cluster].length];
            for (int dimension = 0; dimension < currentCentroids[cluster].length; dimension++)
                updated[cluster][dimension] = sums[cluster][dimension] / counts[cluster];
        }

        return updated;
    }

    static int closestCentroid(double[] point, double[][] centroids) {
        int bestIndex = 0;
        double bestDistance = squaredDistance(point, centroids[0]);

        for (int i = 1; i < centroids.length; i++) {
            double distance = squaredDistance(point, centroids[i]);
            if (distance < bestDistance) {
                bestDistance = distance;
                bestIndex = i;
            }
        }

        return bestIndex;
    }

    static double squaredDistance(double[] left, double[] right) {
        double total = 0.0d;
        for (int i = 0; i < left.length; i++) {
            double delta = left[i] - right[i];
            total += delta * delta;
        }
        return total;
    }

    private static void addPoint(double[] accumulator, double[] point) {
        for (int i = 0; i < accumulator.length; i++)
            accumulator[i] += point[i];
    }

    private static double maxShift(double[][] current, double[][] updated) {
        double maxShift = 0.0d;
        for (int i = 0; i < current.length; i++)
            maxShift = Math.max(maxShift, squaredDistance(current[i], updated[i]));
        return maxShift;
    }

    private static double[][] copy(double[][] centroids) {
        double[][] copy = new double[centroids.length][];
        for (int i = 0; i < centroids.length; i++)
            copy[i] = centroids[i].clone();
        return copy;
    }
}
