package br.edu.ufrn.kmeans;

import java.io.IOException;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.logging.Logger;

public final class KMeans {

    private static final Logger logger = Logger.getLogger(KMeans.class.getName());
    private static final int BATCH_SIZE = 5_000;

    private record PartialResult(double[][] sums, long[] counts) {}
    private record EvaluatePartial(long[] clusterSizes, double sumSquaredError) {}

    public static KMeansResult cluster(Path datasetPath, CsvDataset dataset, int k, int maxIterations, double tolerance) throws IOException {
        List<double[]> seeds = dataset.readInitialCentroids(datasetPath, k);
        logger.info("Initialized " + k + " centroids from the first rows of the dataset.");

        double[][] centroids = new double[k][];
        for (int i = 0; i < k; i++) centroids[i] = seeds.get(i).clone();

        int n = Runtime.getRuntime().availableProcessors();
        // Platform threads: CPU-bound compute workers (closestCentroid / squaredDistance)
        ExecutorService computePool = Executors.newFixedThreadPool(n, Thread.ofPlatform().factory());
        // Virtual thread: I/O-bound coordinator (readLine + Future.get() both unmount)
        ExecutorService coordinator = Executors.newFixedThreadPool(n, Thread.ofVirtual().factory());

        try {
            return coordinator.submit(
                () -> runClustering(datasetPath, dataset, k, maxIterations, tolerance, centroids, computePool)
            ).get();
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            throw new IOException("Clustering interrupted", e);
        } catch (ExecutionException e) {
            Throwable cause = e.getCause();
            if (cause instanceof IOException ioe) throw ioe;
            throw new IOException("Clustering failed", cause);
        } finally {
            coordinator.shutdown();
            computePool.shutdown();
        }
    }

    private static KMeansResult runClustering(Path datasetPath, CsvDataset dataset, int k, int maxIterations, double tolerance, double[][] centroids, ExecutorService computePool) throws IOException, InterruptedException, ExecutionException {
        int completedIterations = 0;

        for (int iteration = 1; iteration <= maxIterations; iteration++) {
            int currentIteration = iteration;
            logger.info("Starting iteration " + currentIteration + " of " + maxIterations + ".");

            double[][] snapshot = copy(centroids);
            List<Future<PartialResult>> futures = new ArrayList<>();
            List<double[]> batch = new ArrayList<>(BATCH_SIZE);

            // readLine() inside forEachPoint blocks on disk I/O — virtual thread unmounts here
            dataset.forEachPoint(datasetPath, (point, rowNumber) -> {
                batch.add(point);
                if (batch.size() == BATCH_SIZE) {
                    List<double[]> current = List.copyOf(batch);
                    futures.add(computePool.submit(() -> processAssignBatch(current, snapshot)));
                    batch.clear();
                }
            });
            if (!batch.isEmpty()) {
                List<double[]> last = List.copyOf(batch);
                futures.add(computePool.submit(() -> processAssignBatch(last, snapshot)));
            }

            double[][] sums = new double[k][dataset.featureCount()];
            long[] counts = new long[k];
            for (Future<PartialResult> f : futures) {
                // Future.get() parks via LockSupport — virtual thread unmounts here
                PartialResult pr = f.get();
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
        return evaluate(datasetPath, dataset, centroids, completedIterations, computePool);
    }

    private static PartialResult processAssignBatch(List<double[]> batch, double[][] centroids) {
        int k = centroids.length, d = centroids[0].length;
        double[][] localSums = new double[k][d];
        long[] localCounts = new long[k];

        for (double[] point : batch) {
            int c = closestCentroid(point, centroids);
            localCounts[c]++;
            addPoint(localSums[c], point);
        }

        return new PartialResult(localSums, localCounts);
    }

    private static KMeansResult evaluate(Path datasetPath, CsvDataset dataset, double[][] centroids, int iterations, ExecutorService executor) throws IOException, InterruptedException, ExecutionException {
        int k = centroids.length;
        List<Future<EvaluatePartial>> futures = new ArrayList<>();
        List<double[]> batch = new ArrayList<>(BATCH_SIZE);

        dataset.forEachPoint(datasetPath, (point, rowNumber) -> {
            batch.add(point);
            if (batch.size() == BATCH_SIZE) {
                List<double[]> current = List.copyOf(batch);
                futures.add(executor.submit(() -> processEvaluateBatch(current, centroids)));
                batch.clear();
            }
        });
        if (!batch.isEmpty()) {
            List<double[]> last = List.copyOf(batch);
            futures.add(executor.submit(() -> processEvaluateBatch(last, centroids)));
        }

        long[] clusterSizes = new long[k];
        double sumSquaredError = 0.0;
        for (Future<EvaluatePartial> f : futures) {
            EvaluatePartial ep = f.get();
            for (int c = 0; c < k; c++)
                clusterSizes[c] += ep.clusterSizes()[c];
            sumSquaredError += ep.sumSquaredError();
        }

        return new KMeansResult(copy(centroids), dataset.featureNames(), clusterSizes, iterations, sumSquaredError);
    }

    private static EvaluatePartial processEvaluateBatch(List<double[]> batch, double[][] centroids) {
        int k = centroids.length;
        long[] clusterSizes = new long[k];
        double sumSquaredError = 0.0;

        for (double[] point : batch) {
            int c = closestCentroid(point, centroids);
            clusterSizes[c]++;
            sumSquaredError += squaredDistance(point, centroids[c]);
        }

        return new EvaluatePartial(clusterSizes, sumSquaredError);
    }

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
