package br.edu.ufrn.kmeans;

import java.io.IOException;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.ForkJoinPool;
import java.util.concurrent.RecursiveTask;
import java.util.logging.Logger;

public final class KMeans {

    private static final Logger logger = Logger.getLogger(KMeans.class.getName());

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
        int threshold = points.length / (n * 4);
        int completedIterations = 0;

        try {
            for (int iteration = 1; iteration <= maxIterations; iteration++) {
                int currentIteration = iteration;
                logger.info("Starting iteration " + currentIteration + " of " + maxIterations + ".");

                double[][] snapshot = copy(centroids);
                PartialResult result = ForkJoinPool.commonPool()
                    .invoke(new KMeansTask(points, 0, points.length, snapshot, threshold));

                double[][] updatedCentroids = recomputeCentroids(centroids, result.sums(), result.counts());
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
            return evaluate(points, dataset, centroids, completedIterations, threshold);

        } catch (Exception e) {
            throw new IOException("Clustering failed", e);
        }
    }

    private static class KMeansTask extends RecursiveTask<PartialResult> {
        private final double[][] points;
        private final double[][] centroids;
        private final int from, to, threshold;

        KMeansTask(double[][] points, int from, int to, double[][] centroids, int threshold) {
            this.points = points;
            this.from = from;
            this.to = to;
            this.centroids = centroids;
            this.threshold = threshold;
        }

        @Override
        protected PartialResult compute() {
            if (to - from <= threshold) {
                return processChunk(points, from, to, centroids);
            }
            int mid = (from + to) / 2;
            KMeansTask left  = new KMeansTask(points, from, mid, centroids, threshold);
            KMeansTask right = new KMeansTask(points, mid,  to,  centroids, threshold);
            left.fork();
            PartialResult rightResult = right.compute();
            PartialResult leftResult  = left.join();
            return mergePartial(leftResult, rightResult);
        }
    }

    private static PartialResult processChunk(double[][] points, int from, int to, double[][] centroids) {
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

    private static PartialResult mergePartial(PartialResult a, PartialResult b) {
        int k = a.sums().length, d = a.sums()[0].length;
        double[][] sums = new double[k][d];
        long[] counts = new long[k];
        for (int c = 0; c < k; c++) {
            counts[c] = a.counts()[c] + b.counts()[c];
            for (int dim = 0; dim < d; dim++)
                sums[c][dim] = a.sums()[c][dim] + b.sums()[c][dim];
        }
        return new PartialResult(sums, counts);
    }

    private static KMeansResult evaluate(double[][] points, CsvDataset dataset, double[][] centroids, int iterations, int threshold) {
        int k = centroids.length;

        EvalResult result = ForkJoinPool.commonPool()
            .invoke(new EvalTask(points, 0, points.length, centroids, threshold, k));

        return new KMeansResult(copy(centroids), dataset.featureNames(), result.sizes(), iterations, result.sse());
    }

    private static class EvalTask extends RecursiveTask<EvalResult> {
        private final double[][] points;
        private final double[][] centroids;
        private final int from, to, threshold, k;

        EvalTask(double[][] points, int from, int to, double[][] centroids, int threshold, int k) {
            this.points = points;
            this.from = from;
            this.to = to;
            this.centroids = centroids;
            this.threshold = threshold;
            this.k = k;
        }

        @Override
        protected EvalResult compute() {
            if (to - from <= threshold) {
                return evaluateChunk(points, from, to, centroids, k);
            }
            int mid = (from + to) / 2;
            EvalTask left  = new EvalTask(points, from, mid, centroids, threshold, k);
            EvalTask right = new EvalTask(points, mid,  to,  centroids, threshold, k);
            left.fork();
            EvalResult rightResult = right.compute();
            EvalResult leftResult  = left.join();
            return mergeEval(leftResult, rightResult);
        }
    }

    private static EvalResult evaluateChunk(double[][] points, int from, int to, double[][] centroids, int k) {
        long[] sizes = new long[k];
        double sse = 0.0;

        for (int i = from; i < to; i++) {
            int c = closestCentroid(points[i], centroids);
            sizes[c]++;
            sse += squaredDistance(points[i], centroids[c]);
        }

        return new EvalResult(sizes, sse);
    }

    private static EvalResult mergeEval(EvalResult a, EvalResult b) {
        int k = a.sizes().length;
        long[] sizes = new long[k];
        for (int c = 0; c < k; c++)
            sizes[c] = a.sizes()[c] + b.sizes()[c];
        return new EvalResult(sizes, a.sse() + b.sse());
    }

    private record EvalResult(long[] sizes, double sse) {}

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
