package br.edu.ufrn.kmeans;

import java.io.IOException;
import java.nio.file.Path;
import java.util.List;
import java.util.logging.Logger;

public final class KMeans {

    private static final Logger logger = Logger.getLogger(KMeans.class.getName());

    public static KMeansResult cluster(Path datasetPath, CsvDataset dataset, int k, int maxIterations, double tolerance) throws IOException {
        List<double[]> seeds = dataset.readInitialCentroids(datasetPath, k);

        logger.info("Initialized " + k + " centroids from the first rows of the dataset.");

        double[][] centroids = new double[k][];

        for (int i = 0; i < k; i++) {
            centroids[i] = seeds.get(i).clone();
        }

        int completedIterations = 0;

        for (int iteration = 1; iteration <= maxIterations; iteration++) {
            int currentIteration = iteration;

            logger.info("Starting iteration " + currentIteration + " of " + maxIterations + ".");

            double[][] sums = new double[k][dataset.featureCount()];
            long[] counts = new long[k];
            double[][] activeCentroids = centroids;

            dataset.forEachPoint(datasetPath, (point, rowNumber) -> {
                int closest = closestCentroid(point, activeCentroids);
                counts[closest]++;
                addPoint(sums[closest], point);
            });

            double[][] updatedCentroids = recomputeCentroids(centroids, sums, counts);
            completedIterations = iteration;

            double shift = maxShift(centroids, updatedCentroids);
            double currentShift = shift;

            logger.info("Finished iteration " + currentIteration + " with max centroid shift " + currentShift + ".");

            if (shift <= tolerance) {
                centroids = updatedCentroids;

                logger.info("Converged at iteration " + currentIteration + ".");

                break;
            }

            centroids = updatedCentroids;
        }

        logger.info("Evaluating final clusters.");

        return evaluate(datasetPath, dataset, centroids, completedIterations);
    }

    private static KMeansResult evaluate(Path datasetPath, CsvDataset dataset, double[][] centroids, int iterations) throws IOException {
        long[] clusterSizes = new long[centroids.length];
        double[] sumSquaredError = new double[1];

        dataset.forEachPoint(datasetPath, (point, rowNumber) -> {
            int closest = closestCentroid(point, centroids);
            clusterSizes[closest]++;
            sumSquaredError[0] += squaredDistance(point, centroids[closest]);
        });

        return new KMeansResult(copy(centroids), dataset.featureNames(), clusterSizes, iterations, sumSquaredError[0]);
    }

    private static double[][] recomputeCentroids(double[][] currentCentroids, double[][] sums, long[] counts) {
        double[][] updated = new double[currentCentroids.length][currentCentroids[0].length];

        for (int cluster = 0; cluster < currentCentroids.length; cluster++) {
            if (counts[cluster] == 0) {
                updated[cluster] = currentCentroids[cluster].clone();
                continue;
            }

            updated[cluster] = new double[currentCentroids[cluster].length];

            for (int dimension = 0; dimension < currentCentroids[cluster].length; dimension++) {
                updated[cluster][dimension] = sums[cluster][dimension] / counts[cluster];
            }
        }

        return updated;
    }

    private static int closestCentroid(double[] point, double[][] centroids) {
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

    private static double squaredDistance(double[] left, double[] right) {
        double total = 0.0d;

        for (int i = 0; i < left.length; i++) {
            double delta = left[i] - right[i];
            total += delta * delta;
        }

        return total;
    }

    private static void addPoint(double[] accumulator, double[] point) {
        for (int i = 0; i < accumulator.length; i++) {
            accumulator[i] += point[i];
        }
    }

    private static double maxShift(double[][] current, double[][] updated) {
        double maxShift = 0.0d;

        for (int i = 0; i < current.length; i++) {
            maxShift = Math.max(maxShift, squaredDistance(current[i], updated[i]));
        }

        return maxShift;
    }

    private static double[][] copy(double[][] centroids) {
        double[][] copy = new double[centroids.length][];

        for (int i = 0; i < centroids.length; i++) {
            copy[i] = centroids[i].clone();
        }
        
        return copy;
    }

}
