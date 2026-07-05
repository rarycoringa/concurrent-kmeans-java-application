package br.edu.ufrn.kmeans;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Locale;
import java.util.logging.Logger;

public final class Main {

    private static final int K = 10;
    private static final int MAX_ITERATIONS = 10;
    private static final double TOLERANCE = 1.0e-6;
    private static final Logger logger = Logger.getLogger(Main.class.getName());

    public static void main(String[] args) {
        if (args.length != 1) {
            logger.severe("Usage: mvn exec:exec -DdatasetPath=\"<dataset-path>\"");
            System.exit(1);
        }

        Path datasetPath = Path.of(args[0]).normalize();

        if (!Files.isRegularFile(datasetPath)) {
            throw new IllegalArgumentException("Dataset file does not exist: " + datasetPath);
        }

        logger.info("Starting K-Means v6 (Spark) with dataset: " + datasetPath);

        long startNs = System.nanoTime();
        KMeansResult result = KMeans.cluster(datasetPath.toString(), K, MAX_ITERATIONS, TOLERANCE);
        long elapsedMs = (System.nanoTime() - startNs) / 1_000_000;

        logger.info("K-Means finished after " + result.iterations() + " iterations in " + elapsedMs + " ms.");

        logSummary(datasetPath, result);
    }

    private static void logSummary(Path datasetPath, KMeansResult result) {
        logger.info("K-Means v6 (Spark)");
        logger.info("dataset=" + datasetPath);
        logger.info("k=" + result.centroids().length);
        logger.info("features=" + String.join(", ", result.featureNames()));
        logger.info("iterations=" + result.iterations());
        logger.info(String.format(Locale.US, "sum_squared_error=%.6f", result.sumSquaredError()));
        logger.info("clusters:");

        for (int cluster = 0; cluster < result.centroids().length; cluster++) {
            logger.info(String.format(
                Locale.US,
                "  [%d] size=%d centroid=%s",
                cluster,
                result.clusterSizes()[cluster],
                formatCentroid(result.centroids()[cluster])
            ));
        }
    }

    private static String formatCentroid(double[] centroid) {
        StringBuilder builder = new StringBuilder("[");

        for (int i = 0; i < centroid.length; i++) {
            if (i > 0) {
                builder.append(", ");
            }
            
            builder.append(String.format(Locale.US, "%.6f", centroid[i]));
        }

        builder.append(']');

        return builder.toString();
    }

}
