package br.edu.ufrn.kmeans;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Locale;
import java.util.logging.Logger;

public final class Main {

    private static final int K = 10;
    private static final int MAX_ITERATIONS = 20;
    private static final double TOLERANCE = 1.0e-6;
    private static final Logger logger = Logger.getLogger(Main.class.getName());

    public static void main(String[] args) throws IOException {
        if (args.length != 1) {
            logger.severe("Usage: mvn exec:java -Dexec.args=\"<dataset-path>\"");
            System.exit(1);
        }

        Path datasetPath = Path.of(args[0]).normalize();

        if (!Files.isRegularFile(datasetPath)) {
            throw new IllegalArgumentException("Dataset file does not exist: " + datasetPath);
        }

        logger.info("Starting K-Means baseline with dataset: " + datasetPath);

        CsvDataset dataset = CsvDataset.load(datasetPath);

        logger.info("Loaded dataset header with " + dataset.featureCount() + " feature columns.");

        KMeansResult result = KMeans.cluster(datasetPath, dataset, K, MAX_ITERATIONS, TOLERANCE);
        
        logger.info("K-Means finished after " + result.iterations() + " iterations.");

        logSummary(datasetPath, result);
    }

    private static void logSummary(Path datasetPath, KMeansResult result) {
        logger.info("K-Means baseline");
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
