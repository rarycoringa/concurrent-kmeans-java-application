package br.edu.ufrn.kmeans;

import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.broadcast.Broadcast;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
import scala.Tuple2;

import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.logging.Logger;

public final class KMeans {

    private static final Logger logger = Logger.getLogger(KMeans.class.getName());

    private static final String[] NON_FEATURE_COLUMNS = {"order_id", "customer_id", "seller_id", "city"};

    public static KMeansResult cluster(String datasetPath, int k, int maxIterations, double tolerance) {
        SparkSession spark = SparkSession.builder()
                .appName("KMeans v6")
                .master("local[*]")
                .config("spark.driver.memory", "4g")
                .config("spark.sql.shuffle.partitions", "16")
                .getOrCreate();

        // Load CSV, drop non-feature columns, and cache — the DataFrame stays in
        // memory for the lifetime of all iterations, same as the double[][] in v1-v5
        Dataset<Row> df = spark.read()
                .format("csv")
                .option("header", "true")
                .option("inferSchema", "true")
                .load(datasetPath)
                .drop(NON_FEATURE_COLUMNS)
                .cache();

        String[] featureNames = df.columns();
        int d = featureNames.length;

        logger.info("Dataset loaded: " + df.count() + " points, " + d + " features.");

        // Convert to JavaRDD<double[]> and cache — reused every iteration
        JavaRDD<double[]> points = df.javaRDD()
                .map(row -> toDoubleArray(row, d))
                .cache();
        points.count(); // materialise cache before timing starts

        // Seed centroids from the first k points (same strategy as all prior versions)
        List<double[]> seeds = points.take(k);
        double[][] centroids = new double[k][];
        for (int i = 0; i < k; i++) centroids[i] = seeds.get(i).clone();

        JavaSparkContext sc = JavaSparkContext.fromSparkContext(spark.sparkContext());
        int completedIterations = 0;

        for (int iter = 1; iter <= maxIterations; iter++) {
            final double[][] snapshot = copy(centroids);
            // Broadcast centroid snapshot to all partitions — same role as the
            // defensive copy() in v3/v4/v5: prevents reading partially-updated state
            Broadcast<double[][]> bcCentroids = sc.broadcast(snapshot);
            final int dims = d;

            // Each point becomes (cluster, [f0..fd-1, count=1.0]).
            // reduceByKey sums partition-locally then merges across partitions —
            // the same local-accumulation pattern as processChunk in v3/v4/v5
            Map<Integer, double[]> clusterSums = points
                    .mapToPair(point -> {
                        int cluster = closestCentroid(point, bcCentroids.value());
                        double[] acc = Arrays.copyOf(point, dims + 1);
                        acc[dims] = 1.0; // count in last element
                        return new Tuple2<>(cluster, acc);
                    })
                    .reduceByKey((a, b) -> {
                        double[] merged = new double[dims + 1];
                        for (int i = 0; i <= dims; i++) merged[i] = a[i] + b[i];
                        return merged;
                    })
                    .collectAsMap();

            bcCentroids.destroy();

            double[][] newCentroids = copy(centroids); // preserve empty clusters unchanged
            for (Map.Entry<Integer, double[]> e : clusterSums.entrySet()) {
                double[] sumCount = e.getValue();
                double count = sumCount[dims];
                double[] mean = new double[dims];
                for (int j = 0; j < dims; j++) mean[j] = sumCount[j] / count;
                newCentroids[e.getKey()] = mean;
            }

            completedIterations = iter;
            double shift = maxShift(centroids, newCentroids);
            centroids = newCentroids;
            logger.info("Iteration " + iter + ": shift=" + shift);

            if (shift <= tolerance) {
                logger.info("Converged at iteration " + iter + ".");
                break;
            }
        }

        // Evaluation pass: compute cluster sizes and SSE in one final RDD scan
        final double[][] finalCentroids = centroids;
        Map<Integer, double[]> evalMap = points
                .mapToPair(point -> {
                    int cluster = closestCentroid(point, finalCentroids);
                    double dist = squaredDistance(point, finalCentroids[cluster]);
                    return new Tuple2<>(cluster, new double[]{1.0, dist});
                })
                .reduceByKey((a, b) -> new double[]{a[0] + b[0], a[1] + b[1]})
                .collectAsMap();

        long[] clusterSizes = new long[k];
        double sse = 0;
        for (Map.Entry<Integer, double[]> e : evalMap.entrySet()) {
            clusterSizes[e.getKey()] = (long) e.getValue()[0];
            sse += e.getValue()[1];
        }

        spark.stop();
        return new KMeansResult(centroids, featureNames, clusterSizes, completedIterations, sse);
    }

    private static double[] toDoubleArray(Row row, int d) {
        double[] point = new double[d];
        for (int i = 0; i < d; i++) {
            Object val = row.get(i);
            point[i] = val instanceof Number ? ((Number) val).doubleValue()
                                             : Double.parseDouble(val.toString());
        }
        return point;
    }

    private static int closestCentroid(double[] point, double[][] centroids) {
        int best = 0;
        double bestDist = squaredDistance(point, centroids[0]);
        for (int i = 1; i < centroids.length; i++) {
            double dist = squaredDistance(point, centroids[i]);
            if (dist < bestDist) { bestDist = dist; best = i; }
        }
        return best;
    }

    private static double squaredDistance(double[] a, double[] b) {
        double total = 0;
        for (int i = 0; i < a.length; i++) {
            double diff = a[i] - b[i];
            total += diff * diff;
        }
        return total;
    }

    private static double[][] copy(double[][] arr) {
        double[][] copy = new double[arr.length][];
        for (int i = 0; i < arr.length; i++) copy[i] = arr[i].clone();
        return copy;
    }

    private static double maxShift(double[][] a, double[][] b) {
        double max = 0;
        for (int i = 0; i < a.length; i++) max = Math.max(max, squaredDistance(a[i], b[i]));
        return max;
    }
}
