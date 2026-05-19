package br.edu.ufrn.kmeans;

import org.openjdk.jmh.annotations.*;

import java.io.IOException;
import java.nio.file.Path;
import java.util.Random;
import java.util.concurrent.TimeUnit;

@BenchmarkMode(Mode.AverageTime)
@OutputTimeUnit(TimeUnit.SECONDS)
@Warmup(iterations = 1, time = 10)
@Measurement(iterations = 3, time = 10)
@Fork(1)
public class KMeansBenchmark {

    @State(Scope.Thread)
    public static class DatasetState {

        @Param({"dataset.csv"})
        public String datasetPath;

        public Path path;
        public CsvDataset dataset;

        @Setup(Level.Trial)
        public void setup() throws IOException {
            path = Path.of(datasetPath);
            dataset = CsvDataset.load(path);
        }
    }

    @State(Scope.Thread)
    public static class HotspotState {

        static final int K = 10;
        static final int DIMS = 38;

        public double[] point;
        public double[][] centroids;
        public double[] left;
        public double[] right;

        @Setup(Level.Trial)
        public void setup() {
            Random rng = new Random(42);
            point = randomVector(rng, DIMS);
            centroids = new double[K][];
            for (int i = 0; i < K; i++) {
                centroids[i] = randomVector(rng, DIMS);
            }
            left = randomVector(rng, DIMS);
            right = randomVector(rng, DIMS);
        }

        private static double[] randomVector(Random rng, int dims) {
            double[] v = new double[dims];
            for (int i = 0; i < dims; i++) {
                v[i] = rng.nextDouble();
            }
            return v;
        }
    }

    // B1: primary report number — full end-to-end cluster run
    @Benchmark
    public KMeansResult b1_fullRun(DatasetState s) throws IOException {
        return KMeans.cluster(s.path, s.dataset, 10, 20, 1e-6);
    }

    // B2: isolate per-iteration cost with a short fixed run
    @Benchmark
    public KMeansResult b2_perIterationCost(DatasetState s) throws IOException {
        return KMeans.cluster(s.path, s.dataset, 10, 5, 0);
    }

    // B3: inner-loop hotspot — point-to-centroid distance scan
    @Benchmark
    public int b3_closestCentroid(HotspotState s) {
        return KMeans.closestCentroid(s.point, s.centroids);
    }

    // B4: arithmetic baseline — squared Euclidean distance
    @Benchmark
    public double b4_squaredDistance(HotspotState s) {
        return KMeans.squaredDistance(s.left, s.right);
    }
}
