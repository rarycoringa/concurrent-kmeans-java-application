package br.edu.ufrn.kmeans;

import java.io.BufferedReader;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Set;

public final class CsvDataset {

    private static final Set<String> NON_FEATURE_COLUMNS = Set.of(
            "order_id",
            "customer_id",
            "seller_id",
            "city"
    );

    private final String[] headers;
    private final int[] featureIndexes;
    private final String[] featureNames;

    private CsvDataset(String[] headers, int[] featureIndexes, String[] featureNames) {
        this.headers = headers;
        this.featureIndexes = featureIndexes;
        this.featureNames = featureNames;
    }

    public static CsvDataset fromHeader(String headerLine) {
        String[] headers = headerLine.split(",", -1);
        List<Integer> indexes = new ArrayList<>();
        List<String> names = new ArrayList<>();

        for (int i = 0; i < headers.length; i++) {
            String header = headers[i].trim();
            if (!NON_FEATURE_COLUMNS.contains(header)) {
                indexes.add(i);
                names.add(header);
            }
        }

        if (indexes.isEmpty()) {
            throw new IllegalArgumentException("No numeric feature columns were found in the dataset header.");
        }

        return new CsvDataset(
            headers,
            indexes.stream().mapToInt(Integer::intValue).toArray(),
            names.toArray(String[]::new)
        );
    }

    public static CsvDataset load(Path path) throws IOException {
        try (BufferedReader reader = Files.newBufferedReader(path)) {
            String headerLine = reader.readLine();

            if (headerLine == null || headerLine.isBlank()) {
                throw new IllegalArgumentException("CSV file is empty.");
            }

            return fromHeader(headerLine);
        }
    }

    public int featureCount() {
        return featureIndexes.length;
    }

    public String[] featureNames() {
        return featureNames.clone();
    }

    public List<double[]> readInitialCentroids(Path path, int k) throws IOException {
        List<double[]> centroids = new ArrayList<>(k);

        try (BufferedReader reader = Files.newBufferedReader(path)) {
            reader.readLine();

            String line;

            while (centroids.size() < k && (line = reader.readLine()) != null) {
                if (!line.isBlank()) {
                    centroids.add(parse(line));
                }
            }
        }

        if (centroids.size() < k) {
            throw new IllegalArgumentException("Dataset does not contain enough rows to initialize " + k + " centroids.");
        }

        return centroids;
    }

    public void forEachPoint(Path path, PointConsumer consumer) throws IOException {
        try (BufferedReader reader = Files.newBufferedReader(path)) {
            reader.readLine();

            String line;
            long rowNumber = 1;

            while ((line = reader.readLine()) != null) {
                rowNumber++;

                if (line.isBlank()) {
                    continue;
                }

                consumer.accept(parse(line), rowNumber);
            }
        }
    }

    private double[] parse(String line) {
        String[] values = line.split(",", -1);

        if (values.length != headers.length) {
            throw new IllegalArgumentException(
                    "Invalid CSV row. Expected " + headers.length + " columns but found " + values.length + ". Row: " + line
            );
        }

        double[] point = new double[featureIndexes.length];

        for (int i = 0; i < featureIndexes.length; i++) {
            int index = featureIndexes[i];
            point[i] = Double.parseDouble(values[index]);
        }
        
        return point;
    }

    @FunctionalInterface
    public interface PointConsumer {
        void accept(double[] point, long rowNumber) throws IOException;
    }

}
