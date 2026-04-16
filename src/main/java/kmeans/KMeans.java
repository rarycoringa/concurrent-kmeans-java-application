package kmeans;

import java.io.BufferedReader;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class KMeans {
    private int k;
    private double[][] centroids;

    private static final String datasetFilePath = "data/consumption.csv";
    private static final String featureNamesFilePath = "data/features.csv";
    private static final String featureMinMaxFilePath = "data/mins_and_maxs.csv";

    private List<String> featuresToIgnore = List.of("sensor_id", "city");
    private List<String> featureNames = new ArrayList<>();
    private Map<String, Double> featureMinValues = new HashMap<>();
    private Map<String, Double> featureMaxValues = new HashMap<>();

    public KMeans(int k) {
        this.k = k;

        loadFeatureNames();
        loadMinMaxValues();
    }

    private void loadFeatureNames() {
        this.featureNames.clear();

        try (BufferedReader reader = Files.newBufferedReader(Path.of(featureNamesFilePath))) {
            String line;
            int lineNumber = 0;

            while ((line = reader.readLine()) != null) {
                lineNumber++;

                line = line.trim();

                if (lineNumber == 1 || line.isEmpty() || line.isBlank()) {
                    continue;
                }

                this.featureNames.add(line);
            }
        } catch (IOException e) {
            throw new RuntimeException("Failed to read CSV file: " + featureNamesFilePath, e);
        }
    }

    private void loadMinMaxValues() {
        this.featureMinValues.clear();
        this.featureMaxValues.clear();

        this.featureMinValues.clear();
        this.featureMaxValues.clear();

        try (BufferedReader reader = Files.newBufferedReader(Path.of(featureMinMaxFilePath))) {
            String line;
            int lineNumber = 0;

            while ((line = reader.readLine()) != null) {
                lineNumber++;

                line = line.trim();

                if (lineNumber == 1 || line.isEmpty() || line.isBlank()) {
                    continue;
                }

                String[] parts = line.split(",");

                if (parts.length < 3) {
                    throw new IllegalArgumentException(
                        "Invalid CSV at line " + lineNumber + ": expected format column,min,max"
                    );
                }

                String column = parts[0].trim();
                String minRaw = parts[1].trim();
                String maxRaw = parts[2].trim();

                double min;
                double max;

                try {
                    min = Double.parseDouble(minRaw);
                    max = Double.parseDouble(maxRaw);
                } catch (NumberFormatException e) {
                    continue;
                }

                this.featureMinValues.put(column, min);
                this.featureMaxValues.put(column, max);
            }
        } catch (IOException e) {
            throw new RuntimeException("Failed to read CSV file: " + featureMinMaxFilePath, e);
        }
    }

    private double normalize(String feature, double value) {
        Double min = featureMinValues.get(feature);
        Double max = featureMaxValues.get(feature);

        if (min == null || max == null) {
            throw new IllegalArgumentException("Feature not found: " + feature);
        }

        if (max.equals(min)) {
            return 0.0;
        }

        return (value - min) / (max - min);
    }

    public void initializeCentroids() {
        int numFeatures = 0;

        for (String feature : featureNames) {
            if (!featuresToIgnore.contains(feature)) {
                numFeatures++;
            }
        }

        this.centroids = new double[k][numFeatures];

        try (BufferedReader reader = Files.newBufferedReader(Path.of(datasetFilePath))) {
            String line;
            int lineNumber = 0;
            int centroidIndex = 0;

            while ((line = reader.readLine()) != null && centroidIndex < k) {
                lineNumber++;

                line = line.trim();

                if (lineNumber == 1 || line.isEmpty() || line.isBlank()) {
                    continue;
                }

                String[] parts = line.split(",");

                if (parts.length != featureNames.size()) {
                    continue;
                }

                int featureIndex = 0;

                for (int i = 0; i < featureNames.size(); i++) {
                    String featureName = featureNames.get(i);

                    if (featuresToIgnore.contains(featureName)) {
                        continue;
                    }

                    try {
                        double value = Double.parseDouble(parts[i].trim());
                        double normalizedValue = normalize(featureName, value);
                        centroids[centroidIndex][featureIndex] = normalizedValue;
                        featureIndex++;
                    } catch (NumberFormatException e) {
                        continue;
                    }
                }

                centroidIndex++;
            }

            if (centroidIndex < k) {
                throw new RuntimeException("Failed to initialize " + k + " centroids. Only initialized " + centroidIndex);
            }
        } catch (IOException e) {
            throw new RuntimeException("Failed to read CSV file: " + datasetFilePath, e);
        }
    }
}
