package br.edu.ufrn.kmeans;

import org.apache.jmeter.config.Arguments;
import org.apache.jmeter.protocol.java.sampler.AbstractJavaSamplerClient;
import org.apache.jmeter.protocol.java.sampler.JavaSamplerContext;
import org.apache.jmeter.samplers.SampleResult;

import java.io.IOException;
import java.io.Serializable;
import java.nio.file.Path;

public class KMeansSampler extends AbstractJavaSamplerClient implements Serializable {

    private static final long serialVersionUID = 1L;

    private static volatile double[][] sharedPoints;
    private static volatile CsvDataset sharedDataset;
    private static final Object LOCK = new Object();

    @Override
    public Arguments getDefaultParameters() {
        Arguments args = new Arguments();
        args.addArgument("dataset", "dataset.csv");
        args.addArgument("k", "10");
        args.addArgument("maxIter", "3");
        args.addArgument("tol", "0.0");
        return args;
    }

    private static void ensureLoaded(String datasetPath) throws IOException {
        if (sharedPoints == null) {
            synchronized (LOCK) {
                if (sharedPoints == null) {
                    Path path = Path.of(datasetPath);
                    CsvDataset csv = CsvDataset.load(path);
                    double[][] pts = csv.loadAllPoints(path);
                    sharedDataset = csv;
                    sharedPoints = pts;
                }
            }
        }
    }

    @Override
    public SampleResult runTest(JavaSamplerContext context) {
        String datasetPath = context.getParameter("dataset");
        int k         = Integer.parseInt(context.getParameter("k"));
        int maxIter   = Integer.parseInt(context.getParameter("maxIter"));
        double tol    = Double.parseDouble(context.getParameter("tol"));

        SampleResult result = new SampleResult();
        result.setSampleLabel("KMeans k=" + k + " maxIter=" + maxIter);
        result.sampleStart();

        try {
            ensureLoaded(datasetPath);
            KMeansResult kmResult = KMeans.cluster(sharedPoints, sharedDataset, k, maxIter, tol);

            result.sampleEnd();
            result.setResponseCode("200");
            result.setResponseMessage("OK");
            result.setSuccessful(true);
            result.setResponseData(
                String.format("iterations=%d sse=%.6f", kmResult.iterations(), kmResult.sumSquaredError()),
                "UTF-8"
            );
        } catch (Exception e) {
            result.sampleEnd();
            result.setResponseCode("500");
            result.setResponseMessage(e.getMessage() != null ? e.getMessage() : e.getClass().getSimpleName());
            result.setSuccessful(false);
        }

        return result;
    }
}
