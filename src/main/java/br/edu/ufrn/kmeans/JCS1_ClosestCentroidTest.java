package br.edu.ufrn.kmeans;

import org.openjdk.jcstress.annotations.*;
import org.openjdk.jcstress.infra.results.II_Result;

@JCStressTest
@Outcome(id = "0, 1", expect = Expect.ACCEPTABLE,
        desc = "Both actors found their correct nearest centroid")
@Outcome(expect = Expect.FORBIDDEN,
        desc = "Wrong centroid - torn read of shared snapshot")
@State
public class JCS1_ClosestCentroidTest {

    final double[][] centroids = {{1.0, 0.0}, {0.0, 1.0}};

    @Actor
    public void actor1(II_Result r) {
        r.r1 = KMeans.closestCentroid(new double[]{0.9, 0.1}, centroids);
    }

    @Actor
    public void actor2(II_Result r) {
        r.r2 = KMeans.closestCentroid(new double[]{0.1, 0.9}, centroids);
    }
}
