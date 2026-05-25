package br.edu.ufrn.kmeans;

import org.openjdk.jcstress.annotations.*;
import org.openjdk.jcstress.infra.results.DD_Result;

@JCStressTest
@Outcome(id = "1.0, 1.0", expect = Expect.ACCEPTABLE,
        desc = "Both actors computed the correct squared distance")
@Outcome(expect = Expect.FORBIDDEN,
        desc = "Wrong distance - torn read of shared arrays")
@State
public class JCS2_SquaredDistanceTest {

    final double[] point    = {1.0, 0.0};
    final double[] centroid = {0.0, 0.0};

    @Actor
    public void actor1(DD_Result r) {
        r.r1 = KMeans.squaredDistance(point, centroid);
    }

    @Actor
    public void actor2(DD_Result r) {
        r.r2 = KMeans.squaredDistance(point, centroid);
    }
}
