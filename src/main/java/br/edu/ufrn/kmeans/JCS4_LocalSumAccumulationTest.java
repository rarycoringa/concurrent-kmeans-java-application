package br.edu.ufrn.kmeans;

import org.openjdk.jcstress.annotations.*;
import org.openjdk.jcstress.infra.results.D_Result;

@JCStressTest
@Outcome(id = "2.0", expect = Expect.ACCEPTABLE, desc = "Both additions visible")
@Outcome(expect = Expect.FORBIDDEN,              desc = "Lost update - race on shared sum")
@State
public class JCS4_LocalSumAccumulationTest {

    double[] sums = new double[]{0.0}; // shared, naive approach

    @Actor
    public void actor1() { sums[0] += 1.0; }

    @Actor
    public void actor2() { sums[0] += 1.0; }

    @Arbiter
    public void arbiter(D_Result r) { r.r1 = sums[0]; }
}
