package br.edu.ufrn.kmeans;

import org.openjdk.jcstress.annotations.*;
import org.openjdk.jcstress.infra.results.L_Result;

@JCStressTest
@Outcome(id = "2", expect = Expect.ACCEPTABLE, desc = "Both increments visible")
@Outcome(id = "1", expect = Expect.FORBIDDEN,  desc = "Lost update — race on shared count")
@State
public class JCS3_LocalCountAccumulationTest {

    long[] counts = new long[]{0}; // shared, naive approach

    @Actor
    public void actor1() { counts[0]++; }

    @Actor
    public void actor2() { counts[0]++; }

    @Arbiter
    public void arbiter(L_Result r) { r.r1 = counts[0]; }
}
