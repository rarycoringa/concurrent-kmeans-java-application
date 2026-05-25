clean:
    mvn --quiet clean

build:
    mvn --quiet compile

run dataset_path:
    mvn --quiet exec:java -Dexec.args="{{dataset_path}}"


bench-micro-build:
    mvn --quiet package -DskipTests

bench-micro dataset_path filter="KMeansBenchmark":
    mkdir -p results
    java -jar target/kmeans-1.0-SNAPSHOT-benchmarks.jar \
        -p datasetPath={{dataset_path}} \
        -rf json \
        -rff results/jmh-results.json \
        {{filter}}

bench-macro-deploy:
    mvn --quiet package -DskipTests
    cp target/kmeans-1.0-SNAPSHOT.jar /home/rarycoringa/university/concurrent/jmeter/lib/ext/

bench-macro-g1:
    GC_ALGO="-XX:+UseG1GC" /home/rarycoringa/university/concurrent/jmeter/bin/jmeter

bench-macro-zgc:
    GC_ALGO="-XX:+UseZGC" /home/rarycoringa/university/concurrent/jmeter/bin/jmeter

bench-macro-parallel:
    GC_ALGO="-XX:+UseParallelGC" /home/rarycoringa/university/concurrent/jmeter/bin/jmeter

profile dataset_path:
    mkdir -p results/profile
    mvn --quiet exec:exec \
        -Dexec.executable=java \
        -Dexec.args="-XX:StartFlightRecording=name=concurrent,settings=profile,dumponexit=true,filename=results/profile/concurrent.jfr -cp %classpath br.edu.ufrn.kmeans.Main {{dataset_path}}"

profile-report:
    jfr print --events jdk.ExecutionSample \
        results/profile/concurrent.jfr > results/profile/hot-methods.txt
    jfr print --events jdk.GarbageCollection,jdk.GCHeapSummary \
        results/profile/concurrent.jfr > results/profile/gc.txt
    jfr print --events jdk.JavaThreadStatistics,jdk.ThreadCPULoad \
        results/profile/concurrent.jfr > results/profile/threads.txt
    jfr print --events jdk.FileRead \
        results/profile/concurrent.jfr > results/profile/io.txt
