clean:
    mvn --quiet clean

build:
    mvn --quiet compile

run dataset_path:
    MAVEN_OPTS="--enable-preview" mvn --quiet exec:java -Dexec.args="{{dataset_path}}"


bench-micro-build:
    mvn --quiet package -DskipTests

bench-micro dataset_path version="v4" filter="KMeansBenchmark":
    mkdir -p results/micro
    java --enable-preview -Xmx4g -jar target/kmeans-1.0-SNAPSHOT-benchmarks.jar \
        -p datasetPath={{dataset_path}} \
        -jvmArgsAppend "--enable-preview" \
        -rf json \
        -rff results/micro/{{version}}.json \
        {{filter}}

bench-macro-deploy:
    mvn --quiet package -DskipTests
    cp target/kmeans-1.0-SNAPSHOT.jar /home/rarycoringa/university/concurrent/jmeter/lib/ext/

bench-macro-run version="v4":
    #!/usr/bin/env bash
    set -e
    mkdir -p results/macro/{{version}}
    for jmx in jmeter/ParallelGC-threads-*.jmx; do
        name=$(basename "$jmx" .jmx)
        echo "=== $name ==="
        GC_ALGO="-XX:+UseParallelGC" JVM_ARGS="-Xmx4g --enable-preview" \
            /home/rarycoringa/university/concurrent/jmeter/bin/jmeter \
            -n -t "$jmx" -l /dev/null
        echo "done"
    done

profile dataset_path version="v4":
    mkdir -p results/profile/{{version}}
    mvn --quiet exec:exec \
        -Dexec.executable=java \
        -Dexec.args="--enable-preview -Xmx4g -XX:StartFlightRecording=name=concurrent,settings=profile,dumponexit=true,filename=results/profile/{{version}}/concurrent.jfr -cp %classpath br.edu.ufrn.kmeans.Main {{dataset_path}}"

profile-report version="v2":
    jfr print --events jdk.ExecutionSample \
        results/profile/{{version}}/concurrent.jfr > results/profile/{{version}}/hot-methods.txt
    jfr print --events jdk.GarbageCollection,jdk.GCHeapSummary \
        results/profile/{{version}}/concurrent.jfr > results/profile/{{version}}/gc.txt
    jfr print --events jdk.JavaThreadStatistics,jdk.ThreadCPULoad \
        results/profile/{{version}}/concurrent.jfr > results/profile/{{version}}/threads.txt
    jfr print --events jdk.FileRead \
        results/profile/{{version}}/concurrent.jfr > results/profile/{{version}}/io.txt

bench-stress-build:
    mvn --quiet package -DskipTests

bench-stress version="v4" filter="JCS":
    mkdir -p results/stress/{{version}}
    java --enable-preview -jar target/kmeans-1.0-SNAPSHOT-stress.jar \
        -t {{filter}} \
        -jvmArgsPrepend "--enable-preview" \
        -r results/stress/{{version}}
