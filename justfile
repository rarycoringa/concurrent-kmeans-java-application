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
