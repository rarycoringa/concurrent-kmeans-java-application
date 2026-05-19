clean:
    mvn --quiet clean

build:
    mvn --quiet compile

run dataset_path:
    mvn --quiet exec:java -Dexec.args="{{dataset_path}}"

bench-build:
    mvn --quiet package -DskipTests

bench dataset_path filter="KMeansBenchmark":
    mkdir -p results
    java -jar target/kmeans-1.0-SNAPSHOT-benchmarks.jar \
        -p datasetPath={{dataset_path}} \
        -rf json \
        -rff results/jmh-results.json \
        {{filter}}
