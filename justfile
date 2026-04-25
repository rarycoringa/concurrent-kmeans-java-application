clean:
    mvn --quiet clean

build:
    mvn --quiet compile

run dataset_path:
    mvn --quiet exec:java -Dexec.args="{{dataset_path}}"
