JAVA_21 := "/usr/lib/jvm/java-21-openjdk-amd64"
OPENS   := "--add-opens=java.base/java.lang=ALL-UNNAMED --add-opens=java.base/java.lang.invoke=ALL-UNNAMED --add-opens=java.base/java.lang.reflect=ALL-UNNAMED --add-opens=java.base/java.io=ALL-UNNAMED --add-opens=java.base/java.net=ALL-UNNAMED --add-opens=java.base/java.nio=ALL-UNNAMED --add-opens=java.base/java.util=ALL-UNNAMED --add-opens=java.base/java.util.concurrent=ALL-UNNAMED --add-opens=java.base/java.util.concurrent.atomic=ALL-UNNAMED --add-opens=java.base/java.util.concurrent.locks=ALL-UNNAMED --add-opens=java.base/sun.nio.ch=ALL-UNNAMED --add-opens=java.base/sun.nio.cs=ALL-UNNAMED --add-opens=java.base/sun.security.action=ALL-UNNAMED --add-opens=java.base/sun.util.calendar=ALL-UNNAMED"

clean:
    mvn --quiet clean

build:
    mvn --quiet compile

# exec:java runs in Maven's own JVM — setting JAVA_HOME forces Maven to start on
# JDK 21 (the version Spark 4.0 is tested on); MAVEN_OPTS injects Xmx and the
# --add-opens flags Spark needs for internal reflection into that same JVM
run dataset_path:
    JAVA_HOME={{JAVA_21}} MAVEN_OPTS="-Xmx4g {{OPENS}}" mvn --quiet exec:java -Dexec.args="{{dataset_path}}"

profile dataset_path version="v6":
    #!/usr/bin/env bash
    set -e
    mkdir -p results/profile/{{version}}
    JFR="name=concurrent,settings=profile,dumponexit=true,filename=$(pwd)/results/profile/{{version}}/concurrent.jfr"
    JAVA_HOME={{JAVA_21}} MAVEN_OPTS="-Xmx4g {{OPENS}} -XX:StartFlightRecording=$JFR" \
        mvn --quiet exec:java -Dexec.args="{{dataset_path}}"

profile-report version="v6":
    jfr print --events jdk.ExecutionSample \
        results/profile/{{version}}/concurrent.jfr > results/profile/{{version}}/hot-methods.txt
    jfr print --events jdk.GarbageCollection,jdk.GCHeapSummary \
        results/profile/{{version}}/concurrent.jfr > results/profile/{{version}}/gc.txt
    jfr print --events jdk.JavaThreadStatistics,jdk.ThreadCPULoad \
        results/profile/{{version}}/concurrent.jfr > results/profile/{{version}}/threads.txt
    jfr print --events jdk.FileRead \
        results/profile/{{version}}/concurrent.jfr > results/profile/{{version}}/io.txt
