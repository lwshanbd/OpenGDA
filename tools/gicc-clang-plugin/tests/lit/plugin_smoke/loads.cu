// RUN: %clang -std=c++17 -fplugin=%plugin -x c++ -c %s -o /dev/null 2>&1 | FileCheck %s
// CHECK: [gicc-plugin] consumer ran
int main() { return 0; }
