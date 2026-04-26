import lit.formats
import os

config.name = "gicc-clang-plugin"
config.test_format = lit.formats.ShTest(execute_external=True)
config.suffixes = [".cu", ".cpp"]
config.test_source_root = os.path.dirname(__file__)
config.test_exec_root  = os.path.join(config.gicc_obj_root, "lit")

# FileCheck lives in the ROCm LLVM bin dir, which is not on the default PATH.
config.environment["PATH"] = (
    "/opt/rocm-6.4.0/lib/llvm/bin:" + config.environment.get("PATH", "")
)

config.substitutions.append(("%clang", config.clang_executable))
config.substitutions.append(("%plugin", config.plugin_path))
config.substitutions.append(("%gicc_src", config.gicc_src_root))
