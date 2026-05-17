import os
import lit.formats

config.name = 'gicc-passes'
config.test_format = lit.formats.ShTest()
config.suffixes = ['.ll', '.cpp', '.cu']
config.test_source_root = os.path.dirname(__file__)
config.test_exec_root = os.path.join(
    os.path.dirname(config.gicc_passes_so), 'lit-output')

config.substitutions.append(('%gicc_passes_so', config.gicc_passes_so))
config.substitutions.append(('%opt', os.path.join(config.llvm_bin_dir, 'opt')))
config.substitutions.append(
    ('%FileCheck', os.path.join(config.llvm_bin_dir, 'FileCheck')))

# Tests for the two lowering passes carry `REQUIRES: gicc_lowering`;
# in ANALYZE_ONLY plugin builds the feature is absent and those tests
# are reported as UNSUPPORTED instead of failing on "unknown pass".
if getattr(config, 'gicc_have_lowering', True):
    config.available_features.add('gicc_lowering')
