#!/bin/bash
#
#
#
# Traces:
#    streaming_phase0_core1
#
#
# Experiments:
#    nopref: --warmup_instructions=100000000 --simulation_instructions=500000000 --config=$(PYTHIA_HOME)/config/nopref.ini
#    pythia: --warmup_instructions=100000000 --simulation_instructions=500000000 --l2c_prefetcher_types=scooby --config=$(PYTHIA_HOME)/config/pythia.ini
#
#
#
#
/home/wzc/project/Pythia/bin/perceptron-multi-multi-no-ship-1core --warmup_instructions=100000000 --simulation_instructions=500000000 --config=/home/wzc/project/Pythia/config/nopref.ini --knob_cloudsuite=true --warmup_instructions=100000000 --simulation_instructions=150000000 -traces /home/wzc/project/Pythia/traces/streaming_phase0_core1.trace.xz > streaming_phase0_core1_nopref.out 2>&1
/home/wzc/project/Pythia/bin/perceptron-multi-multi-no-ship-1core --warmup_instructions=100000000 --simulation_instructions=500000000 --l2c_prefetcher_types=scooby --config=/home/wzc/project/Pythia/config/pythia.ini --knob_cloudsuite=true --warmup_instructions=100000000 --simulation_instructions=150000000 -traces /home/wzc/project/Pythia/traces/streaming_phase0_core1.trace.xz > streaming_phase0_core1_pythia.out 2>&1
