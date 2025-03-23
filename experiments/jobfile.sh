#!/bin/bash
#
#
#
# Traces:
#    streaming_phase0_core1
#    628.pop2_s-17B
#
#
# Experiments:
#    nnp: --warmup_instructions=100000000 --simulation_instructions=500000000 --l2c_prefetcher_types=scooby --config=$(PYTHIA_HOME)/config/pythia_nn.ini
#
#
#
#
/home/wzc/project/Pythia/bin/perceptron-multi-multi-no-ship-1core --warmup_instructions=100000000 --simulation_instructions=500000000 --l2c_prefetcher_types=scooby --config=/home/wzc/project/Pythia/config/pythia_nn.ini --knob_cloudsuite=true --warmup_instructions=100000000 --simulation_instructions=150000000 -traces /home/wzc/project/Pythia/traces/streaming_phase0_core1.trace.xz > streaming_phase0_core1_nnp.out 2>&1
/home/wzc/project/Pythia/bin/perceptron-multi-multi-no-ship-1core --warmup_instructions=100000000 --simulation_instructions=500000000 --l2c_prefetcher_types=scooby --config=/home/wzc/project/Pythia/config/pythia_nn.ini  -traces /home/wzc/project/Pythia/traces/628.pop2_s-17B.champsimtrace.xz > 628.pop2_s-17B_nnp.out 2>&1
