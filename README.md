# 3-Stage-Pipelined-FP8-Multiplier-
# 3-Stage Pipelined FP8 Multiplier

A Python implementation of a 3-stage pipelined floating-point multiplier using 8-bit floating-point (FP8) E4M3 format, designed for efficient dot product computation.

## Overview

This project implements a hardware-style pipelined multiplier that processes FP8 numbers through three distinct pipeline stages. The design demonstrates how pipelining increases throughput in floating-point arithmetic units, commonly used in machine learning accelerators and digital signal processors.

## FP8 E4M3 Format

The implementation uses the E4M3 FP8 format:

```
Bit Layout: [S][EEEE][MMM]
            Sign (1 bit) | Exponent (4 bits) | Mantissa (3 bits)
```

| Field | Bits | Description |
|-------|------|-------------|
| Sign | 1 | 0 = positive, 1 = negative |
| Exponent | 4 | Biased exponent (bias = 7) |
| Mantissa | 3 | Fractional part (implicit leading 1) |

### Format Parameters

| Parameter | Value |
|-----------|-------|
| Total bits | 8 |
| Exponent bias | 7 |
| Max exponent | 14 |
| Min exponent | 1 |
| Mantissa precision | 3 bits |

## Pipeline Architecture

The multiplier is divided into three stages, allowing concurrent processing of multiple operands:

```
┌─────────────────────────────────────────────────────────────────────┐
│                    3-STAGE PIPELINED MULTIPLIER                     │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐           │
│  │   STAGE 1    │    │   STAGE 2    │    │   STAGE 3    │           │
│  │              │    │              │    │              │           │
│  │ • Sign XOR   │───►│ • Mantissa   │───►│ • Normalize  │───► Result│
│  │ • Exp Add    │REG │   Multiply   │REG │ • Overflow   │           │
│  │ • Pass Mant  │    │              │    │ • Pack FP8   │           │
│  └──────────────┘    └──────────────┘    └──────────────┘           │
│                                                                     │
│  Input A ───►                                                       │
│  Input B ───►                                                       │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### Stage Descriptions

| Stage | Name | Operations |
|-------|------|------------|
| Stage 1 | Sign & Exponent | XOR sign bits, Add exponents, Extract mantissas with implicit 1 |
| Stage 2 | Mantissa Multiplication | Multiply mantissas (4-bit × 4-bit = 8-bit product) |
| Stage 3 | Normalization | Normalize mantissa, Adjust exponent, Handle overflow/underflow, Pack result |

### Pipeline Timing

```
Cycle:    1    2    3    4    5    6    ...
         ┌────┬────┬────┬────┬────┬────┐
Input 0: │ S1 │ S2 │ S3 │    │    │    │ → Output 0 at Cycle 3
         ├────┼────┼────┼────┼────┼────┤
Input 1: │    │ S1 │ S2 │ S3 │    │    │ → Output 1 at Cycle 4
         ├────┼────┼────┼────┼────┼────┤
Input 2: │    │    │ S1 │ S2 │ S3 │    │ → Output 2 at Cycle 5
         └────┴────┴────┴────┴────┴────┘

Latency: 3 cycles
Throughput: 1 result/cycle (after pipeline filled)
```

## Files

| File | Description |
|------|-------------|
| `fp8_3stage_pipelined.py` | Main implementation with pipeline classes and dot product demo |
| `README.md` | This documentation file |

## Usage

### Requirements

```bash
pip install numpy
```

### Running the Demo

```bash
python fp8_3stage_pipelined.py
```

### Example Output

```
============================================================
3-STAGE PIPELINED FP8 MULTIPLIER - DOT PRODUCT
============================================================

Input Vectors (FP8 hex):
A: ['0x1d', '0x25', '0x28', '0xaa', '0x2d', '0x30', '0x31', '0x32', '0xb4', '0x35', '0x36', '0x36']
B: ['0x28', '0xb6', '0x20', '0x35', '0x36', '0xb4', '0x2a', '0x32', '0x1d', '0x25', '0xad', '0x31']

============================================================
CYCLE-BY-CYCLE PIPELINE EXECUTION
============================================================
Cycle  Stage1 Input         Stage2 (Mant Mult)   Stage3 Output   Product (dec)
------------------------------------------------------------
1      A[0]*B[0] (0x1d*0x28) Computing...         ---             ---
2      A[1]*B[1] (0x25*0xb6) Computing...         ---             ---
3      A[2]*B[2] (0x28*0x20) Computing...         0xd             0.0254
...

============================================================
FINAL RESULTS
============================================================
Dot Product (FP8 hex): 0x2b
Dot Product (decimal): 0.34375
```

## API Reference

### Core Functions

#### `float_to_fp894(value)`
Converts a Python float to FP8 E4M3 format.

```python
fp8_val = float_to_fp894(0.5)  # Returns 0x30
```

#### `fp8_to_float94(fp8)`
Converts FP8 E4M3 format to Python float.

```python
float_val = fp8_to_float94(0x30)  # Returns 0.5
```

#### `fp8_add94(a, b)`
Adds two FP8 numbers.

```python
result = fp8_add94(0x30, 0x28)  # 0.5 + 0.25 = 0.75
```

### Pipeline Classes

#### `FP8_3StagePipelinedMultiplier`
Main pipelined multiplier class.

```python
multiplier = FP8_3StagePipelinedMultiplier()

# Execute one clock cycle with new inputs
result = multiplier.clock_cycle(input_a, input_b)

# Flush remaining pipeline data
remaining_results = multiplier.flush_pipeline()
```

#### `PipelineStage1`, `PipelineStage2`, `PipelineStage3`
Individual pipeline stage classes with `compute()` methods.

## Dot Product Computation

The implementation computes the dot product of two vectors:

```
A = [0.1, 0.2, 0.25, -0.3, 0.4, 0.5, 0.55, 0.6, -0.75, 0.8, 0.875, 0.9]
B = [0.25, -0.9, 0.125, 0.8, 0.875, -0.75, 0.3, 0.6, 0.1, 0.2, -0.4, 0.55]

Dot Product = Σ(A[i] × B[i]) for i = 0 to 11
```

### Computation Flow

```
1. Convert inputs to FP8 format
2. Feed pairs through pipelined multiplier
3. Collect products (with 3-cycle latency)
4. Accumulate products using FP8 addition
5. Output final dot product result
```

## Performance Characteristics

| Metric | Value |
|--------|-------|
| Pipeline Depth | 3 stages |
| Latency | 3 clock cycles |
| Throughput | 1 multiplication/cycle |
| FP8 Dynamic Range | ~2^-6 to ~2^7 |
| Precision | 3 mantissa bits (~12.5% relative error) |

## Quantization Error

Due to the limited precision of FP8 format:

```
Direct float computation: 0.36625
FP8 result: 0.34375
Quantization error: 0.02250 (~6.1%)
```

## Applications

This design is applicable to:

- Machine learning inference accelerators
- Neural network quantization research
- Digital signal processing education
- Hardware design prototyping
- Low-power AI edge devices

## Extending the Design

### Adding More Pipeline Stages

To add finer granularity (e.g., 4 or 5 stages), split existing operations:

```python
# Example: Split Stage 2 into two sub-stages
# Stage 2a: Partial product generation
# Stage 2b: Partial product reduction
```

### Implementing in Hardware (Verilog/VHDL)

The Python classes map directly to hardware modules:

```
PipelineStage1 → stage1_sign_exp.v
PipelineStage2 → stage2_mant_mult.v
PipelineStage3 → stage3_normalize.v
```
