import numpy as np

# FP8 E4M3 Format Constants
EXP_BITS94 = 4
MANT_BITS94 = 3
BIAS94 = (2**(EXP_BITS94 - 1)) - 1  # 7
MAX_EXP94 = (2**EXP_BITS94) - 2      # 14
MIN_EXP94 = 1
MAX_MANT94 = 2**MANT_BITS94          # 8

def float_to_fp894(value94):
    """Convert float to FP8 E4M3 format"""
    if value94 == 0:
        return 0x00
    
    sign94 = 0 if value94 > 0 else 1
    value94 = abs(value94)
    
    exponent94 = int(np.floor(np.log2(value94)))
    mantissa94 = (value94 / (2 ** exponent94)) - 1
    
    exp_bits94 = exponent94 + BIAS94
    
    if exp_bits94 < MIN_EXP94:
        return 0x00
    elif exp_bits94 > MAX_EXP94:
        return (sign94 << 7) | (MAX_EXP94 << MANT_BITS94)
    
    mant_bits94 = round(mantissa94 * MAX_MANT94)
    if mant_bits94 >= MAX_MANT94:
        mant_bits94 = MAX_MANT94 - 1
    
    return (sign94 << 7) | (exp_bits94 << MANT_BITS94) | mant_bits94

def fp8_to_float94(fp894):
    """Convert FP8 E4M3 format to float"""
    if fp894 == 0x00:
        return 0.0
    
    sign94 = -1 if (fp894 & 0x80) else 1
    exp_bits94 = (fp894 >> MANT_BITS94) & 0x0F
    mant_bits94 = fp894 & 0x07
    
    exponent94 = exp_bits94 - BIAS94
    mantissa94 = 1 + (mant_bits94 / MAX_MANT94)
    
    return sign94 * mantissa94 * (2 ** exponent94)

def extract_fp8_fields(fp894):
    """Extract sign, exponent, and mantissa fields from FP8"""
    sign = (fp894 >> 7) & 0x01
    exp = (fp894 >> MANT_BITS94) & 0x0F
    mant = fp894 & 0x07
    return sign, exp, mant

# ============================================================
# 3-STAGE PIPELINED FP8 MULTIPLIER
# ============================================================
# Stage 1: Sign calculation & Exponent addition
# Stage 2: Mantissa multiplication
# Stage 3: Normalization & Final result assembly
# ============================================================

class PipelineStage1:
    """Stage 1: Sign XOR and Exponent Addition"""
    def __init__(self):
        self.valid = False
        self.sign_result = 0
        self.exp_sum = 0
        self.mant_a = 0
        self.mant_b = 0
        self.is_zero = False
    
    def compute(self, fp8_a, fp8_b):
        sign_a, exp_a, mant_a = extract_fp8_fields(fp8_a)
        sign_b, exp_b, mant_b = extract_fp8_fields(fp8_b)
        
        self.valid = True
        self.is_zero = (fp8_a == 0x00) or (fp8_b == 0x00)
        
        # Stage 1 operations
        self.sign_result = sign_a ^ sign_b  # XOR for sign
        self.exp_sum = exp_a + exp_b        # Add exponents (will subtract bias in stage 3)
        
        # Pass mantissas to next stage (with implicit 1)
        self.mant_a = (1 << MANT_BITS94) | mant_a  # 1.xxx format (4 bits: 1 + 3)
        self.mant_b = (1 << MANT_BITS94) | mant_b
        
        return {
            'sign': self.sign_result,
            'exp_sum': self.exp_sum,
            'mant_a': self.mant_a,
            'mant_b': self.mant_b,
            'is_zero': self.is_zero
        }

class PipelineStage2:
    """Stage 2: Mantissa Multiplication"""
    def __init__(self):
        self.valid = False
        self.sign_result = 0
        self.exp_sum = 0
        self.mant_product = 0
        self.is_zero = False
    
    def compute(self, stage1_output):
        self.valid = stage1_output is not None
        if not self.valid:
            return None
        
        self.sign_result = stage1_output['sign']
        self.exp_sum = stage1_output['exp_sum']
        self.is_zero = stage1_output['is_zero']
        
        # Stage 2: Multiply mantissas
        # (1.m1) * (1.m2) = product in format xx.xxxxxx (8 bits for 4x4 multiplication)
        self.mant_product = stage1_output['mant_a'] * stage1_output['mant_b']
        
        return {
            'sign': self.sign_result,
            'exp_sum': self.exp_sum,
            'mant_product': self.mant_product,
            'is_zero': self.is_zero
        }

class PipelineStage3:
    """Stage 3: Normalization and Result Assembly"""
    def __init__(self):
        self.valid = False
        self.result = 0x00
    
    def compute(self, stage2_output):
        self.valid = stage2_output is not None
        if not self.valid:
            return None
        
        if stage2_output['is_zero']:
            self.result = 0x00
            return self.result
        
        sign = stage2_output['sign']
        exp_sum = stage2_output['exp_sum']
        mant_product = stage2_output['mant_product']
        
        # Normalize: product is in format 1x.xxxxxx or 01.xxxxxx
        # Check if we need to shift (if MSB is in bit position 7)
        if mant_product & (1 << (2 * MANT_BITS94 + 1)):  # Check bit 7
            # Product >= 2.0, need to shift right and increment exponent
            mant_normalized = mant_product >> (MANT_BITS94 + 1)
            exp_result = exp_sum - BIAS94 + 1
        else:
            # Product < 2.0, normal case
            mant_normalized = mant_product >> MANT_BITS94
            exp_result = exp_sum - BIAS94
        
        # Extract final mantissa bits (remove implicit 1)
        final_mant = mant_normalized & ((1 << MANT_BITS94) - 1)
        
        # Handle overflow/underflow
        if exp_result > MAX_EXP94:
            self.result = (sign << 7) | (MAX_EXP94 << MANT_BITS94) | 0x07  # Max value
        elif exp_result < MIN_EXP94:
            self.result = 0x00  # Underflow to zero
        else:
            self.result = (sign << 7) | (exp_result << MANT_BITS94) | final_mant
        
        return self.result

class FP8_3StagePipelinedMultiplier:
    """3-Stage Pipelined FP8 Multiplier"""
    def __init__(self):
        self.stage1 = PipelineStage1()
        self.stage2 = PipelineStage2()
        self.stage3 = PipelineStage3()
        
        # Pipeline registers
        self.reg_s1_to_s2 = None
        self.reg_s2_to_s3 = None
        
        self.cycle_count = 0
    
    def clock_cycle(self, new_input_a=None, new_input_b=None):
        """Execute one clock cycle of the pipeline"""
        self.cycle_count += 1
        
        # Stage 3: Process data from stage 2 register
        result = self.stage3.compute(self.reg_s2_to_s3)
        
        # Stage 2: Process data from stage 1 register, store to stage 2 register
        self.reg_s2_to_s3 = self.stage2.compute(self.reg_s1_to_s2)
        
        # Stage 1: Process new inputs, store to stage 1 register
        if new_input_a is not None and new_input_b is not None:
            self.reg_s1_to_s2 = self.stage1.compute(new_input_a, new_input_b)
        else:
            self.reg_s1_to_s2 = None
        
        return result
    
    def flush_pipeline(self):
        """Flush remaining data through pipeline"""
        results = []
        for _ in range(2):  # Need 2 more cycles to flush
            result = self.clock_cycle()
            if result is not None:
                results.append(result)
        return results

def fp8_add94(aXX94, bXX94):
    """FP8 Addition (non-pipelined for simplicity)"""
    sum_floatXX94 = fp8_to_float94(aXX94) + fp8_to_float94(bXX94)
    return float_to_fp894(sum_floatXX94)

# ============================================================
# MAIN: Dot Product using 3-Stage Pipelined Multiplier
# ============================================================

print("=" * 60)
print("3-STAGE PIPELINED FP8 MULTIPLIER - DOT PRODUCT")
print("=" * 60)

# Given input vectors
A94 = [0.1, 0.2, 0.25, -0.3, 0.4, 0.5, 0.55, 0.6, -0.75, 0.8, 0.875, 0.9]
B94 = [0.25, -0.9, 0.125, 0.8, 0.875, -0.75, 0.3, 0.6, 0.1, 0.2, -0.4, 0.55]

# Convert to FP8 format
A_fp894 = [float_to_fp894(x94) for x94 in A94]
B_fp894 = [float_to_fp894(x94) for x94 in B94]

print("\nInput Vectors (FP8 hex):")
print(f"A: {[hex(x) for x in A_fp894]}")
print(f"B: {[hex(x) for x in B_fp894]}")

# Create pipelined multiplier
multiplier = FP8_3StagePipelinedMultiplier()

print("\n" + "=" * 60)
print("CYCLE-BY-CYCLE PIPELINE EXECUTION")
print("=" * 60)
print(f"{'Cycle':<6} {'Stage1 Input':<20} {'Stage2 (Mant Mult)':<20} {'Stage3 Output':<15} {'Product (dec)':<12}")
print("-" * 60)

# Feed inputs and collect products
products = []
n = len(A_fp894)

for i in range(n + 2):  # n inputs + 2 cycles to flush pipeline
    if i < n:
        input_a, input_b = A_fp894[i], B_fp894[i]
        input_str = f"A[{i}]*B[{i}] ({hex(input_a)}*{hex(input_b)})"
    else:
        input_a, input_b = None, None
        input_str = "---"
    
    result = multiplier.clock_cycle(input_a, input_b)
    
    # Pipeline stage status
    s2_status = "Computing..." if multiplier.reg_s1_to_s2 else "---"
    
    if result is not None:
        products.append(result)
        result_str = hex(result)
        result_dec = f"{fp8_to_float94(result):.4f}"
    else:
        result_str = "---"
        result_dec = "---"
    
    print(f"{i+1:<6} {input_str:<20} {s2_status:<20} {result_str:<15} {result_dec:<12}")

print("\n" + "=" * 60)
print("ACCUMULATION (DOT PRODUCT SUM)")
print("=" * 60)

# Accumulate products for dot product
accumulator = 0x00
print(f"{'Step':<6} {'Product':<12} {'Accumulator':<12} {'Acc (decimal)':<15}")
print("-" * 45)

for i, prod in enumerate(products):
    accumulator = fp8_add94(accumulator, prod)
    print(f"{i+1:<6} {hex(prod):<12} {hex(accumulator):<12} {fp8_to_float94(accumulator):<15.5f}")

print("\n" + "=" * 60)
print("FINAL RESULTS")
print("=" * 60)
print(f"Dot Product (FP8 hex): {hex(accumulator)}")
print(f"Dot Product (decimal): {fp8_to_float94(accumulator):.5f}")

# Verify with direct float computation
direct_result = sum(a * b for a, b in zip(A94, B94))
print(f"\nDirect float computation: {direct_result:.5f}")
print(f"FP8 quantization error: {abs(fp8_to_float94(accumulator) - direct_result):.5f}")
