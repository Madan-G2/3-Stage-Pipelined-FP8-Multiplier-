import numpy as np  # Add this import statement 
EXP_BITS94 = 4 
MANT_BITS94 = 3 
BIAS94 = (2**(EXP_BITS94 - 1)) - 1 
MAX_EXP94 = (2**EXP_BITS94) - 2 
MIN_EXP94 = 1 
MAX_MANT94 = 2**MANT_BITS94 
def float_to_fp894(value94): 
if value94 == 0: 
return 0x00  # Zero representation 
sign94 = 0 if value94 > 0 else 1 
value94 = abs(value94) 
exponent94 = int(np.floor(np.log2(value94))) 
mantissa94 = (value94 / (2 ** exponent94)) - 1 
# Normalize exponent with bias 
exp_bits94 = exponent94 + BIAS94 
if exp_bits94 < MIN_EXP94: 
return 0x00  # Underflow to zero 
elif exp_bits94 > MAX_EXP94: 
return (sign94 << 7) | (MAX_EXP94 << MANT_BITS94)  # Overflow to max 
# Convert mantissa to FP8 
mant_bits94 = round(mantissa94 * MAX_MANT94) 
if mant_bits94 >= MAX_MANT94: 
mant_bits94 = MAX_MANT94 - 1  # Clamp mantissa 
return (sign94 << 7) | (exp_bits94 << MANT_BITS94) | mant_bits94 
def fp8_to_float94(fp894): 
if fp894 == 0x00: 
return 0.0 
sign94 = -1 if (fp894 & 0x80) else 1 
exp_bits94 = (fp894 >> MANT_BITS94) & 0x0F 
mant_bits94 = fp894 & 0x07 
exponent94 = exp_bits94 - BIAS94 
mantissa94 = 1 + (mant_bits94 / MAX_MANT94) 
return sign94 * mantissa94 * (2 ** exponent94) 
def fp8_multiply94(aXX94, bXX94): 
a_floatXX94 = fp8_to_float94(aXX94) 
b_floatXX94 = fp8_to_float94(bXX94) 
product_floatXX94 = a_floatXX94 * b_floatXX94 
return float_to_fp894(product_floatXX94) 
def fp8_add94(aXX94, bXX94): 
sum_floatXX94 = fp8_to_float94(aXX94) + fp8_to_float94(bXX94) 
return float_to_fp894(sum_floatXX94) 
def fp8_dot_product_pipelined94(vec194, vec294): 
assert len(vec194) == len(vec294), "Vectors must be of the same length" 
pipeline_stage194 = [fp8_multiply94(a94, b94) for a94, b94 in zip(vec194, vec294)] 
pipeline_stage294 = [0x00] * len(vec194) 
pipeline_stage294[0] = pipeline_stage194[0] 
for i94 in range(1, len(vec194)): 
pipeline_stage294[i94] 
= 
pipeline_stage194[i94]) 
fp8_add94(pipeline_stage294[i94 - 
1], 
return pipeline_stage294[-1] 
# Given input vectors 
A94 = [0.1, 0.2, 0.25, -0.3, 0.4, 0.5, 0.55, 0.6, -0.75, 0.8, 0.875, 0.9] 
B94 = [0.25, -0.9, 0.125, 0.8, 0.875, -0.75, 0.3, 0.6, 0.1, 0.2, -0.4, 0.55] 
# Convert to FP8 format 
A_fp894 = [float_to_fp894(x94) for x94 in A94] 
B_fp894 = [float_to_fp894(x94) for x94 in B94] 
dot_product_fp894 = fp8_dot_product_pipelined94(A_fp894, B_fp894) 
print("Dot Product in FP8:", hex(dot_product_fp894)) 
print("Dot Product in Float:", fp8_to_float94(dot_product_fp894)) 
# Cycle-by-cycle pipelined computation 
print("\nCycle-by-cycle dot product computation:") 
pipeline_stage194 = [fp8_multiply94(A_fp894[i94], B_fp894[i94]) for i94 in 
range(len(A_fp894))] 
pipeline_stage294 = [0x00] * len(A_fp894) 
pipeline_stage294[0] = pipeline_stage194[0] 
print(f"Cycle1: 
Accumulator94={pipeline_stage294[0]:02x}") 
for i94 in range(1, len(A_fp894)): 
pipeline_stage294[i94] 
= 
Product94={pipeline_stage194[0]:02x}, 
fp8_add94(pipeline_stage294[i94 
pipeline_stage194[i94]) 
print(f"Cycle{i94+1}: - 
1], 
Product94={pipeline_stage194[i94]:02x}, 
Accumulator94={pipeline_stage294[i94]:02x}") 
print(f"\nFinal dot product result94: 0x{pipeline_stage294[-1]:02x}") 
print(f"Decimal equivalent: {fp8_to_float94(pipeline_stage294[-1]):.5f}")
