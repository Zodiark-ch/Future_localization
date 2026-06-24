BASE_DATASET_NAMES = ("bool", "gender", "ioi_mistral", "sst2")
ARITHMETIC_DATASET_NAMES = tuple(f"{digit}_digit_arithmetic" for digit in range(1, 6))
SUPPORTED_DATASET_NAMES = BASE_DATASET_NAMES + ARITHMETIC_DATASET_NAMES