from ml.config import (
    PROJECT_ROOT,
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
    MODELS_DIR,
    FORECASTS_DIR,
    create_directories,
)


create_directories()


print("PROJECT ROOT:")
print(PROJECT_ROOT)

print()

print("RAW DATA:")
print(RAW_DATA_DIR)

print()

print("PROCESSED DATA:")
print(PROCESSED_DATA_DIR)

print()

print("MODELS:")
print(MODELS_DIR)

print()

print("FORECASTS:")
print(FORECASTS_DIR)