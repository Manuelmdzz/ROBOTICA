import os


print("-" * 50)
print(f"Ruta del script: {__file__}")
print(f"Ruta del script: {os.path.abspath(__file__)}")

print("#" * 50)
script_dir = os.path.dirname(os.path.abspath(__file__))
print(f"Directorio del script: {script_dir}")
print(f"Directorio actual: {os.getcwd()}")


print("-" * 50)
