import zipfile

try:
    with zipfile.ZipFile('autos.zip', 'r') as zip_ref:
        zip_ref.extractall('.')
    print("Successfully extracted autos.zip")
    
    # Вывести список файлов после распаковки
    import os
    print("Files in the current directory:")
    for file in os.listdir('.'):
        if file.endswith('.csv'):
            print(f"- {file}")
except Exception as e:
    print(f"Error extracting file: {e}") 