from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name='cursofiap-package-gabrielle',
    version='1.0.0',
    packages=find_packages(),
    description='pacote criado para aula de python',
    author='Gabrielle M Feitosa',
    author_email='gabriellemiguel14@gmail.com',
    url='https://github.com/gabrielle19/cursofiap',  
    license='MIT',  
    long_description=long_description,
    long_description_content_type='text/markdown'
)
